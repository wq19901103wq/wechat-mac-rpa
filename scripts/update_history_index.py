#!/usr/bin/env python3
"""增量更新历史消息语义索引（BGE ONNX 编码）。

支持三种模式：
1. 全量重建（默认）：从 input-dir 读取所有 JSON，重建完整 pkl
2. 单条添加/更新：--add-one JSON
3. 单条删除：--remove-one MSG_ID

用法:
    python3 scripts/update_history_index.py
    python3 scripts/update_history_index.py --input-dir data/exports/b
    python3 scripts/update_history_index.py --output data/memory/cache/vector_index_dense_messages.pkl
    python3 scripts/update_history_index.py --add-one '{"id":"...","text":"..."}'
    python3 scripts/update_history_index.py --remove-one "xxx"
"""

import argparse
import hashlib
import json
import os
import pickle  # nosec B403
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# 复用项目内已验证的 ONNX 编码器
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.memory.history_search import _BGEEncoder  # noqa: E402

DEFAULT_INPUT_DIR = Path(__file__).parent.parent / "data" / "exports" / "b"
DEFAULT_OUTPUT = (
    Path(__file__).parent.parent / "data" / "memory" / "cache" / "vector_index_dense_messages.pkl"
)
DEFAULT_MODEL = Path(
    os.environ.get(
        "WECHAT_BGE_MODEL_PATH",
        Path(__file__).parent.parent / "models" / "bge-small-zh-v1.5",
    )
)
DEFAULT_BATCH_SIZE = 512
DEFAULT_MAX_LENGTH = 200
CONTEXT_WINDOW = 5


# ── 编码 ──


def create_encoder(model_path: Path, max_length: int = DEFAULT_MAX_LENGTH) -> _BGEEncoder:
    """创建 ONNX 编码器，并覆盖 tokenizer 的 max_length。"""
    encoder = _BGEEncoder(model_path, "onnx")
    # _BGEEncoder 内部已固定 max_length=200；如需覆盖可在此扩展
    return encoder


# ── 消息加载 ──


def _content_hash(text: str) -> str:
    # SHA1 仅用于内容变更检测，非安全场景；nosec B324
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]  # nosec B324


def _chat_type_from_name(file_name: str) -> str:
    return "single" if "私聊" in file_name or "曾经的好友" in file_name else "group"


def load_messages(input_dir: Path) -> List[Dict[str, Any]]:
    """从 input-dir 加载所有文本消息。"""
    all_messages: List[Dict[str, Any]] = []
    json_files = sorted(input_dir.glob("*.json"))
    print(f"[DenseMsg] 发现 {len(json_files)} 个 JSON 文件")

    for json_file in json_files:
        if "_unmapped" in json_file.name:
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  跳过 {json_file.name}: {e}")
            continue

        msgs = data.get("messages", [])
        chat_name = data.get("session", {}).get("title", json_file.stem)
        chat_type = _chat_type_from_name(json_file.name)

        for i, m in enumerate(msgs):
            text = m.get("content", "") or ""
            if not text or len(text.strip()) < 2:
                continue
            if m.get("type") not in ("文本消息",):
                continue

            pmid = m.get("platformMessageId")
            msg_id = pmid if pmid else f"{json_file.stem}_{i}"
            sender = m.get("senderDisplayName", "未知")
            is_self = m.get("isSend", 0) == 1
            all_messages.append(
                {
                    "id": msg_id,
                    "text": text.strip(),
                    "sender": sender,
                    "is_self": is_self,
                    "chat_name": chat_name,
                    "chat_type": chat_type,
                    "timestamp": m.get("createTime", 0),
                    "file": str(json_file),
                    "index_in_file": i,
                    "content_hash": _content_hash(text.strip()),
                }
            )

    return all_messages


# ── 索引结构构建 ──


def build_context_ids(messages: List[Dict[str, Any]]) -> None:
    """按 file 分组计算前后 CONTEXT_WINDOW 条上下文。"""
    by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in messages:
        by_file[m["file"]].append(m)

    for file_msgs in by_file.values():
        file_msgs.sort(key=lambda x: x["index_in_file"])
        for i, m in enumerate(file_msgs):
            start = max(0, i - CONTEXT_WINDOW)
            end = min(len(file_msgs), i + CONTEXT_WINDOW + 1)
            m["context_ids"] = [file_msgs[j]["id"] for j in range(start, end)]


def build_indexes(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """从 messages 构建所有辅助索引。"""
    build_context_ids(messages)

    sender_index: Dict[str, List[str]] = defaultdict(list)
    chat_type_index: Dict[str, List[str]] = defaultdict(list)
    for m in messages:
        sender_index[m["sender"]].append(m["id"])
        chat_type_index[m["chat_type"]].append(m["id"])

    return {
        "messages": messages,
        "msg_by_id": {m["id"]: m for m in messages},
        "id_to_idx": {m["id"]: i for i, m in enumerate(messages)},
        "sender_index": dict(sender_index),
        "chat_type_index": dict(chat_type_index),
    }


def save_index(
    output: Path,
    embeddings: np.ndarray,
    messages: List[Dict[str, Any]],
    model_name: str,
    version: str = "message_level_v2_dense",
) -> None:
    """保存完整索引 pkl。"""
    indexes = build_indexes(messages)

    # 行归一化兜底
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = (embeddings / norms).astype(np.float32)

    data = {
        "model_name": model_name,
        "dim": embeddings.shape[1],
        "count": len(messages),
        "embeddings": embeddings,
        "version": version,
        **indexes,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        bak = output.with_suffix(".pkl.bak")
        output.rename(bak)
        print(f"[DenseMsg] 备份旧索引到: {bak}")

    with open(output, "wb") as f:
        pickle.dump(data, f)  # nosec B301


# ── 全量重建 ──


def do_full_build(
    input_dir: Path,
    output: Path,
    model_path: Path,
    batch_size: int,
    max_length: int,
) -> int:
    """全量重建索引。"""
    print(f"[DenseMsg] 加载 ONNX 编码器: {model_path}")
    encoder = create_encoder(model_path, max_length)

    print(f"[DenseMsg] 加载消息: {input_dir}")
    messages = load_messages(input_dir)
    total = len(messages)
    if total == 0:
        print("[DenseMsg] 没有可索引的消息")
        return 0
    print(f"[DenseMsg] 共 {total} 条消息")

    # 按长度排序减少 padding
    order = sorted(range(total), key=lambda i: len(messages[i]["text"]))
    sorted_texts = [messages[i]["text"] for i in order]

    embeddings = np.zeros((total, 512), dtype=np.float32)
    t0_total = time.time()

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = sorted_texts[start:end]
        t0 = time.time()
        emb = encoder.encode(batch)
        elapsed = time.time() - t0

        for j, emb_row in enumerate(emb):
            embeddings[order[start + j]] = emb_row

        speed = len(batch) / elapsed if elapsed > 0 else 0
        progress = end / total * 100
        remaining = (total - end) / speed / 60 if speed > 0 else 0
        print(
            f"  [{end}/{total}] ({progress:.1f}%) | "
            f"速度: {speed:.1f}条/秒 | 预计剩余: {remaining:.1f}分钟",
            flush=True,
        )

    print(f"[DenseMsg] 编码完成: {embeddings.shape}")

    save_index(output, embeddings, messages, str(model_path))

    elapsed_total = time.time() - t0_total
    fsize = output.stat().st_size / 1024 / 1024
    print(f"[DenseMsg] 保存到: {output}")
    print(f"[DenseMsg] 文件大小: {fsize:.1f} MB")
    print(f"[DenseMsg] 总耗时: {elapsed_total/60:.1f} 分钟")
    print(f"[DenseMsg] 平均速度: {total/elapsed_total:.1f} 条/秒")
    return 0


# ── 单条操作 ──


def load_existing_index(output: Path) -> Optional[Dict[str, Any]]:
    if not output.exists():
        return None
    print(f"[DenseMsg] 加载现有索引: {output}")
    with open(output, "rb") as f:
        # 索引由本地自产，非不可信来源；nosec B301
        return pickle.load(f)  # nosec B301


def normalize_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    """确保单条消息字段完整。"""
    required = ["id", "text"]
    for key in required:
        if key not in msg or not msg[key]:
            raise ValueError(f"消息缺少必要字段: {key}")

    text = str(msg["text"]).strip()
    if len(text) < 2:
        raise ValueError("消息 text 太短")

    normalized = {
        "id": str(msg["id"]),
        "text": text,
        "sender": str(msg.get("sender", "未知")),
        "is_self": bool(msg.get("is_self", False)),
        "chat_name": str(msg.get("chat_name", "未知")),
        "chat_type": str(msg.get("chat_type", "single")),
        "timestamp": int(msg.get("timestamp", 0)),
        "file": str(msg.get("file", "")),
        "index_in_file": int(msg.get("index_in_file", 0)),
        "content_hash": _content_hash(text),
    }
    return normalized


def do_add_one(
    output: Path,
    model_path: Path,
    msg_json: str,
    max_length: int,
) -> int:
    """添加或更新单条消息到 pkl。"""
    msg = normalize_message(json.loads(msg_json))
    data = load_existing_index(output)

    encoder = create_encoder(model_path, max_length)

    if data is None:
        # 没有旧索引，直接新建
        emb = encoder.encode([msg["text"]])
        save_index(output, emb, [msg], str(model_path))
        print(f"[DenseMsg] 已创建新索引并添加消息: {msg['id']}")
        return 0

    messages: List[Dict[str, Any]] = list(data["messages"])
    embeddings: np.ndarray = data["embeddings"].astype(np.float32)

    # 如果已存在，先删除旧记录
    if msg["id"] in data.get("msg_by_id", {}):
        old_idx = data["id_to_idx"][msg["id"]]
        messages.pop(old_idx)
        embeddings = np.delete(embeddings, old_idx, axis=0)
        print(f"[DenseMsg] 更新已有消息: {msg['id']}")
    else:
        print(f"[DenseMsg] 添加新消息: {msg['id']}")

    # 编码并追加
    emb = encoder.encode([msg["text"]])
    embeddings = np.vstack([embeddings, emb])
    messages.append(msg)

    save_index(output, embeddings, messages, data.get("model_name", str(model_path)))
    print(f"[DenseMsg] 已保存到: {output}")
    return 0


def do_remove_one(
    output: Path,
    msg_id: str,
) -> int:
    """从 pkl 删除单条消息。"""
    data = load_existing_index(output)
    if data is None:
        print(f"[DenseMsg] 索引不存在: {output}")
        return 1

    if msg_id not in data.get("msg_by_id", {}):
        print(f"[DenseMsg] 消息不存在: {msg_id}")
        return 1

    messages: List[Dict[str, Any]] = list(data["messages"])
    embeddings: np.ndarray = data["embeddings"].astype(np.float32)

    idx = data["id_to_idx"][msg_id]
    messages.pop(idx)
    embeddings = np.delete(embeddings, idx, axis=0)

    save_index(
        output,
        embeddings,
        messages,
        data.get("model_name", "unknown"),
    )
    print(f"[DenseMsg] 已删除消息: {msg_id}")
    return 0


# ── CLI ──


def main() -> int:
    ap = argparse.ArgumentParser(description="更新历史消息语义索引（ONNX 后端）")
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"消息 JSON 导出目录 (默认: {DEFAULT_INPUT_DIR})",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"输出 pkl 路径 (默认: {DEFAULT_OUTPUT})",
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"BGE 模型本地路径 (默认: {DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"编码批次大小 (默认: {DEFAULT_BATCH_SIZE})",
    )
    ap.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"单条文本最大 token 长度 (默认: {DEFAULT_MAX_LENGTH})",
    )
    ap.add_argument(
        "--add-one",
        metavar="JSON",
        help='添加/更新单条消息，JSON 字符串，例如：\'{"id":"x","text":"y"}\'',
    )
    ap.add_argument(
        "--remove-one",
        metavar="MSG_ID",
        help="删除指定 msg_id 的消息",
    )
    args = ap.parse_args()

    if args.add_one:
        return do_add_one(args.output, args.model, args.add_one, args.max_length)
    if args.remove_one:
        return do_remove_one(args.output, args.remove_one)

    # 默认全量重建
    if not args.input_dir.exists():
        print(f"[DenseMsg] 输入目录不存在: {args.input_dir}")
        return 1
    return do_full_build(
        args.input_dir,
        args.output,
        args.model,
        args.batch_size,
        args.max_length,
    )


if __name__ == "__main__":
    raise SystemExit(main())
