#!/usr/bin/env python3
"""增量更新历史消息语义索引（BGE 编码）。

把 wechat-mac-rpa/data/exports/b 下的 JSON 消息导出文件编码成向量索引，
供 src/memory/history_search.py 和 src/memory/history_lookup.py 消费。

用法:
    python3 scripts/update_history_index.py
    python3 scripts/update_history_index.py --input-dir data/exports/b
    python3 scripts/update_history_index.py --output data/memory/cache/vector_index_dense_messages.pkl
    python3 scripts/update_history_index.py --model BAAI/bge-small-zh-v1.5 --batch-size 256
"""

import argparse
import json
import pickle  # nosec B403
import time
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
from typing import Any

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError as e:
    raise ImportError(
        "需要安装 torch 和 transformers: pip install torch transformers"
    ) from e


DEFAULT_INPUT_DIR = Path(__file__).parent.parent / "data" / "exports" / "b"
DEFAULT_OUTPUT = (
    Path(__file__).parent.parent / "data" / "memory" / "cache" / "vector_index_dense_messages.pkl"
)
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_BATCH_SIZE = 512
DEFAULT_MAX_LENGTH = 200


def encode_texts(tokenizer, model, texts, device, max_length: int = 200):
    """用 BGE 编码一批文本，返回 L2 归一化后的 numpy 向量。"""
    if not texts:
        return np.array([])
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        output = model(**encoded)
        emb = output.last_hidden_state[:, 0]
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.numpy()


def load_messages(input_dir: Path):
    """从 data/exports/b 加载所有文本消息。"""
    all_messages = []
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
        chat_type = (
            "single"
            if "私聊" in json_file.name or "曾经的好友" in json_file.name
            else "group"
        )

        for i, m in enumerate(msgs):
            text = m.get("content", "") or ""
            if not text or len(text.strip()) < 2:
                continue
            if m.get("type") not in ("文本消息",):
                continue

            sender = m.get("senderDisplayName", "未知")
            is_self = m.get("isSend", 0) == 1
            all_messages.append(
                {
                    "id": f"{json_file.stem}_{i}",
                    "text": text.strip(),
                    "sender": sender,
                    "is_self": is_self,
                    "chat_name": chat_name,
                    "chat_type": chat_type,
                    "timestamp": m.get("createTime", 0),
                    "file": str(json_file),
                    "index_in_file": i,
                }
            )

    return all_messages


def main() -> int:
    ap = argparse.ArgumentParser(description="更新历史消息语义索引")
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
        default=DEFAULT_MODEL,
        help=f"BGE 模型名称或本地路径 (默认: {DEFAULT_MODEL})",
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
    args = ap.parse_args()

    if not args.input_dir.exists():
        print(f"[DenseMsg] 输入目录不存在: {args.input_dir}")
        return 1

    print(f"[DenseMsg] 加载 BGE 模型: {args.model}")
    # 默认模型为公开已知模型；生产环境建议用 --model 指定本地路径或 pinned revision
    tokenizer = AutoTokenizer.from_pretrained(args.model)  # nosec B615
    model = AutoModel.from_pretrained(args.model)  # nosec B615
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    print(f"[DenseMsg] 加载消息: {args.input_dir}")
    all_messages = load_messages(args.input_dir)
    total = len(all_messages)
    if total == 0:
        print("[DenseMsg] 没有可索引的消息")
        return 0
    print(f"[DenseMsg] 共 {total} 条消息")

    # 按长度排序，减少 padding 开销
    all_messages.sort(key=lambda m: len(m["text"]))

    embeddings: List[Any] = []
    t0_total = time.time()

    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        batch = [m["text"] for m in all_messages[start:end]]

        t0 = time.time()
        emb = encode_texts(tokenizer, model, batch, device, args.max_length)
        elapsed = time.time() - t0

        embeddings.append(emb)
        speed = len(batch) / elapsed
        progress = end / total * 100
        remaining = (total - end) / speed / 60 if speed > 0 else 0

        print(
            f"  [{end}/{total}] ({progress:.1f}%) | "
            f"速度: {speed:.1f}条/秒 | 预计剩余: {remaining:.1f}分钟",
            flush=True,
        )

    stacked_embeddings = np.vstack(embeddings)
    print(f"[DenseMsg] 编码完成: {stacked_embeddings.shape}")

    # 构建辅助索引（存消息 id 字符串，与 history_search.py 匹配）
    sender_index = defaultdict(list)
    chat_type_index = defaultdict(list)
    for m in all_messages:
        sender_index[m["sender"]].append(m["id"])
        chat_type_index[m["chat_type"]].append(m["id"])

    data = {
        "model_name": args.model,
        "dim": stacked_embeddings.shape[1],
        "count": total,
        "embeddings": stacked_embeddings.astype(np.float32),
        "messages": all_messages,
        "sender_index": dict(sender_index),
        "chat_type_index": dict(chat_type_index),
        "version": "message_level_v1",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data, f)  # nosec B301

    elapsed_total = time.time() - t0_total
    print(f"[DenseMsg] 保存到: {args.output}")
    print(f"[DenseMsg] 文件大小: {args.output.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"[DenseMsg] 总耗时: {elapsed_total/60:.1f} 分钟")
    print(f"[DenseMsg] 平均速度: {total/elapsed_total:.1f} 条/秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
