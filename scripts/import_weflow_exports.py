#!/usr/bin/env python3
"""
从 WeFlow 导出目录解析聊天记录，写入分片存储。
所有消息默认 replied=True（避免 Bot 疯狂回复历史消息）。

用法:
    python3 scripts/import_weflow_exports.py
"""

import json
import sys

from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base import ChatMessage, SenderType
from src.session.global_store import _msg_id
from src.utils.chat_utils import _safe_filename

_WEFLOW_TYPE_MAP = {
    "文本消息": "text",
    "图片消息": "image",
    "视频消息": "video",
    "语音消息": "voice",
    "动画表情": "sticker",
    "引用消息": "text",
    "系统消息": "system",
    "位置消息": "location",
    "名片消息": "contact",
    "通话消息": "call",
    "其他消息": "other",
}


def parse_weflow_json(json_path: Path, account: str):
    """解析单个 WeFlow 导出文件，返回 (chat_name, [ChatMessage])。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    session = data.get("session", {})
    # chat_name 优先用 nickname，fallback 到 wxid
    chat_name = session.get("nickname") or session.get("displayName") or session.get("wxid") or json_path.stem

    messages = []
    for m in data.get("messages", []):
        sender = m.get("senderDisplayName") or m.get("senderUsername") or ""
        is_send = bool(m.get("isSend"))
        msg_type = _WEFLOW_TYPE_MAP.get(m.get("type", "文本消息"), "text")

        msg = ChatMessage(
            text=m.get("content") or "",
            sender=sender,
            sender_type=SenderType.SELF if is_send else SenderType.OTHER,
            chat_name=chat_name,
            is_at_me=False,
            replied=True,              # 🔒 安全：所有历史消息标记为已回复
            reply_text="",
            reply_time=None,
            message_type=msg_type,
            image_description="",
            image_text="",
            account=account,
            local_id=m.get("localId"),
            server_id=str(m.get("platformMessageId")) if m.get("platformMessageId") else None,
            create_time=m.get("createTime"),
            raw_type=m.get("localType"),
            sender_wxid=m.get("senderUsername"),
        )
        messages.append(msg)

    return chat_name, messages


def main():
    base_dir = Path("data")
    exports_dir = base_dir / "exports"
    chats_dir = base_dir / "chats"
    chats_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("WeFlow 导出 → 分片存储")
    print("=" * 50)

    # 收集所有导出文件
    sources = []
    for account, subdir in [("main", "main"), ("b", "b")]:
        src_dir = exports_dir / subdir
        if src_dir.exists():
            files = list(src_dir.glob("*.json"))
            sources.append((account, files))
            print(f"[{account}] {len(files)} 个导出文件")

    if not any(files for _, files in sources):
        print("错误: 没有找到导出文件")
        sys.exit(1)

    # 第一遍：解析所有文件，按 chat_name 聚合
    print("\n[Parse] 解析 WeFlow 导出...")
    chat_messages = defaultdict(list)  # chat_name -> [ChatMessage]
    total_raw = 0

    for account, files in sources:
        for json_path in files:
            try:
                chat_name, msgs = parse_weflow_json(json_path, account)
                chat_messages[chat_name].extend(msgs)
                total_raw += len(msgs)
            except Exception as e:
                print(f"  ❌ {json_path.name}: {e}")

    print(f"[Parse] 共 {len(chat_messages)} 个聊天，{total_raw} 条原始消息")

    # 第二遍：对每个聊天去重 + 排序 + 写入分片
    print("\n[Shard] 写入分片文件...")
    index = {"version": 2, "format": "sharded", "chats": {}}
    total_dedup = 0

    for chat_name, msgs in chat_messages.items():
        # 按时间排序
        msgs.sort(key=lambda m: (m.create_time or 0, m.local_id or 0))

        # 去重：基于 msg_id（chat_name + sender + content_hash）
        seen_ids = set()
        unique = []
        for msg in msgs:
            mid = _msg_id(chat_name, msg)
            if mid not in seen_ids:
                seen_ids.add(mid)
                unique.append(msg)

        dedup_count = len(unique)
        total_dedup += dedup_count

        # 写入分片文件
        safe_name = _safe_filename(chat_name)
        shard_file = chats_dir / f"{safe_name}.json"
        shard_data = {
            "chat_id": f"chat_{hash(chat_name) & 0x7fffffff:08x}",
            "chat_name": chat_name,
            "messages": [
                {
                    "text": m.text,
                    "sender": m.sender,
                    "sender_type": m.sender_type.value,
                    "chat_name": m.chat_name,
                    "is_at_me": m.is_at_me,
                    "replied": m.replied,
                    "reply_text": m.reply_text,
                    "reply_time": m.reply_time,
                    "message_type": m.message_type,
                    "image_description": m.image_description,
                    "image_text": m.image_text,
                    "is_image_duplicate": m.is_image_duplicate,
                    "account": m.account,
                    "local_id": m.local_id,
                    "server_id": m.server_id,
                    "create_time": m.create_time,
                    "raw_type": m.raw_type,
                    "sender_wxid": m.sender_wxid,
                }
                for m in unique
            ],
        }

        tmp = shard_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(shard_data, f, ensure_ascii=False, indent=2)
        tmp.replace(shard_file)

        index["chats"][chat_name] = {
            "chat_id": shard_data["chat_id"],
            "chat_name": chat_name,
            "msg_count": dedup_count,
            "file": f"chats/{safe_name}.json",
        }

        if dedup_count > 1000:
            print(f"  {chat_name}: {dedup_count} 条")

    # 写入索引
    index_file = chats_dir / "index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 50}")
    print(f"完成！")
    print(f"  聊天数: {len(chat_messages)}")
    print(f"  原始消息: {total_raw}")
    print(f"  去重后: {total_dedup}")
    print(f"  索引: {index_file}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
