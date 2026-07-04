#!/usr/bin/env python3
"""把历史聊天记录导入 SQLite 数据库。

用法:
    python scripts/db/migrate_exports_to_db.py
    python scripts/db/migrate_exports_to_db.py --dry-run
    python scripts/db/migrate_exports_to_db.py --db-path data/db/chat_history.db

安全:
    - 迁移前自动备份目标数据库（如果存在）
    - 自动备份 data/chats/ 目录
    - 使用 INSERT OR IGNORE 避免重复
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db import ChatHistoryRepository, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_logger = logging.getLogger("migrate_exports_to_db")

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "db" / "chat_history.db"
EXPORT_DIRS = [
    PROJECT_ROOT / "data" / "exports" / "main",
    PROJECT_ROOT / "data" / "exports" / "b",
]
CHATS_DIR = PROJECT_ROOT / "data" / "chats"


def _backup_if_exists(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if path.is_file():
        bak = path.with_suffix(f"{path.suffix}.bak.{ts}")
        shutil.copy2(path, bak)
    else:
        bak = Path(f"{path}.bak.{ts}")
        shutil.copytree(path, bak)
    _logger.info("[backup] 已备份 %s -> %s", path, bak)
    return bak


def _parse_export_json(file_path: Path) -> Optional[Dict]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        _logger.warning("[parse] 读取失败 %s: %s", file_path, e)
        return None


def _convert_export_message(m: Dict, source_file: str) -> Optional[Dict]:
    """把 WeChat 导出格式转成 repository 接受的字典。"""
    content = m.get("content")
    if not content:
        return None
    msg_type = "text"
    local_type = m.get("localType")
    if local_type == 47:
        msg_type = "emoji"
    elif local_type == 3:
        msg_type = "image"
    elif local_type == 34:
        msg_type = "voice"
    elif local_type == 43:
        msg_type = "video"
    elif local_type == 10000:
        msg_type = "system"

    return {
        "content": content,
        "local_id": m.get("localId"),
        "server_id": m.get("platformMessageId"),
        "wxid": m.get("senderUsername"),
        "sender_display_name": m.get("senderDisplayName"),
        "is_self": bool(m.get("isSend")),
        "message_type": msg_type,
        "create_time": m.get("createTime"),
        "raw_type": local_type,
        "source_file": source_file,
    }


def _convert_globalstore_message(m: Dict, source_file: Optional[str] = None) -> Optional[Dict]:
    """把 GlobalStore 分片格式转成 repository 接受的字典。"""
    content = m.get("text")
    if not content:
        return None
    return {
        "content": content,
        "local_id": m.get("local_id"),
        "server_id": m.get("server_id"),
        "wxid": m.get("sender_wxid") or m.get("sender"),
        "sender_display_name": m.get("sender"),
        "is_self": m.get("sender_type") == "self",
        "message_type": m.get("message_type", "text"),
        "is_at_me": m.get("is_at_me", False),
        "replied": m.get("replied", False),
        "reply_text": m.get("reply_text"),
        "reply_time": m.get("reply_time"),
        "create_time": m.get("create_time"),
        "raw_type": m.get("raw_type"),
        "source_file": source_file,
    }


def migrate_export_file(repo: ChatHistoryRepository, file_path: Path, dry_run: bool = False) -> Dict:
    """迁移单个导出文件。"""
    data = _parse_export_json(file_path)
    if data is None:
        return {"chatrooms": 0, "messages": 0, "members": 0, "skipped": 0}

    session = data.get("session", {})
    chatroom_id = session.get("wxid")
    chat_name = session.get("nickname") or session.get("displayName") or file_path.stem
    chat_type = "group" if "@chatroom" in (chatroom_id or "") else "single"

    if not chatroom_id:
        _logger.warning("[migrate] %s 缺少 wxid，跳过", file_path)
        return {"chatrooms": 0, "messages": 0, "members": 0, "skipped": 0}

    raw_messages = data.get("messages", [])
    messages = []
    for m in raw_messages:
        converted = _convert_export_message(m, str(file_path))
        if converted:
            messages.append(converted)

    if dry_run:
        _logger.info("[dry-run] %s: %d 条消息将导入 %s", file_path, len(messages), chatroom_id)
        return {"chatrooms": 1, "messages": len(messages), "members": 0, "skipped": 0}

    stats = repo.bulk_sync_chat(
        chatroom_id=chatroom_id,
        display_name=chat_name,
        chat_type=chat_type,
        messages=messages,
        source_file=str(file_path),
    )
    return stats


def migrate_chats_dir(repo: ChatHistoryRepository, chats_dir: Path, dry_run: bool = False) -> Dict:
    """迁移 GlobalStore 分片目录。"""
    total = {"chatrooms": 0, "messages": 0, "members": 0, "skipped": 0}
    index_file = chats_dir / "index.json"
    if not index_file.exists():
        _logger.info("[migrate] %s/index.json 不存在，跳过", chats_dir)
        return total

    try:
        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)
    except Exception as e:
        _logger.warning("[migrate] 读取索引失败: %s", e)
        return total

    for chat_name, meta in index.get("chats", {}).items():
        file_name = meta.get("file", "")
        # index 里 file 可能是 "chats/xxx.json" 或 "xxx.json"
        if file_name.startswith("chats/"):
            file_path = chats_dir.parent / file_name
        else:
            file_path = chats_dir / file_name
        if not file_path.exists():
            _logger.debug("[migrate] 分片文件不存在: %s", file_path)
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            _logger.warning("[migrate] 读取 %s 失败: %s", file_path, e)
            continue

        # GlobalStore 分片没有 chatroom_id，用 chat_name 作为临时 chatroom_id
        # （前缀 _name_ 表示待解析，避免与导出文件中的真实 chatroom_id 冲突）
        chatroom_id = data.get("chatroom_id") or f"_name_{chat_name}"
        chat_type = "group" if data.get("is_group") else "single"
        raw_messages = data.get("messages", [])
        messages = [_convert_globalstore_message(m, str(file_path)) for m in raw_messages]
        messages = [m for m in messages if m]

        if dry_run:
            _logger.info("[dry-run] %s: %d 条消息将导入 %s", file_path, len(messages), chatroom_id)
            total["chatrooms"] += 1
            total["messages"] += len(messages)
            continue

        try:
            stats = repo.bulk_sync_chat(
                chatroom_id=chatroom_id,
                display_name=chat_name,
                chat_type=chat_type,
                messages=messages,
                source_file=str(file_path),
            )
            for k in total:
                total[k] += stats.get(k, 0)
        except Exception as e:
            _logger.error("[migrate] %s 同步失败: %s", chat_name, e)

    return total


def main():
    parser = argparse.ArgumentParser(description="导入历史聊天记录到 SQLite")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="目标数据库路径")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写入")
    parser.add_argument("--skip-backup", action="store_true", help="跳过备份（不推荐）")
    args = parser.parse_args()

    _logger.info("[start] 开始迁移聊天记录到 %s", args.db_path)

    # 强制备份
    if not args.dry_run and not args.skip_backup:
        _backup_if_exists(args.db_path)
        _backup_if_exists(CHATS_DIR)

    # 初始化数据库
    init_db(args.db_path)
    repo = ChatHistoryRepository(db_path=args.db_path)

    total = {"chatrooms": 0, "messages": 0, "members": 0, "skipped": 0}

    # 1. 迁移导出文件
    for export_dir in EXPORT_DIRS:
        if not export_dir.exists():
            continue
        for file_path in sorted(export_dir.glob("*.json")):
            stats = migrate_export_file(repo, file_path, dry_run=args.dry_run)
            for k in total:
                total[k] += stats.get(k, 0)

    # 2. 迁移 GlobalStore 分片
    if CHATS_DIR.exists():
        stats = migrate_chats_dir(repo, CHATS_DIR, dry_run=args.dry_run)
        for k in total:
            total[k] += stats.get(k, 0)

    _logger.info("[done] 迁移完成: %s", total)
    if not args.dry_run:
        db_stats = repo.get_stats()
        _logger.info("[done] 数据库当前状态: %s", db_stats)


if __name__ == "__main__":
    main()
