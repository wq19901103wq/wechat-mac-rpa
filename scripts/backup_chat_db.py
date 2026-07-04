#!/usr/bin/env python3
"""聊天记录数据库备份与清理。

用法:
    python scripts/backup_chat_db.py
    python scripts/backup_chat_db.py --retention 7
    python scripts/backup_chat_db.py --db-path data/db/chat_history.db --backup-dir backups/chat_db
"""

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_logger = logging.getLogger("backup_chat_db")


def backup_db(db_path: Path, backup_dir: Path) -> Path:
    """创建数据库文件备份。"""
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"chat_history.db.bak.{ts}"
    shutil.copy2(db_path, backup_path)
    _logger.info("[backup] %s -> %s", db_path, backup_path)
    return backup_path


def cleanup_old_backups(backup_dir: Path, retention: int) -> int:
    """清理超过保留数的旧备份。"""
    backups = sorted(backup_dir.glob("chat_history.db.bak.*"))
    if len(backups) <= retention:
        return 0
    to_remove = backups[:-retention]
    removed = 0
    for bak in to_remove:
        try:
            bak.unlink()
            _logger.info("[cleanup] 删除旧备份: %s", bak)
            removed += 1
        except Exception as e:
            _logger.warning("[cleanup] 删除失败 %s: %s", bak, e)
    return removed


def main():
    parser = argparse.ArgumentParser(description="备份聊天记录数据库")
    parser.add_argument("--db-path", type=Path, default=Path("data/db/chat_history.db"))
    parser.add_argument("--backup-dir", type=Path, default=Path("backups/chat_db"))
    parser.add_argument("--retention", type=int, default=7, help="保留最近 N 个备份")
    args = parser.parse_args()

    if not args.db_path.exists():
        _logger.warning("[skip] 数据库不存在: %s", args.db_path)
        sys.exit(0)

    backup_path = backup_db(args.db_path, args.backup_dir)
    removed = cleanup_old_backups(args.backup_dir, args.retention)

    _logger.info("[done] 新备份: %s, 清理旧备份: %d 个", backup_path.name, removed)


if __name__ == "__main__":
    main()
