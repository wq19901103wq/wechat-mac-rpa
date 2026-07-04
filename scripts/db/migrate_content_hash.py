#!/usr/bin/env python3
"""把消息表 content_hash 从 MD5 迁移到 SHA256。

背景：CodeQL 对 MD5 告警，repository.py 已改为 sha256。
运行本脚本可更新现有记录的 content_hash，避免后续 bulk_sync 出现重复。
"""

import hashlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select

from src.db import init_db
from src.db.connection import get_engine
from src.db.models import Message

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    db_path = Path("data/db/chat_history.db")
    if not db_path.exists():
        _logger.error("数据库不存在: %s", db_path)
        sys.exit(1)

    init_db(db_path)
    engine = get_engine(db_path)
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        stmt = select(Message).order_by(Message.id)
        total = 0
        updated = 0
        for msg in session.execute(stmt).scalars().yield_per(1000):
            total += 1
            new_hash = _sha256(msg.content or "")
            if new_hash != msg.content_hash:
                msg.content_hash = new_hash
                updated += 1
                if updated % 10000 == 0:
                    session.flush()
                    _logger.info("已更新 %d / %d 条", updated, total)
        session.commit()
        _logger.info("完成：共 %d 条消息，更新 %d 条", total, updated)
    except Exception:
        session.rollback()
        _logger.exception("迁移失败")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
