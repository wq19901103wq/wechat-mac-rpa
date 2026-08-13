#!/usr/bin/env python3
"""对聊天记录数据库执行全局去重。

基于 ChatHistoryRepository.deduplicate_all()，按 (chatroom_id, create_time, content_hash)
保留每组重复中 id 最小的一条。
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import init_db
from src.db.repository import ChatHistoryRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


def main():
    db_path = Path("data/db/chat_history.db")
    if not db_path.exists():
        _logger.error("数据库不存在: %s", db_path)
        sys.exit(1)

    init_db(db_path)
    repo = ChatHistoryRepository(db_path=db_path)
    result = repo.deduplicate_all()
    total = sum(result.values())
    if total:
        _logger.info("共删除 %d 条重复消息，涉及 %d 个 chatroom", total, len(result))
        for chatroom_id, count in sorted(result.items(), key=lambda x: -x[1])[:10]:
            _logger.info("  %s: %d 条", chatroom_id, count)
    else:
        _logger.info("未发现重复消息")


if __name__ == "__main__":
    main()
