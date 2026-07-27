"""数据库连接管理。"""

import logging
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from src.db.models import Base

_logger = logging.getLogger("src.db")

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "db" / "chat_history.db"


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_engine(db_path: Optional[Path] = None):
    """获取 SQLAlchemy engine，启用 WAL 模式。"""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    else:
        db_path = Path(db_path)
    _ensure_dir(db_path)
    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        future=True,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def get_session(db_path: Optional[Path] = None) -> Session:
    """获取一个新 Session。"""
    engine = get_engine(Path(db_path) if db_path else None)
    SessionLocal = sessionmaker(bind=engine, future=True)
    return SessionLocal()


def init_db(db_path: Optional[Path] = None):
    """创建所有表。幂等，不会删除已有数据。"""
    engine = get_engine(Path(db_path) if db_path else None)
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        result = conn.execute(text(
            "UPDATE messages SET is_self=1 "
            "WHERE sender_display_name='自己' AND is_self=0 AND source_file IS NULL"
        ))
        if result.rowcount:
            _logger.info("[db] 修复历史 self 消息标记: %d 条", result.rowcount)
    _logger.info("[db] 已初始化/确认表结构: %s", db_path or DEFAULT_DB_PATH)
    return engine


def check_wal_mode(db_path: Optional[Path] = None) -> bool:
    """检查当前数据库是否启用 WAL 模式。"""
    engine = get_engine(db_path)
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA journal_mode"))
        mode = result.scalar()
    return mode == "wal"
