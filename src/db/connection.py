"""Compatibility defaults for the RPA database location."""
from pathlib import Path
from wechat_reader.db import connection as _shared
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data/db/chat_history.db"

def get_engine(db_path=None):
    return _shared.get_engine(db_path if db_path is not None else DEFAULT_DB_PATH)

def get_session(db_path=None):
    return _shared.get_session(db_path if db_path is not None else DEFAULT_DB_PATH)

def init_db(db_path=None):
    return _shared.init_db(db_path if db_path is not None else DEFAULT_DB_PATH)

def check_wal_mode(db_path=None):
    return _shared.check_wal_mode(db_path if db_path is not None else DEFAULT_DB_PATH)
