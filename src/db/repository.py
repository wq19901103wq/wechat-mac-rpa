"""RPA path compatibility over the shared raw repository."""
from wechat_reader.db.repository import ChatHistoryRepository as _SharedRepository
from .connection import DEFAULT_DB_PATH

class ChatHistoryRepository(_SharedRepository):
    def __init__(self, db_path=None, session=None):
        super().__init__(db_path if db_path is not None else DEFAULT_DB_PATH, session)
