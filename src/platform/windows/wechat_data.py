"""Windows 微信本地数据读取层。

提供稳定接口 list_contacts() / get_sessions() / get_messages(talker, limit, offset)。
首次调用会自动：定位 db_storage -> 提 key -> 解密到 data/wechat/（gitignored）。
也可直接传入 decrypted_dir 复用已有解密结果（测试/离线场景）。

数据来源为微信 4.1+ 的 WCDB/SQLCipher4 本地库，读取过程只读进程内存 + 解密
本地副本，不注入、不挂钩、不修改微信文件。
"""

import hashlib
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.platform.windows.config import get_cache_dir, get_db_storage_dir
from src.platform.windows.decryptor import ensure_decrypted
from src.platform.windows.key_provider import extract_keys, keys_cache_path
from src.platform.windows.message_codec import decode_message_content, split_sender_prefix

_logger = logging.getLogger("src.platform.windows.wechat_data")


@dataclass
class WeChatContact:
    username: str
    nickname: str = ""
    remark: str = ""
    alias: str = ""
    local_type: int = 0

    @property
    def display_name(self) -> str:
        return self.remark or self.nickname or self.username

    @property
    def is_group(self) -> bool:
        return self.username.endswith("@chatroom")


@dataclass
class WeChatSession:
    username: str
    unread_count: int = 0
    summary: str = ""
    last_timestamp: int = 0
    sort_timestamp: int = 0
    last_msg_local_id: int = 0
    last_msg_type: int = 0
    last_msg_sub_type: int = 0
    last_msg_sender: str = ""
    last_sender_display_name: str = ""

    @property
    def name(self) -> str:
        return self.last_sender_display_name or self.username


@dataclass
class WeChatMessage:
    local_id: int
    local_type: int
    create_time: int
    content: str
    raw_content: str
    server_id: Optional[int] = None
    sort_seq: Optional[int] = None
    sender_username: Optional[str] = None
    packed_info_data: Optional[bytes] = None


class WeChatData:
    """微信本地数据库读取客户端。"""

    def __init__(
        self,
        db_storage_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        decrypted_dir_override: Optional[Path] = None,
    ):
        self.db_storage_dir = Path(db_storage_dir) if db_storage_dir else get_db_storage_dir()
        self.cache_dir = Path(cache_dir) if cache_dir else get_cache_dir()
        self._decrypted_dir_override = Path(decrypted_dir_override) if decrypted_dir_override else None
        self._decrypted: Optional[Path] = None
        self._sender_maps: dict[str, dict[int, str]] = {}

    # ------------------------------------------------------------------
    # 准备（提 key + 解密）
    # ------------------------------------------------------------------
    def _resolve_decrypted_dir(self) -> Path:
        if self._decrypted_dir_override is not None:
            return self._decrypted_dir_override
        if self._decrypted is not None:
            return self._decrypted
        keys = keys_cache_path(self.cache_dir)
        if not keys.exists():
            keys = extract_keys(self.db_storage_dir, self.cache_dir)
        self._decrypted = ensure_decrypted(self.db_storage_dir, keys, self.cache_dir)
        return self._decrypted

    # ------------------------------------------------------------------
    # 联系人
    # ------------------------------------------------------------------
    def list_contacts(self) -> list[WeChatContact]:
        """返回全部联系人（按 local_type、username 排序）。"""
        db = self._resolve_decrypted_dir() / "contact" / "contact.db"
        conn = self._connect_readonly(db)
        try:
            rows = conn.execute(
                "SELECT username, COALESCE(nick_name,''), COALESCE(remark,''), "
                "COALESCE(alias,''), COALESCE(local_type,0) "
                "FROM contact WHERE username IS NOT NULL AND username != '' "
                "ORDER BY local_type, username"
            ).fetchall()
        finally:
            conn.close()
        return [
            WeChatContact(
                username=r[0], nickname=r[1], remark=r[2], alias=r[3], local_type=r[4]
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------
    def get_sessions(self) -> list[WeChatSession]:
        """返回会话列表（按 sort_timestamp 降序）。"""
        db = self._resolve_decrypted_dir() / "session" / "session.db"
        conn = self._connect_readonly(db)
        try:
            rows = conn.execute(
                "SELECT username, COALESCE(unread_count,0), COALESCE(summary,''), "
                "COALESCE(last_timestamp,0), COALESCE(sort_timestamp,0), "
                "COALESCE(last_msg_locald_id,0), COALESCE(last_msg_type,0), "
                "COALESCE(last_msg_sub_type,0), COALESCE(last_msg_sender,''), "
                "COALESCE(last_sender_display_name,'') "
                "FROM SessionTable WHERE username IS NOT NULL AND username != '' "
                "ORDER BY sort_timestamp DESC"
            ).fetchall()
        finally:
            conn.close()
        return [
            WeChatSession(
                username=r[0], unread_count=r[1], summary=r[2], last_timestamp=r[3],
                sort_timestamp=r[4], last_msg_local_id=r[5], last_msg_type=r[6],
                last_msg_sub_type=r[7], last_msg_sender=r[8], last_sender_display_name=r[9],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # 消息
    # ------------------------------------------------------------------
    def get_messages(
        self, talker: str, limit: int = 50, offset: int = 0
    ) -> list[WeChatMessage]:
        """按 talker 拉取聊天记录（跨 message_*.db 分片，按时间升序）。

        talker 为会话 user_name（如 wxid_xxx / xxx@chatroom），对应表
        Msg_<md5(talker)>。
        """
        if limit < 0 or offset < 0:
            raise ValueError("limit/offset 不能为负数")
        table = "Msg_" + hashlib.md5(talker.encode("utf-8"), usedforsecurity=False).hexdigest()
        decrypted = self._resolve_decrypted_dir()
        message_dir = decrypted / "message"
        if not message_dir.is_dir():
            return []
        merged: list[WeChatMessage] = []
        for db in sorted(message_dir.glob("message_*.db")):
            merged.extend(self._query_message_db(db, table))
        merged.sort(key=lambda m: (m.create_time, m.sort_seq or 0, m.local_id))
        return merged[offset : offset + limit]

    _TABLE_RE = re.compile(r"^Msg_[0-9a-f]{32}$")

    def _query_message_db(self, db: Path, table: str) -> list[WeChatMessage]:
        if not self._TABLE_RE.match(table):
            raise ValueError(f"非法消息表名: {table!r}")
        conn = self._connect_readonly(db)
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if not exists:
                return []
            # 表名已由 _TABLE_RE 白名单校验（仅内部 md5 生成的 Msg_<32hex>），
            # 非用户输入，见 get_messages 调用处。
            rows = conn.execute(
                f"SELECT local_id, server_id, local_type, sort_seq, real_sender_id, "
                f"create_time, message_content, packed_info_data FROM {table}"  # nosec B608
            ).fetchall()
        finally:
            conn.close()
        sender_map = self._sender_map(db)
        known = set(sender_map.values())
        messages: list[WeChatMessage] = []
        for row in rows:
            local_id, server_id, local_type, sort_seq, real_sender_id, create_time, raw, packed = row
            raw_content = decode_message_content(raw)
            sender, body = split_sender_prefix(raw_content, known)
            sender_name = sender_map.get(real_sender_id) if real_sender_id is not None else None
            messages.append(
                WeChatMessage(
                    local_id=local_id or 0,
                    server_id=server_id,
                    local_type=local_type or 0,
                    create_time=create_time or 0,
                    sort_seq=sort_seq,
                    sender_username=sender_name,
                    content=body,
                    raw_content=raw_content,
                    packed_info_data=packed,
                )
            )
        return messages

    def _sender_map(self, db: Path) -> dict[int, str]:
        """Name2Id 表 rowid -> user_name（rowid 即 real_sender_id）。"""
        key = str(db)
        if key not in self._sender_maps:
            conn = self._connect_readonly(db)
            try:
                rows = conn.execute(
                    "SELECT rowid, user_name FROM Name2Id "
                    "WHERE user_name IS NOT NULL AND user_name != ''"
                ).fetchall()
            finally:
                conn.close()
            self._sender_maps[key] = {rowid: name for rowid, name in rows}
        return self._sender_maps[key]

    @staticmethod
    def _connect_readonly(db: Path) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro", uri=True)
