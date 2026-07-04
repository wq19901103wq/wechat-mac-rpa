"""聊天记录 Repository（Phase 1 MVP）。"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from src.db.connection import get_engine
from src.db.models import ChatMember, Chatroom, Message

_logger = logging.getLogger("src.db.repository")


class ChatHistoryRepository:
    """聊天记录持久化仓库。

    职责：
    - 把 GlobalStore 中的聊天/消息/成员同步到 SQLite
    - 按 chatroom_id 区分同名群
    - 幂等写入（按复合唯一键去重）
    """

    def __init__(self, db_path: Optional[Path] = None, session: Optional[Session] = None):
        """初始化仓库。

        Args:
            db_path: 数据库文件路径。若同时传入 session，此参数仅用于日志。
            session: 外部传入的 SQLAlchemy Session，便于测试和事务控制。
        """
        self.db_path = db_path
        self._external_session = session
        self._engine = None if session else get_engine(db_path)

    def _session(self) -> Session:
        if self._external_session is not None:
            return self._external_session
        from sqlalchemy.orm import sessionmaker
        SessionLocal = sessionmaker(bind=self._engine, future=True)
        return SessionLocal()

    @staticmethod
    def _content_hash(content: Optional[str]) -> str:
        """计算消息内容哈希（用 md5，足够去重且比 sha256 快）。

        仅用于消息去重，不用于安全场景；已显式声明 usedforsecurity=False。
        """
        text = content or ""
        return hashlib.md5(  # lgtm[py/weak-cryptographic-algorithm] # codeql[py/weak-cryptographic-algorithm]
            text.encode("utf-8"), usedforsecurity=False
        ).hexdigest()

    def _get_or_create_chatroom(
        self,
        session: Session,
        chatroom_id: str,
        display_name: str,
        chat_type: str,
        first_seen_at: Optional[float] = None,
        last_active_at: Optional[float] = None,
    ) -> Chatroom:
        """获取或创建 chatroom，并更新活跃时间。"""
        now = time.time()
        stmt = select(Chatroom).where(Chatroom.chatroom_id == chatroom_id)
        chatroom = session.execute(stmt).scalar_one_or_none()
        if chatroom is None:
            chatroom = Chatroom(
                chatroom_id=chatroom_id,
                display_name=display_name,
                chat_type=chat_type,
                first_seen_at=first_seen_at or now,
                last_active_at=last_active_at or now,
                created_at=now,
                updated_at=now,
            )
            session.add(chatroom)
            session.flush()
            _logger.debug("[repo] 新建 chatroom: %s (%s)", chatroom_id, display_name)
        else:
            chatroom.display_name = display_name
            if last_active_at and (chatroom.last_active_at is None or last_active_at > chatroom.last_active_at):
                chatroom.last_active_at = last_active_at
            chatroom.updated_at = now
            session.flush()
        return chatroom

    def _upsert_message(
        self,
        session: Session,
        chatroom_db_id: int,
        msg: Dict,
        source_file: Optional[str] = None,
    ) -> Tuple[bool, Optional[Message]]:
        """插入单条消息，若已存在则跳过。

        Returns:
            (是否新增, Message 对象或 None)
        """
        content = msg.get("content") or msg.get("text") or ""
        content_hash = self._content_hash(content)
        wxid = msg.get("wxid") or msg.get("sender_wxid") or ""
        create_time = msg.get("create_time")
        if create_time is None:
            create_time = msg.get("timestamp") or time.time()

        # 检查是否已存在
        stmt = select(Message).where(
            Message.chatroom_id == chatroom_db_id,
            Message.wxid == wxid,
            Message.create_time == create_time,
            Message.content_hash == content_hash,
        )
        existing = session.execute(stmt).scalar_one_or_none()
        if existing is not None:
            return False, existing

        message = Message(
            chatroom_id=chatroom_db_id,
            local_id=msg.get("local_id"),
            server_id=msg.get("server_id"),
            wxid=wxid,
            sender_display_name=msg.get("sender_display_name") or msg.get("sender"),
            is_self=bool(msg.get("is_self", False)),
            content=content,
            message_type=msg.get("message_type", "text"),
            image_description=msg.get("image_description"),
            is_at_me=bool(msg.get("is_at_me", False)),
            is_revoked=bool(msg.get("is_revoked", False)),
            replied=bool(msg.get("replied", False)),
            reply_text=msg.get("reply_text"),
            reply_time=msg.get("reply_time"),
            create_time=create_time,
            raw_type=msg.get("raw_type"),
            source_file=source_file,
            content_hash=content_hash,
            created_at=time.time(),
        )
        session.add(message)
        return True, message

    def _upsert_chat_member(
        self,
        session: Session,
        chatroom_db_id: int,
        wxid: str,
        group_nickname: Optional[str] = None,
        joined_at: Optional[float] = None,
    ) -> bool:
        """插入或更新群成员。"""
        stmt = select(ChatMember).where(
            ChatMember.chatroom_id == chatroom_db_id,
            ChatMember.wxid == wxid,
        )
        member = session.execute(stmt).scalar_one_or_none()
        now = time.time()
        if member is None:
            member = ChatMember(
                chatroom_id=chatroom_db_id,
                wxid=wxid,
                group_nickname=group_nickname,
                joined_at=joined_at or now,
                is_active=True,
                created_at=now,
            )
            session.add(member)
            return True
        else:
            if group_nickname:
                member.group_nickname = group_nickname
            member.is_active = True
            return False

    def sync_chat(
        self,
        chatroom_id: str,
        display_name: str,
        chat_type: str,
        messages: List[Dict],
        source_file: Optional[str] = None,
    ) -> Dict:
        """同步一个聊天的消息到数据库。

        Args:
            chatroom_id: 微信 chatroom_id 或 wxid（必须）
            display_name: 显示名
            chat_type: group / single
            messages: 消息字典列表
            source_file: 来源文件路径（可选）

        Returns:
            统计信息 dict
        """
        if not chatroom_id:
            _logger.warning("[repo] chatroom_id 为空，跳过同步")
            return {"chatrooms": 0, "messages": 0, "members": 0, "skipped": 0}

        session = self._session()
        try:
            with session.begin():
                chatroom = self._get_or_create_chatroom(
                    session,
                    chatroom_id,
                    display_name,
                    chat_type,
                )

                inserted = 0
                skipped = 0
                senders: Dict[str, Optional[str]] = {}

                for msg in messages:
                    is_new, _ = self._upsert_message(
                        session,
                        chatroom.id,
                        msg,
                        source_file=source_file,
                    )
                    if is_new:
                        inserted += 1
                    else:
                        skipped += 1

                    wxid = msg.get("wxid") or msg.get("sender_wxid")
                    if wxid:
                        senders[wxid] = msg.get("sender_display_name") or msg.get("sender")

                members_inserted = 0
                sender_first_seen: Dict[str, float] = {}
                for msg in messages:
                    wxid = msg.get("wxid") or msg.get("sender_wxid")
                    if not wxid:
                        continue
                    ct = msg.get("create_time")
                    if ct is not None and (wxid not in sender_first_seen or ct < sender_first_seen[wxid]):
                        sender_first_seen[wxid] = ct

                for wxid, nickname in senders.items():
                    is_new = self._upsert_chat_member(
                        session,
                        chatroom.id,
                        wxid,
                        group_nickname=nickname,
                        joined_at=sender_first_seen.get(wxid),
                    )
                    if is_new:
                        members_inserted += 1

                # 更新最后活跃时间
                if messages:
                    last_time = max(
                        (m.get("create_time") or time.time()) for m in messages
                    )
                    chatroom.last_active_at = last_time
                    chatroom.updated_at = time.time()

            return {
                "chatrooms": 1,
                "messages": inserted,
                "members": members_inserted,
                "skipped": skipped,
            }
        except Exception as e:
            _logger.error("[repo] 同步 chatroom %s 失败: %s", chatroom_id, e)
            raise
        finally:
            if self._external_session is None:
                session.close()

    def bulk_sync_chat(
        self,
        chatroom_id: str,
        display_name: str,
        chat_type: str,
        messages: List[Dict],
        source_file: Optional[str] = None,
    ) -> Dict:
        """批量同步一个聊天的消息到数据库（用于迁移场景，比 sync_chat 快很多）。

        使用 SQLite INSERT OR IGNORE 批量插入，不做逐条存在性检查。
        """
        if not chatroom_id:
            return {"chatrooms": 0, "messages": 0, "members": 0, "skipped": 0}

        session = self._session()
        now = time.time()
        try:
            with session.begin():
                chatroom = self._get_or_create_chatroom(
                    session,
                    chatroom_id,
                    display_name,
                    chat_type,
                )

                # 准备消息数据
                message_values = []
                senders: Dict[str, Optional[str]] = {}
                sender_first_seen: Dict[str, float] = {}

                for msg in messages:
                    content = msg.get("content") or msg.get("text") or ""
                    wxid = msg.get("wxid") or msg.get("sender_wxid") or ""
                    create_time = msg.get("create_time")
                    if create_time is None:
                        create_time = msg.get("timestamp") or now

                    if not wxid:
                        continue

                    message_values.append({
                        "chatroom_id": chatroom.id,
                        "local_id": msg.get("local_id"),
                        "server_id": msg.get("server_id"),
                        "wxid": wxid,
                        "sender_display_name": msg.get("sender_display_name") or msg.get("sender"),
                        "is_self": bool(msg.get("is_self", False)),
                        "content": content,
                        "message_type": msg.get("message_type", "text"),
                        "image_description": msg.get("image_description"),
                        "is_at_me": bool(msg.get("is_at_me", False)),
                        "is_revoked": bool(msg.get("is_revoked", False)),
                        "replied": bool(msg.get("replied", False)),
                        "reply_text": msg.get("reply_text"),
                        "reply_time": msg.get("reply_time"),
                        "create_time": create_time,
                        "raw_type": msg.get("raw_type"),
                        "source_file": source_file,
                        "content_hash": self._content_hash(content),
                        "created_at": now,
                    })

                    senders[wxid] = msg.get("sender_display_name") or msg.get("sender")
                    if create_time is not None and (wxid not in sender_first_seen or create_time < sender_first_seen[wxid]):
                        sender_first_seen[wxid] = create_time

                # 批量插入消息（分块，避免超过 SQLite 参数上限）
                # 每条消息 19 个字段，SQLite 支持 32766 参数，保守每批 1000 条
                BATCH_SIZE = 1000
                inserted = 0
                for i in range(0, len(message_values), BATCH_SIZE):
                    batch = message_values[i:i + BATCH_SIZE]
                    stmt = sqlite_insert(Message).values(batch)
                    stmt = stmt.on_conflict_do_nothing(
                        index_elements=["chatroom_id", "wxid", "create_time", "content_hash"]
                    )
                    result = session.execute(stmt)
                    inserted += result.rowcount

                # 批量插入成员（每个成员 6 个字段，每批 100 个）
                member_values = []
                for wxid, nickname in senders.items():
                    member_values.append({
                        "chatroom_id": chatroom.id,
                        "wxid": wxid,
                        "group_nickname": nickname,
                        "joined_at": sender_first_seen.get(wxid, now),
                        "is_active": True,
                        "created_at": now,
                    })

                MEMBER_BATCH_SIZE = 3000
                members_inserted = 0
                for i in range(0, len(member_values), MEMBER_BATCH_SIZE):
                    batch = member_values[i:i + MEMBER_BATCH_SIZE]
                    stmt = sqlite_insert(ChatMember).values(batch)
                    stmt = stmt.on_conflict_do_nothing(
                        index_elements=["chatroom_id", "wxid"]
                    )
                    result = session.execute(stmt)
                    members_inserted += result.rowcount

                if messages:
                    last_time = max(
                        (m.get("create_time") or now) for m in messages
                    )
                    chatroom.last_active_at = last_time
                    chatroom.updated_at = now

            return {
                "chatrooms": 1,
                "messages": inserted,
                "members": members_inserted,
                "skipped": len(message_values) - inserted,
            }
        except Exception as e:
            _logger.error("[repo] 批量同步 chatroom %s 失败: %s", chatroom_id, e)
            raise
        finally:
            if self._external_session is None:
                session.close()

    def get_chatroom(self, chatroom_id: str) -> Optional[Chatroom]:
        """按 chatroom_id 查询聊天。"""
        session = self._session()
        try:
            stmt = select(Chatroom).where(Chatroom.chatroom_id == chatroom_id)
            return session.execute(stmt).scalar_one_or_none()
        finally:
            if self._external_session is None:
                session.close()

    def get_messages(
        self,
        chatroom_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Message]:
        """按 chatroom_id 查询消息。"""
        session = self._session()
        try:
            chatroom = self.get_chatroom(chatroom_id)
            if chatroom is None:
                return []
            stmt = (
                select(Message)
                .where(Message.chatroom_id == chatroom.id)
                .order_by(Message.create_time.asc())
                .offset(offset)
            )
            if limit is not None:
                stmt = stmt.limit(limit)
            return list(session.execute(stmt).scalars().all())
        finally:
            if self._external_session is None:
                session.close()

    def get_latest_messages(
        self,
        chatroom_id: str,
        limit: int = 200,
    ) -> List[Message]:
        """按 chatroom_id 查询最近 N 条消息（按 create_time 降序）。"""
        session = self._session()
        try:
            chatroom = self.get_chatroom(chatroom_id)
            if chatroom is None:
                return []
            stmt = (
                select(Message)
                .where(Message.chatroom_id == chatroom.id)
                .order_by(Message.create_time.desc())
                .limit(limit)
            )
            results = list(session.execute(stmt).scalars().all())
            results.reverse()  # 返回升序，符合 GlobalStore 习惯
            return results
        finally:
            if self._external_session is None:
                session.close()

    def list_chatrooms(self) -> List[Chatroom]:
        """返回所有聊天会话。"""
        session = self._session()
        try:
            return list(session.execute(select(Chatroom)).scalars().all())
        finally:
            if self._external_session is None:
                session.close()

    def get_stats(self) -> Dict:
        """返回数据库统计。"""
        session = self._session()
        try:
            from sqlalchemy import func
            return {
                "chatrooms": session.execute(select(func.count(Chatroom.id))).scalar(),
                "messages": session.execute(select(func.count(Message.id))).scalar(),
                "members": session.execute(select(func.count(ChatMember.id))).scalar(),
            }
        finally:
            if self._external_session is None:
                session.close()


    # ------------------------------------------------------------------
    # GlobalStore 拆分后下沉的 DB 身份解析与加载逻辑
    # ------------------------------------------------------------------

    @staticmethod
    def synthetic_chatroom_id(chat_name: str) -> str:
        """为没有真实 chatroom_id 的 OCR 聊天生成稳定合成 ID。"""
        h = hashlib.md5(chat_name.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
        return f"_chats_{chat_name}_{h}"

    def count_messages(self, chatroom_db_id: int) -> int:
        """返回某个 chatroom 的消息数。"""
        session = self._session()
        try:
            from sqlalchemy import func
            stmt = select(func.count(Message.id)).where(Message.chatroom_id == chatroom_db_id)
            return session.execute(stmt).scalar() or 0
        finally:
            if self._external_session is None:
                session.close()

    def select_chatroom_for_load(
        self,
        chat_name: str,
        rooms: List[Chatroom],
    ) -> Optional[Chatroom]:
        """为同名群选择一个要加载的 chatroom。

        优先级：
        1. 合成 ID 匹配（OCR 活跃群）
        2. 非合成 ID 中消息数最多的房间
        3. 兜底第一个
        """
        if len(rooms) == 1:
            return rooms[0]

        synthetic_id = self.synthetic_chatroom_id(chat_name)
        for room in rooms:
            if room.chatroom_id == synthetic_id:
                return room

        non_synthetic = [r for r in rooms if not r.chatroom_id.startswith("_chats_")]
        candidates = non_synthetic or rooms
        try:
            return max(candidates, key=lambda r: self.count_messages(r.id))
        except Exception:
            return candidates[0]

    def resolve_chatroom_id(
        self,
        chat_name: str,
        messages: List[Dict],
    ) -> str:
        """根据聊天名和消息解析 chatroom_id。

        优先级：
        1. 消息里已有 chatroom_id（WeFlow 模式）直接使用。
        2. OCR 模式：按 display_name 查 DB，只有唯一匹配时才复用，
           否则生成稳定合成 ID，避免同名群误合并或新群丢失。
        """
        # 1. WeFlow 模式
        for m in messages:
            cid = m.get("chatroom_id")
            if cid:
                return cid

        # 2. OCR 模式：尝试唯一匹配
        try:
            session = self._session()
            try:
                rooms = list(
                    session.execute(
                        select(Chatroom).where(Chatroom.display_name == chat_name)
                    ).scalars().all()
                )
                if len(rooms) == 1:
                    return rooms[0].chatroom_id
                if len(rooms) > 1:
                    _logger.warning(
                        "[repo] %s 在 DB 中匹配到 %d 个 chatroom_id，使用合成 ID",
                        chat_name,
                        len(rooms),
                    )
            finally:
                if self._external_session is None:
                    session.close()
        except Exception as e:
            _logger.debug("[repo] 按名称解析 chatroom_id 失败: %s", e)

        # 3. 生成合成 ID
        return self.synthetic_chatroom_id(chat_name)

    def load_active_chatrooms(
        self,
        max_messages: int,
    ) -> Dict[str, Tuple[str, str, List[Message]]]:
        """加载每个 display_name 对应的活动聊天。

        返回：
            {display_name: (chatroom_id, chat_type, messages)}
        """
        result: Dict[str, Tuple[str, str, List[Message]]] = {}
        try:
            from collections import defaultdict
            rooms = self.list_chatrooms()
            by_name: Dict[str, List[Chatroom]] = defaultdict(list)
            for room in rooms:
                by_name[room.display_name].append(room)

            for chat_name, room_list in by_name.items():
                selected: Optional[Chatroom] = self.select_chatroom_for_load(chat_name, room_list)
                if selected is None:
                    continue
                db_messages = self.get_latest_messages(selected.chatroom_id, limit=max_messages)
                result[chat_name] = (selected.chatroom_id, selected.chat_type, db_messages)
        except Exception as e:
            _logger.error("[repo] 加载活动聊天失败: %s", e)
            raise
        return result

    # ------------------------------------------------------------------
    # 去重逻辑封装
    # ------------------------------------------------------------------

    def deduplicate_chatroom(self, chatroom_id: str) -> int:
        """删除某个 chatroom 内按 (chatroom_db_id, create_time, content_hash) 重复的 Message。

        保留每组重复中 id 最小的一条。

        Returns:
            删除的消息条数
        """
        session = self._session()
        try:
            with session.begin():
                chatroom = self.get_chatroom(chatroom_id)
                if chatroom is None:
                    return 0

                from sqlalchemy import func
                subq = (
                    select(
                        Message.chatroom_id,
                        Message.create_time,
                        Message.content_hash,
                        func.min(Message.id).label("keep_id"),
                        func.count(Message.id).label("cnt"),
                    )
                    .where(Message.chatroom_id == chatroom.id)
                    .group_by(Message.chatroom_id, Message.create_time, Message.content_hash)
                    .having(func.count(Message.id) > 1)
                    .subquery()
                )
                dup_stmt = select(Message.id).join(
                    subq,
                    (
                        (Message.chatroom_id == subq.c.chatroom_id)
                        & (Message.create_time == subq.c.create_time)
                        & (Message.content_hash == subq.c.content_hash)
                        & (Message.id != subq.c.keep_id)
                    ),
                )
                ids_to_delete = [row[0] for row in session.execute(dup_stmt).all()]
                if not ids_to_delete:
                    return 0
                from sqlalchemy import delete
                delete_stmt = delete(Message).where(Message.id.in_(ids_to_delete))
                session.execute(delete_stmt)
                _logger.info(
                    "[repo] 删除 %s 重复消息 %d 条",
                    chatroom_id,
                    len(ids_to_delete),
                )
                return len(ids_to_delete)
        except Exception:
            _logger.exception("[repo] 去重 %s 失败", chatroom_id)
            raise
        finally:
            if self._external_session is None:
                session.close()

    def deduplicate_all(self) -> Dict[str, int]:
        """对所有 chatroom 执行去重。

        Returns:
            {chatroom_id: 删除条数}
        """
        result: Dict[str, int] = {}
        for room in self.list_chatrooms():
            try:
                removed = self.deduplicate_chatroom(room.chatroom_id)
                if removed:
                    result[room.chatroom_id] = removed
            except Exception as e:
                _logger.error("[repo] 去重 %s 失败: %s", room.chatroom_id, e)
        return result
