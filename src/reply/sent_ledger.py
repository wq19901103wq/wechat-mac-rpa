"""Read confirmed sends for DH history eligibility, without display-name matching."""

from datetime import datetime
import json
import logging
import math
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo


def _observation(message):
    basis = "observed_at" if message.get("observed_at") is not None else "create_time"
    observed = message.get(basis)
    role = message.get("sender_type")
    if (type(observed) not in (int, float) or not math.isfinite(observed) or observed <= 0
            or not isinstance(message.get("text"), str) or not message["text"]
            or not isinstance(message.get("sender"), str) or not message["sender"]
            or role not in ("self", "other")):
        return None
    if "is_self" in message and message["is_self"] is not (role == "self"):
        return None
    return basis, observed, message["text"], message["sender"], role == "self"


def _history_index(database):
    if database is None:
        return {}
    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        index = {}
        rows = connection.execute("SELECT m.observed_at, m.create_time, m.content, m.sender_display_name, "
            "m.is_self, c.chatroom_id FROM messages m JOIN chatrooms c ON m.chatroom_id = c.id")
        for observed, created, content, sender, is_self, identity in rows:
            if is_self in (0, 1):
                key = _observation(dict(observed_at=observed, create_time=created, text=content,
                    sender=sender, sender_type="self" if is_self else "other"))
                if key is not None:
                    index.setdefault(key, set()).add(identity)
        return index
    finally:
        connection.close()


def _row_evidence(created, inputs, replies, mapping, history):
    inputs, replies = json.loads(inputs), json.loads(replies)
    if not isinstance(inputs, list) or not isinstance(replies, list):
        raise ValueError("发送记录格式无效")
    if any(not isinstance(reply, str) or not reply.strip() for reply in replies):
        raise ValueError("已发送回复必须是非空文本")
    identities = [m.get("chatroom_id") for m in inputs if isinstance(m, dict) and m.get("chatroom_id")]
    if any(not isinstance(identity, str) for identity in identities):
        raise ValueError("发送记录的聊天 ID 无效")
    identities = set(identities)
    for message in inputs:
        if isinstance(message, dict):
            identities.update(history.get(_observation(message), ()))
    sources = {mapping.get(identity, identity if not identity.startswith("_chats_") else None)
               for identity in identities}
    if len(sources) != 1 or None in sources:
        return []
    timestamp = datetime.fromisoformat(created)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    sent_at = timestamp.timestamp()
    if not math.isfinite(sent_at) or sent_at <= 0:
        raise ValueError("发送记录时间无效")
    source = sources.pop()
    return [dict(source_chat_id=source, timestamp=sent_at, text=reply) for reply in replies]


class SentReplyLedger:
    """Follow the bot's serial insert/send/update cycle, overlapping its latest tick."""

    def __init__(self, database, source_chat_ids=None, *, history_database=None):
        mapping = {} if source_chat_ids is None else source_chat_ids
        if not isinstance(mapping, dict) or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v
            for k, v in mapping.items()):
            raise ValueError("发送记录的聊天 ID 映射无效")
        self.mapping = dict(mapping)
        self.history = _history_index(history_database)
        self.database = Path(database).resolve().as_uri() + "?mode=ro"
        self.cursor = 0
        self._frontier = set()

    def read(self):
        evidence = []
        connection = sqlite3.connect(self.database, uri=True)
        try:
            rows = connection.execute("SELECT id, created_at, session_input_messages_json, "
                "replies_sent_json, send_success FROM tick_log WHERE id >= ? ORDER BY id", (self.cursor,))
            for identity, created, inputs, replies, success in rows:
                previous = self._frontier if identity == self.cursor else set()
                emitted = set(previous)
                if success is not None and replies is not None and replies != '[]':
                    try:
                        batch = _row_evidence(created, inputs, replies, self.mapping, self.history)
                    except (ValueError, TypeError, OverflowError, OSError):
                        logging.getLogger(__name__).warning("跳过无效发送记录 id=%s", identity)
                    else:
                        for position, item in enumerate(batch):
                            key = (position, item['source_chat_id'], item['timestamp'], item['text'])
                            if key not in previous:
                                evidence.append(item)
                                emitted.add(key)
                self.cursor, self._frontier = identity, emitted
            return sorted(evidence, key=lambda item: item['timestamp'])
        finally:
            connection.close()


def load_sent_replies(database, source_chat_ids=None, *, history_database=None):
    return SentReplyLedger(database, source_chat_ids, history_database=history_database).read()
