"""RPA reply state over the shared raw WeChat store."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from wechat_reader.models.base import ChatMessage, SenderType
from wechat_reader.store import RawChatState, RawMessageStore
from wechat_reader.store import (
    _normalize_text, _jaccard_2gram, _normalize_sender, _msg_id, _media_matches,
    _is_fuzzy_duplicate, _match_single, _lcs_match,
)
from wechat_reader.utils.chat_utils import _is_group_chat_name


@dataclass
class ChatState(RawChatState):
    pending_self_messages: List[ChatMessage] = field(default_factory=list)


class GlobalStore(RawMessageStore):
    state_type = ChatState

    def __init__(self, max_messages=None, state_file=None):
        if state_file is None:
            state_file = Path(__file__).parent.parent.parent / "data/global_state.json"
        self._state_file = Path(state_file)
        super().__init__(self._state_file.parent / "db/chat_history.db",
                         data_dir=self._state_file.parent,
                         max_messages=max_messages, strict=False)

    def merge_tick(self, chat_name, messages, is_group=False):
        state, _ = super().merge_tick(chat_name, messages, is_group)
        unreplied = [m for m in state.messages if not m.replied
                     and m.sender_type != SenderType.SELF and m.sender != "自己"]
        return state, unreplied

    @staticmethod
    def _db_msg_to_chat_message(db_msg, chat_name):
        msg = RawMessageStore._db_msg_to_chat_message(db_msg, chat_name)
        msg.replied = bool(db_msg.replied)
        msg.reply_text = db_msg.reply_text or ""
        msg.reply_time = db_msg.reply_time
        return msg

    @staticmethod
    def _msg_to_dict(msg):
        return dict(RawMessageStore._msg_to_dict(msg), replied=msg.replied,
                    reply_text=msg.reply_text, reply_time=msg.reply_time)

    def mark_replied(self, chat_name: str, target_msg: ChatMessage, reply_text: str):
        """标记单条消息已回复。"""
        state = self.chats.get(chat_name)
        if not state:
            return
        now = time.time()

        # 用 is 匹配（target_msg 就是 state.messages 中的对象引用）
        # 如果 is 匹配不到，再用 text+sender 兜底
        marked = False
        for msg in state.messages:
            if msg is target_msg or (msg.text == target_msg.text and msg.sender == target_msg.sender):
                msg.replied = True
                msg.reply_text = reply_text
                msg.reply_time = now
                marked = True
                # 不 break，继续标记所有匹配的消息（OCR 不稳定可能导致同一条消息存了多份）
        if marked:
            self._dirty.add(chat_name)


    def get_unreplied(self, chat_name: str) -> List[ChatMessage]:
        """获取某聊天中所有未回复的消息（按时间顺序）"""
        state = self.chats.get(chat_name)
        if not state:
            return []
        return [
            m for m in state.messages
            if not m.replied and m.sender_type != SenderType.SELF
        ]


    def last_reply_time(self, chat_name: str) -> Optional[float]:
        """最后回复时间（从消息中推导）"""
        state = self.chats.get(chat_name)
        if not state:
            return None
        replied_times = [
            m.reply_time for m in state.messages
            if m.replied and m.reply_time
        ]
        return max(replied_times) if replied_times else None


    def reply_count(self, chat_name: str) -> int:
        """回复次数（从消息中推导）"""
        state = self.chats.get(chat_name)
        if not state:
            return 0
        return sum(1 for m in state.messages if m.replied)
