#!/usr/bin/env python3
"""L4 Reply Policy - 回复决策."""

from typing import Any

from src.models.base import ChatMessage, SenderType


class ReplyPolicy:
    def __init__(self, require_at_in_group: bool = False):
        self.require_at_in_group = require_at_in_group

    def should_reply(self, msg: ChatMessage, session: Any) -> bool:
        """
        所有回复判断交给 AI 自主决定（输出 replies: [] 表示不回复）。
        代码层只做最基本的过滤。
        """
        if msg.sender_type == SenderType.SELF:
            return False
        if msg.sender_type == SenderType.SYSTEM:
            return False
        return True
