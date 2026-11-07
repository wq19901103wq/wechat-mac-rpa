"""Shared perception models; sending results remain in RPA."""
from dataclasses import dataclass
from typing import Optional
from wechat_reader.models.base import (
    MEDIA_MESSAGE_TYPES, Point, Rect, OCRTextElement, SenderType, ChatMessage,
    ChatListItem, PerceptionResult,
)

@dataclass
class ActionResult:
    success: bool
    sent_text: Optional[str] = None
    error: Optional[str] = None
    # skipped=True 表示静默模式主动跳过发送（非真实失败）。
    # 调用方据此区分：静默跳过应 mark_replied 避免卡循环，真实失败不 mark_replied。
    skipped: bool = False
