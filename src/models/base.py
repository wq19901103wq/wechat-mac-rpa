#!/usr/bin/env python3
"""L1 Domain Models - 基础数据类型"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, List, Optional

# 媒体消息类型常量，供各模块统一引用
MEDIA_MESSAGE_TYPES: FrozenSet[str] = frozenset({"image", "sticker", "mixed", "link_card", "video"})


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass
class OCRTextElement:
    """OCR 原始输出元素"""
    text: str
    bbox: Rect           # 外接矩形
    center: Point        # 中心点，用于位置判断
    confidence: float


class SenderType(Enum):
    SELF = "self"
    OTHER = "other"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class ChatMessage:
    """领域模型：一条聊天消息（自带回复状态）"""
    text: str
    sender: str
    sender_type: SenderType
    chat_name: str
    chatroom_id: Optional[str] = None  # 微信 chatroom_id / wxid，用于区分同名群
    is_at_me: bool = False
    timestamp: Optional[str] = None
    source_elements: Optional[List[OCRTextElement]] = None

    # === 消息级回复状态 ===
    replied: bool = False              # 是否已回复
    reply_text: str = ""              # 回复内容
    reply_time: Optional[float] = None # 回复时间戳

    # === 图片/表情相关 ===
    message_type: str = "text"         # "text" / "image" / "sticker" / "mixed" / "link_card" / "video"
    image_description: str = ""        # 视觉模型对图片内容的描述
    image_text: str = ""               # 图片上的文字（如有）
    is_image_duplicate: bool = False   # 是否被去重标记（原始描述仍保留在 image_description 中）

    # === 引用/回复 ===
    quoted_text: str = ""             # 被引用的消息内容（微信"引用"功能提取的文字/[图片]/[表情]等）

    # === 多账号标记 ===
    account: str = ""                  # 消息来源账号标识（如 "work"、"personal"），空字符串=主账号

    # === WeFlow 专属字段（可选，不影响现有代码）===
    local_id: Optional[int] = None     # WeFlow 数据库 localId
    server_id: Optional[str] = None    # WeFlow 数据库 serverId
    create_time: Optional[int] = None  # 秒级时间戳
    raw_type: Optional[int] = None     # WeFlow localType
    sender_wxid: Optional[str] = None  # 原始 sender wxid


@dataclass
class ActionResult:
    success: bool
    sent_text: Optional[str] = None
    error: Optional[str] = None
    # skipped=True 表示静默模式主动跳过发送（非真实失败）。
    # 调用方据此区分：静默跳过应 mark_replied 避免卡循环，真实失败不 mark_replied。
    skipped: bool = False


@dataclass
class ChatListItem:
    """左侧聊天列表项，属于 Domain Model，被 Layout 和 UIInteractor 共用"""
    nickname: str
    last_message_preview: str
    unread_count: str
    timestamp: str
    rect: Rect  # 列表项在屏幕上的位置，供 UIInteractor 点击使用


@dataclass
class PerceptionResult:
    """VisionPipeline 的输出，对 Bot 层隐藏所有视觉实现细节"""
    chat_name: str
    messages: List[ChatMessage]
    chat_list_items: List[ChatListItem]
    screenshot_path: str
    is_group: bool = False  # 感知层统一判断，下游只读不再判断
    window_rect: Optional[Rect] = None  # 窗口屏幕逻辑坐标
    scale_factor: float = 1.0  # Retina 缩放因子
    debug_info: Optional[Dict] = None  # 完整调试信息（tick 级）
    is_service_account_list: bool = False  # 当前是否为服务号/订阅号/公众号列表
