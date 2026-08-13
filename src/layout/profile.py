#!/usr/bin/env python3
"""L2 LayoutProfile - 布局配置"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class LayoutProfile:
    """
    针对特定微信版本 + 分辨率的布局配置。
    当微信更新或窗口缩放异常时，优先调整此配置。
    """
    name: str
    window_width: int           # 适配窗口宽度
    window_height: int          # 适配窗口高度

    # 区域边界（像素）
    left_boundary: int          # 聊天列表右边界
    chat_list_x_max: int        # 聊天列表最大 x
    title_y_max: int            # 标题栏底部
    title_x_max_ratio: float    # 标题栏右侧比例上限
    input_y_min: int            # 输入框顶部

    # 颜色检测
    self_green: Tuple[int, int, int]
    self_green_tolerance: int
    min_bubble_pixels: int      # 气泡最小像素数
    # 昵称识别区域（相对坐标 0.0-1.0）
    nickname_x_min_ratio: float
    nickname_x_max_ratio: float
    nickname_y_offset_min: int
    nickname_y_offset_max: int
    message_cluster_threshold: int = 80  # 消息按 y 聚类的阈值（像素）


# 预配置实例
PROFILE_WECHAT_MAC_1760X1280 = LayoutProfile(
    name="wechat_mac_4.1.8_1760x1280",
    window_width=1760,
    window_height=1280,
    left_boundary=480,
    chat_list_x_max=360,
    title_y_max=70,  # 微信标题栏实际高度相对固定（约 50-60px），不应随窗口高度线性缩放
    title_x_max_ratio=0.95,
    input_y_min=1040,
    self_green=(176, 240, 167),
    self_green_tolerance=35,
    min_bubble_pixels=1000,
    message_cluster_threshold=80,
    nickname_x_min_ratio=0.30,
    nickname_x_max_ratio=0.45,
    nickname_y_offset_min=15,
    nickname_y_offset_max=50,
)
