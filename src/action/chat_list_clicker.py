#!/usr/bin/env python3
"""L5 Action - 点击聊天列表项切换窗口

将 OCR/Layouter 识别的 ChatListItem 转换为屏幕点击动作。
坐标计算规则：
- item.rect 是截图中的像素坐标（Retina 实际像素）
- window_rect 是屏幕逻辑坐标（AppleScript 报告的尺寸）
- screen_abs = window_rect + item_rect / scale_factor
"""

import logging
import subprocess
import time
from typing import Optional

from src.models.base import ChatListItem, Rect

_logger = logging.getLogger("src.chat_list_clicker")


class ChatListClicker:
    """点击左侧聊天列表中的指定项，切换当前聊天窗口。"""

    # 类级全局冷却：任意两次真实点击之间至少间隔这么多秒，
    # 防止 bot 在异常状态下高频连点导致微信窗口布局错乱。
    MIN_CLICK_INTERVAL_SECONDS = 1.0
    _last_click_time: float = 0.0

    def __init__(self, window_rect: Rect, scale_factor: float = 2.0):
        self.window_rect = window_rect
        self.scale_factor = scale_factor

    def _can_click(self) -> bool:
        """检查是否距离上次点击已超过最小冷却时间。"""
        now = time.time()
        elapsed = now - ChatListClicker._last_click_time
        if elapsed < ChatListClicker.MIN_CLICK_INTERVAL_SECONDS:
            _logger.warning(
                f"点击被全局冷却跳过: 距离上次点击仅 {elapsed:.2f}s, "
                f"需间隔 {ChatListClicker.MIN_CLICK_INTERVAL_SECONDS}s"
            )
            return False
        return True

    def click_item(self, item: ChatListItem) -> bool:
        """
        点击聊天列表项的中心位置。

        策略：
        1. 先检查全局冷却，避免 1 秒内连续点击
        2. 激活微信窗口（确保有焦点）
        3. 点击位置取列表项中心（rect 包含昵称+预览，x 偏移确保在条目内）
        4. 点击后等待右侧展开，避免快速连续点击导致误触

        Args:
            item: 要点击的 ChatListItem，包含 rect（截图像素坐标）

        Returns:
            True 如果点击命令执行成功
        """
        if not self._can_click():
            return False

        # 点击位置：取 rect 中心，避免偏左偏右点到相邻项
        click_x = item.rect.x + item.rect.width // 2
        click_y = item.rect.y + item.rect.height // 2

        abs_x = int(self.window_rect.x + click_x / self.scale_factor)
        abs_y = int(self.window_rect.y + click_y / self.scale_factor)

        try:
            # Step 1: 强制置顶微信窗口
            subprocess.run(
                ["osascript", "-e",
                 'tell application "System Events" to tell process "WeChat" to set frontmost to true'],
                timeout=3,
                capture_output=True,
                check=True,
            )
            time.sleep(0.5)
            # Step 2: 点击聊天列表项
            subprocess.run(
                ["/opt/homebrew/bin/cliclick", f"c:{abs_x},{abs_y}"],
                check=True,
                timeout=5,
            )
            ChatListClicker._last_click_time = time.time()
            # Step 3: 等待右侧聊天内容加载
            time.sleep(2.5)
            _logger.info(
                f"点击聊天列表: screen=({abs_x},{abs_y}) "
                f"window=({self.window_rect.x},{self.window_rect.y}) "
                f"rect_in_screenshot=({click_x},{click_y}) scale={self.scale_factor}"
            )
            return True
        except Exception:
            return False

    # 服务号/订阅号/公众号列表返回按钮的默认偏移（截图像素，相对窗口内容左上角）。
    # 实测标题栏显示 "＜ 服务号"，返回箭头在标题栏左侧，中心约 (165,140)。
    BACK_BUTTON_OFFSET_X = 165
    BACK_BUTTON_OFFSET_Y = 140

    def click_back_button(self) -> bool:
        """点击窗口左上角返回按钮，用于从服务号/订阅号/公众号列表返回聊天视图。"""
        if not self._can_click():
            return False

        click_x = int(self.window_rect.x + self.BACK_BUTTON_OFFSET_X / self.scale_factor)
        click_y = int(self.window_rect.y + self.BACK_BUTTON_OFFSET_Y / self.scale_factor)

        try:
            # Step 1: 强制置顶微信窗口
            subprocess.run(
                ["osascript", "-e",
                 'tell application "System Events" to tell process "WeChat" to set frontmost to true'],
                timeout=3,
                capture_output=True,
                check=True,
            )
            time.sleep(0.3)
            # Step 2: 点击返回按钮
            subprocess.run(
                ["/opt/homebrew/bin/cliclick", f"c:{click_x},{click_y}"],
                check=True,
                timeout=5,
            )
            ChatListClicker._last_click_time = time.time()
            # Step 3: 等待页面返回动画
            time.sleep(0.8)
            _logger.info(
                f"点击返回按钮: screen=({click_x},{click_y}) "
                f"window=({self.window_rect.x},{self.window_rect.y}) "
                f"offset=({self.BACK_BUTTON_OFFSET_X},{self.BACK_BUTTON_OFFSET_Y}) "
                f"scale={self.scale_factor}"
            )
            return True
        except Exception:
            return False

    def click_by_index(self, items: list[ChatListItem], index: int) -> bool:
        """按索引点击列表项。"""
        if 0 <= index < len(items):
            return self.click_item(items[index])
        return False

    def click_first_unread(
        self, items: list[ChatListItem], exclude_nickname: Optional[str] = None
    ) -> Optional[ChatListItem]:
        """
        点击第一个有未读消息的聊天项（排除当前已打开的聊天）。

        Args:
            items: 聊天列表项
            exclude_nickname: 要排除的聊天名称（通常是当前聊天）

        Returns:
            被点击的 ChatListItem，如果没有则返回 None
        """
        for item in items:
            if not item.unread_count:
                continue
            if exclude_nickname and item.nickname == exclude_nickname:
                continue
            if self.click_item(item):
                return item
        return None
