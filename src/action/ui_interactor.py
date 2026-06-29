#!/usr/bin/env python3
"""L4 Action Layer - UI Interactor

基于坐标进行鼠标点击操作，用于切换聊天或聚焦输入框。
"""

import logging
from abc import ABC, abstractmethod

from src.models.base import ChatListItem

try:
    import pyautogui
except Exception:  # pragma: no cover
    pyautogui = None

_logger = logging.getLogger("src.ui_interactor")


class UIInteractor(ABC):
    """UI 交互器抽象基类"""

    @abstractmethod
    def click_chat_item(self, item: ChatListItem) -> bool:
        """点击左侧聊天列表中的某一项"""
        pass

    @abstractmethod
    def click_input_box(self) -> bool:
        """点击聊天窗口底部的输入框区域"""
        pass


class PyAutoGUIInteractor(UIInteractor):
    """基于 pyautogui 的 UI 交互实现"""

    # 输入框默认硬编码区域（相对于屏幕），实际使用应由调用方传入或后续配置化
    DEFAULT_INPUT_BOX_X = 800
    DEFAULT_INPUT_BOX_Y = 900

    def click_chat_item(self, item: ChatListItem) -> bool:
        """
        根据 ChatListItem.rect 计算中心点并点击。
        center_x = rect.x + rect.width / 2
        center_y = rect.y + rect.height / 2
        """
        try:
            center_x = item.rect.x + item.rect.width // 2
            center_y = item.rect.y + item.rect.height // 2
            pyautogui.click(center_x, center_y)
            return True
        except Exception as e:
            _logger.warning("click_chat_item 失败: %s", e)
            return False

    def click_input_box(self) -> bool:
        """点击输入框区域，默认使用硬编码坐标。"""
        try:
            pyautogui.click(self.DEFAULT_INPUT_BOX_X, self.DEFAULT_INPUT_BOX_Y)
            return True
        except Exception as e:
            _logger.warning("click_input_box 失败: %s", e)
            return False
