#!/usr/bin/env python3
"""微信登录恢复处理器

当 WindowCapture 检测到窗口尺寸异常（未登录/浮窗）时，
尝试自动点击登录按钮；若仍无法恢复，则提示用户扫码。
"""

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import Quartz

from src.action.system_automation import MacOSSystemAutomation, SystemAutomation
from src.models.base import OCRTextElement, Rect
from src.ocr.vision_ocr import VisionOCREngine

_logger = logging.getLogger("src.login_recovery")


class LoginRecoveryStatus(Enum):
    SUCCESS = "success"                   # 已恢复为主窗口
    NEEDS_PHONE_CONFIRM = "needs_phone_confirm"  # 已点击登录，等待手机上确认
    NEEDS_QRCODE = "needs_qrcode"         # 需要用户扫码或手动点击
    NO_LOGIN_BUTTON = "no_login_button"   # 未找到登录按钮


@dataclass
class LoginRecoveryResult:
    status: LoginRecoveryStatus
    message: str


class WeChatLoginHandler:
    """处理微信未登录状态下的自动恢复"""

    def __init__(
        self,
        capture_output: Optional[str] = None,
        login_keywords: Optional[List[str]] = None,
        min_effective_width: int = 800,
        min_effective_height: int = 600,
        automation: SystemAutomation | None = None,
    ):
        if capture_output is None:
            capture_output = os.path.join(tempfile.gettempdir(), "wechat_login_capture.png")
        self.capture_output = capture_output
        self.login_keywords = login_keywords or ["登录", "进入微信", "确认登录"]
        self.min_effective_width = min_effective_width
        self.min_effective_height = min_effective_height
        self.automation = automation or MacOSSystemAutomation()
        self.ocr = VisionOCREngine()

    def _find_window(self) -> Optional[Rect]:
        """查找面积最大的微信窗口"""
        window_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly |
            Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID
        )

        best: Optional[Rect] = None
        best_area = 0
        for window in window_list:
            owner = window.get(Quartz.kCGWindowOwnerName, '')
            if owner in ('WeChat', '微信'):
                bounds = window.get(Quartz.kCGWindowBounds, {})
                width = int(bounds.get('Width', 0))
                height = int(bounds.get('Height', 0))
                if width > 200 and height > 200:
                    area = width * height
                    if area > best_area:
                        best_area = area
                        best = Rect(
                            x=int(bounds.get('X', 0)),
                            y=int(bounds.get('Y', 0)),
                            width=width,
                            height=height,
                        )
        return best

    def _capture_window(self, rect: Rect) -> str:
        """截图指定窗口"""
        ok, err = self.automation.capture_screen(rect, self.capture_output)
        if not ok:
            raise RuntimeError(f"截图失败: {err}")
        return self.capture_output

    def _detect_login_button(self, elements: List[OCRTextElement]) -> Optional[Rect]:
        """从 OCR 结果中查找登录按钮"""
        for elem in elements:
            text = elem.text.strip()
            for keyword in self.login_keywords:
                if keyword in text:
                    return elem.bbox
        return None

    def _is_phone_confirm_state(self, elements: List[OCRTextElement]) -> bool:
        """判断是否已处于'等待手机确认登录'状态"""
        for elem in elements:
            if "需在手机上完成登录" in elem.text or "请在手机上确认" in elem.text:
                return True
        return False

    def _click_login_button(self, window_rect: Rect, btn_rect: Rect) -> bool:
        """
        尝试点击登录按钮。

        注意：macOS 及微信的安全机制可能阻止外部模拟点击，
        因此此函数仅做最佳-effort 尝试，不保证一定成功。
        """
        try:
            center_x = window_rect.x + btn_rect.x + btn_rect.width // 2
            center_y = window_rect.y + btn_rect.y + btn_rect.height // 2

            # 方法 1: AppleScript 点击（需要辅助功能权限）
            script = f'''
                tell application "System Events"
                    tell process "WeChat"
                        click at {{{center_x}, {center_y}}}
                    end tell
                end tell
            '''
            rc, _, _ = self.automation.run_applescript(script, timeout=5)
            return rc == 0
        except (TypeError, AttributeError) as e:
            _logger.warning("_click_login_button 坐标计算异常: %s", e)
            return False

    def handle(self) -> LoginRecoveryResult:
        """
        主流程：
        1. 查找微信窗口
        2. 截图并 OCR
        3. 如果已经是'等待手机确认'状态，直接返回 NEEDS_PHONE_CONFIRM
        4. 点击登录按钮
        5. 等待后再次检查窗口尺寸
        6. 若变大返回 SUCCESS；若显示'手机确认'返回 NEEDS_PHONE_CONFIRM；
           否则提示用户手动干预
        """
        window_rect = self._find_window()
        if window_rect is None:
            return LoginRecoveryResult(
                status=LoginRecoveryStatus.NO_LOGIN_BUTTON,
                message="未找到微信窗口，无法自动恢复"
            )

        # 截图 + OCR
        image_path = self._capture_window(window_rect)
        elements = self.ocr.recognize(image_path)

        # 如果已经是等待手机确认状态，不需要再点击
        if self._is_phone_confirm_state(elements):
            return LoginRecoveryResult(
                status=LoginRecoveryStatus.NEEDS_PHONE_CONFIRM,
                message="已检测到'请在手机上确认'提示，请在手机上确认登录"
            )

        button_rect = self._detect_login_button(elements)

        if button_rect is None:
            return LoginRecoveryResult(
                status=LoginRecoveryStatus.NO_LOGIN_BUTTON,
                message="未检测到登录按钮，无法自动恢复"
            )

        # 点击登录按钮（最佳-effort）
        clicked = self._click_login_button(window_rect, button_rect)

        # 等待窗口响应（给手机上确认留足时间）
        time.sleep(3.0)

        # 再次检查窗口
        new_rect = self._find_window()
        if new_rect is not None and (
            new_rect.width >= self.min_effective_width
            and new_rect.height >= self.min_effective_height
        ):
            return LoginRecoveryResult(
                status=LoginRecoveryStatus.SUCCESS,
                message="微信已恢复为主窗口"
            )

        # 窗口仍小，再次 OCR 判断是否已进入手机确认流程
        image_path2 = self._capture_window(new_rect or window_rect)
        elements2 = self.ocr.recognize(image_path2)
        if self._is_phone_confirm_state(elements2):
            return LoginRecoveryResult(
                status=LoginRecoveryStatus.NEEDS_PHONE_CONFIRM,
                message="已尝试自动点击登录按钮，检测到'请在手机上确认'提示，请在手机上确认登录"
            )

        # 提示用户手动干预
        if clicked:
            return LoginRecoveryResult(
                status=LoginRecoveryStatus.NEEDS_QRCODE,
                message="已尝试自动点击登录按钮，若未自动登录，请手动点击「登录」按钮或在手机上确认"
            )
        else:
            return LoginRecoveryResult(
                status=LoginRecoveryStatus.NEEDS_QRCODE,
                message="已激活微信，但自动登录点击未生效，请手动点击「登录」按钮或在手机上确认"
            )
