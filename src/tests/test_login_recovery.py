#!/usr/bin/env python3
"""登录恢复流程单元测试

通过 Mock SystemAutomation 验证 WeChatLoginHandler 的恢复逻辑，
不再直接依赖 subprocess.run。
"""

from unittest.mock import Mock, patch

import pytest

from src.action.login_recovery import LoginRecoveryStatus, WeChatLoginHandler
from src.action.system_automation import SystemAutomation
from src.models.base import OCRTextElement, Point, Rect


class MockSystemAutomation(SystemAutomation):
    """可编程的 SystemAutomation mock，用于测试登录恢复流程。"""

    def __init__(self, capture_success: bool = True, applescript_rc: int = 0):
        self.capture_success = capture_success
        self.applescript_rc = applescript_rc
        self.calls: list[tuple[str, tuple, dict]] = []

    def _log(self, name: str, *args, **kwargs):
        self.calls.append((name, args, kwargs))

    def activate_app(self, app_name: str) -> bool:
        self._log("activate_app", app_name)
        return True

    def get_frontmost_app(self, app_name: str) -> tuple[bool, str]:
        self._log("get_frontmost_app", app_name)
        return True, app_name

    def get_window_rect(self, app_name: str) -> tuple[bool, Rect | None, str]:
        self._log("get_window_rect", app_name)
        return True, Rect(x=0, y=0, width=800, height=600), ""

    def click_at(self, x: int, y: int) -> bool:
        self._log("click_at", x, y)
        return True

    def send_keys(self, key_spec: str) -> bool:
        self._log("send_keys", key_spec)
        return True

    def run_applescript(self, script: str, timeout: int = 5) -> tuple[int, str, str]:
        self._log("run_applescript", script, timeout=timeout)
        return self.applescript_rc, "", ""

    def set_clipboard_text(self, text: str) -> bool:
        self._log("set_clipboard_text", text)
        return True

    def get_clipboard_text(self) -> tuple[bool, str]:
        self._log("get_clipboard_text")
        return True, ""

    def capture_screen(
        self,
        rect: Rect,
        output_path: str,
        window_id: int | None = None,
    ) -> tuple[bool, str]:
        self._log("capture_screen", rect, output_path, window_id=window_id)
        return self.capture_success, "" if self.capture_success else "capture failed"


class TestWeChatLoginHandler:
    @pytest.fixture
    def handler(self, tmp_path):
        return WeChatLoginHandler(
            capture_output=str(tmp_path / "capture.png"),
            login_keywords=["登录", "进入微信", "确认登录"],
        )

    def test_detect_login_button_found(self, handler):
        """OCR 结果中包含登录关键词时应返回按钮位置"""
        elements = [
            OCRTextElement(
                text="进入微信", bbox=Rect(100, 300, 80, 30),
                center=Point(140, 315), confidence=0.95
            ),
        ]
        rect = handler._detect_login_button(elements)
        assert rect is not None
        assert rect.x == 100

    def test_detect_login_button_not_found(self, handler):
        """OCR 结果中无登录关键词时返回 None"""
        elements = [
            OCRTextElement(
                text="取消", bbox=Rect(100, 300, 40, 20),
                center=Point(120, 310), confidence=0.9
            ),
        ]
        rect = handler._detect_login_button(elements)
        assert rect is None

    def test_detect_phone_confirm_state(self, handler):
        """检测到'需在手机上完成登录'时返回 True"""
        elements = [
            OCRTextElement(
                text="需在手机上完成登录", bbox=Rect(50, 200, 200, 30),
                center=Point(150, 215), confidence=0.95
            ),
        ]
        assert handler._is_phone_confirm_state(elements) is True

    def test_detect_phone_confirm_state_false(self, handler):
        """无手机确认文本时返回 False"""
        elements = [
            OCRTextElement(
                text="登录", bbox=Rect(100, 300, 80, 30),
                center=Point(140, 315), confidence=0.95
            ),
        ]
        assert handler._is_phone_confirm_state(elements) is False

    @patch("sys.platform", "darwin")
    def test_click_login_button_success(self):
        """点击登录按钮成功（macOS 分支：AppleScript 点击）"""
        automation = MockSystemAutomation()
        handler = WeChatLoginHandler(automation=automation)
        window_rect = Rect(x=500, y=200, width=280, height=380)
        btn_rect = Rect(x=100, y=300, width=80, height=30)
        result = handler._click_login_button(window_rect, btn_rect)
        assert result is True
        assert any(c[0] == "run_applescript" for c in automation.calls)
        script = next(c[1][0] for c in automation.calls if c[0] == "run_applescript")
        # 中心坐标：500+100+40=640, 200+300+15=515
        assert '640' in script
        assert '515' in script

    @patch("sys.platform", "darwin")
    def test_click_login_button_failure(self):
        """AppleScript 异常时返回 False（macOS 分支）"""
        automation = MockSystemAutomation(applescript_rc=1)
        handler = WeChatLoginHandler(automation=automation)
        window_rect = Rect(x=500, y=200, width=280, height=380)
        btn_rect = Rect(x=100, y=300, width=80, height=30)
        result = handler._click_login_button(window_rect, btn_rect)
        assert result is False

    @patch("sys.platform", "win32")
    def test_click_login_button_windows(self):
        """Windows 分支：登录按钮点击走坐标 click_at"""
        automation = MockSystemAutomation()
        handler = WeChatLoginHandler(automation=automation)
        window_rect = Rect(x=500, y=200, width=280, height=380)
        btn_rect = Rect(x=100, y=300, width=80, height=30)
        result = handler._click_login_button(window_rect, btn_rect)
        assert result is True
        assert any(c[0] == "click_at" for c in automation.calls)
        call = next(c for c in automation.calls if c[0] == "click_at")
        # 中心坐标：500+100+40=640, 200+300+15=515
        assert call[1][0] == 640
        assert call[1][1] == 515

    @patch.object(WeChatLoginHandler, "_capture_window")
    @patch.object(WeChatLoginHandler, "_find_window")
    @patch("src.action.login_recovery.time.sleep")
    def test_handle_success_after_click(
        self, mock_sleep, mock_find, mock_capture, tmp_path
    ):
        """点击登录后窗口恢复正常，返回 SUCCESS"""
        automation = MockSystemAutomation()
        capture_path = str(tmp_path / "capture.png")
        mock_capture.return_value = capture_path

        # 第一次查找：小窗口（带"登录"按钮）；第二次查找：正常大窗口
        mock_find.side_effect = [
            Rect(x=500, y=200, width=280, height=380),
            Rect(x=100, y=100, width=1760, height=1280),
        ]

        handler = WeChatLoginHandler(capture_output=capture_path, automation=automation)
        # Mock OCR: 第一次返回"登录"按钮
        handler.ocr.recognize = Mock(return_value=[
            OCRTextElement(
                text="登录", bbox=Rect(100, 300, 80, 30),
                center=Point(140, 315), confidence=0.95
            ),
        ])

        result = handler.handle()

        assert result.status == LoginRecoveryStatus.SUCCESS
        assert result.message == "微信已恢复为主窗口"

    @patch.object(WeChatLoginHandler, "_capture_window")
    @patch.object(WeChatLoginHandler, "_find_window")
    @patch("src.action.login_recovery.time.sleep")
    def test_handle_needs_phone_confirm_after_click(
        self, mock_sleep, mock_find, mock_capture, tmp_path
    ):
        """点击登录后窗口仍小但显示'需在手机上完成登录'，返回 NEEDS_PHONE_CONFIRM"""
        automation = MockSystemAutomation()
        capture_path = str(tmp_path / "capture.png")
        mock_capture.return_value = capture_path

        mock_find.return_value = Rect(x=500, y=200, width=280, height=380)

        handler = WeChatLoginHandler(capture_output=capture_path, automation=automation)
        # Mock OCR: 第一次返回"登录"按钮；第二次返回手机确认提示
        handler.ocr.recognize = Mock(side_effect=[
            [
                OCRTextElement(
                    text="登录", bbox=Rect(100, 300, 80, 30),
                    center=Point(140, 315), confidence=0.95
                ),
            ],
            [
                OCRTextElement(
                    text="需在手机上完成登录", bbox=Rect(50, 200, 200, 30),
                    center=Point(150, 215), confidence=0.95
                ),
                OCRTextElement(
                    text="取消", bbox=Rect(100, 350, 40, 20),
                    center=Point(120, 360), confidence=0.9
                ),
            ],
        ])

        result = handler.handle()

        assert result.status == LoginRecoveryStatus.NEEDS_PHONE_CONFIRM
        assert "手机上确认" in result.message

    @patch.object(WeChatLoginHandler, "_capture_window")
    @patch.object(WeChatLoginHandler, "_find_window")
    @patch("src.action.login_recovery.time.sleep")
    def test_handle_needs_qrcode_after_click(
        self, mock_sleep, mock_find, mock_capture, tmp_path
    ):
        """点击登录后窗口仍小且无手机确认提示，返回 NEEDS_QRCODE"""
        automation = MockSystemAutomation()
        capture_path = str(tmp_path / "capture.png")
        mock_capture.return_value = capture_path

        mock_find.return_value = Rect(x=500, y=200, width=280, height=380)

        handler = WeChatLoginHandler(capture_output=capture_path, automation=automation)
        # Mock OCR: 第一次返回"登录"按钮；第二次仍无手机确认提示
        handler.ocr.recognize = Mock(side_effect=[
            [
                OCRTextElement(
                    text="登录", bbox=Rect(100, 300, 80, 30),
                    center=Point(140, 315), confidence=0.95
                ),
            ],
            [],
        ])

        result = handler.handle()

        assert result.status == LoginRecoveryStatus.NEEDS_QRCODE
        assert "手机上确认" in result.message or "扫码" in result.message

    @patch.object(WeChatLoginHandler, "_capture_window")
    @patch.object(WeChatLoginHandler, "_find_window")
    def test_handle_no_login_button(
        self, mock_find, mock_capture, tmp_path
    ):
        """窗口小但找不到登录按钮，返回 NO_LOGIN_BUTTON"""
        automation = MockSystemAutomation()
        capture_path = str(tmp_path / "capture.png")
        mock_capture.return_value = capture_path

        mock_find.return_value = Rect(x=500, y=200, width=280, height=380)

        handler = WeChatLoginHandler(capture_output=capture_path, automation=automation)
        handler.ocr.recognize = Mock(return_value=[])

        result = handler.handle()

        assert result.status == LoginRecoveryStatus.NO_LOGIN_BUTTON
