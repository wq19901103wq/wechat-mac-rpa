#!/usr/bin/env python3
"""WindowCapture 窗口异常恢复测试"""

import unittest
from unittest.mock import patch

from src.action.system_automation import SystemAutomation
from src.capture.window_capture import WeChatNotReadyError, WindowCapture, WindowNotFoundError
from src.models.base import Rect


class MockCaptureAutomation(SystemAutomation):
    """可编程 SystemAutomation mock，用于 WindowCapture 测试。"""

    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []
        self.capture_success = True
        self.capture_error = ""

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
        return 0, "", ""

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
        return self.capture_success, self.capture_error


class TestWindowCaptureRecovery(unittest.TestCase):
    """测试 WindowCapture 对异常小窗口的恢复逻辑"""

    def _make_mock_window(self, owner, x, y, width, height, window_id=1):
        return {
            'kCGWindowOwnerName': owner,
            'kCGWindowOwnerPID': 12345,
            'kCGWindowBounds': {
                'X': x, 'Y': y, 'Width': width, 'Height': height
            },
            'kCGWindowNumber': window_id,
        }

    @patch.object(WindowCapture, '_validate_wechat_screenshot', return_value=True)
    @patch('src.capture.window_capture.time.sleep')
    @patch('src.capture.window_capture.Quartz')
    @patch('src.capture.window_capture.AppKit')
    def test_small_window_triggers_activation_and_retry(
        self, mock_appkit, mock_quartz, mock_sleep, mock_validate
    ):
        """窗口尺寸过小时应自动激活微信并重试截图"""
        mock_quartz.CGWindowListCopyWindowInfo.side_effect = [
            [self._make_mock_window('微信', 500, 200, 560, 760)],
            [self._make_mock_window('微信', 100, 100, 1760, 1280)],
        ]
        mock_quartz.kCGWindowListOptionOnScreenOnly = 1
        mock_quartz.kCGWindowListExcludeDesktopElements = 2
        mock_quartz.kCGNullWindowID = 0
        mock_quartz.kCGWindowOwnerName = 'kCGWindowOwnerName'
        mock_quartz.kCGWindowBounds = 'kCGWindowBounds'
        mock_quartz.kCGWindowNumber = 'kCGWindowNumber'

        mock_appkit.NSScreen.mainScreen.return_value.backingScaleFactor.return_value = 1.0

        automation = MockCaptureAutomation()
        capture = WindowCapture(automation=automation)
        result = capture.capture()

        # 应自动调用 activate WeChat
        self.assertTrue(any(c[0] == "activate_app" and c[1] == ("WeChat",) for c in automation.calls))
        # 应 sleep 等待
        mock_sleep.assert_called()
        # 最终结果应为正常窗口
        self.assertEqual(result.window_rect.width, 1760)
        self.assertEqual(result.window_rect.height, 1280)

    @patch('src.capture.window_capture.time.sleep')
    @patch('src.capture.window_capture.Quartz')
    @patch('src.capture.window_capture.AppKit')
    def test_persistent_small_window_raises_not_ready(
        self, mock_appkit, mock_quartz, mock_sleep
    ):
        """激活重试后仍然只有小窗口时，应抛出 WeChatNotReadyError"""
        mock_quartz.CGWindowListCopyWindowInfo.return_value = [
            self._make_mock_window('微信', 500, 200, 560, 760),
        ]
        mock_quartz.kCGWindowListOptionOnScreenOnly = 1
        mock_quartz.kCGWindowListExcludeDesktopElements = 2
        mock_quartz.kCGNullWindowID = 0
        mock_quartz.kCGWindowOwnerName = 'kCGWindowOwnerName'
        mock_quartz.kCGWindowBounds = 'kCGWindowBounds'
        mock_quartz.kCGWindowNumber = 'kCGWindowNumber'

        mock_appkit.NSScreen.mainScreen.return_value.backingScaleFactor.return_value = 1.0

        automation = MockCaptureAutomation()
        capture = WindowCapture(automation=automation)
        with self.assertRaises(WeChatNotReadyError) as ctx:
            capture.capture()

        self.assertIn("扫码", str(ctx.exception))
        self.assertTrue(any(c[0] == "activate_app" and c[1] == ("WeChat",) for c in automation.calls))

    @patch('src.capture.window_capture.Quartz')
    @patch('src.capture.window_capture.AppKit')
    def test_no_window_raises_window_not_found(
        self, mock_appkit, mock_quartz
    ):
        """完全找不到微信窗口时保持原有行为"""
        mock_quartz.CGWindowListCopyWindowInfo.return_value = []
        mock_quartz.kCGWindowListOptionOnScreenOnly = 1
        mock_quartz.kCGWindowListExcludeDesktopElements = 2
        mock_quartz.kCGNullWindowID = 0
        mock_quartz.kCGWindowOwnerName = 'kCGWindowOwnerName'
        mock_quartz.kCGWindowBounds = 'kCGWindowBounds'
        mock_quartz.kCGWindowNumber = 'kCGWindowNumber'

        mock_appkit.NSScreen.mainScreen.return_value.backingScaleFactor.return_value = 1.0

        automation = MockCaptureAutomation()
        capture = WindowCapture(automation=automation)
        with self.assertRaises(WindowNotFoundError):
            capture.capture()

    @patch('src.capture.window_capture.time.sleep')
    @patch('src.capture.window_capture.Quartz')
    @patch('src.capture.window_capture.AppKit')
    def test_small_window_raises_not_ready(
        self, mock_appkit, mock_quartz, mock_sleep
    ):
        """小窗口时直接抛出 WeChatNotReadyError，由上层处理恢复"""
        mock_quartz.CGWindowListCopyWindowInfo.return_value = [
            self._make_mock_window('微信', 500, 200, 560, 760),
        ]
        mock_quartz.kCGWindowListOptionOnScreenOnly = 1
        mock_quartz.kCGWindowListExcludeDesktopElements = 2
        mock_quartz.kCGNullWindowID = 0
        mock_quartz.kCGWindowOwnerName = 'kCGWindowOwnerName'
        mock_quartz.kCGWindowBounds = 'kCGWindowBounds'
        mock_quartz.kCGWindowNumber = 'kCGWindowNumber'

        mock_appkit.NSScreen.mainScreen.return_value.backingScaleFactor.return_value = 1.0

        automation = MockCaptureAutomation()
        capture = WindowCapture(automation=automation)
        with self.assertRaises(WeChatNotReadyError) as ctx:
            capture.capture()

        self.assertIn("扫码登录", str(ctx.exception))
