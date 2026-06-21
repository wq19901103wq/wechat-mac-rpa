#!/usr/bin/env python3
"""WindowCapture 窗口异常恢复测试"""

import unittest
from unittest.mock import patch

from src.capture.window_capture import WeChatNotReadyError, WindowCapture, WindowNotFoundError


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
    @patch('src.capture.window_capture.subprocess.run')
    @patch('src.capture.window_capture.time.sleep')
    @patch('src.capture.window_capture.Quartz')
    @patch('src.capture.window_capture.AppKit')
    def test_small_window_triggers_activation_and_retry(
        self, mock_appkit, mock_quartz, mock_sleep, mock_subprocess, mock_validate
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

        mock_appkit.NSScreen.mainScreen.return_value.backingScaleFactor.return_value = 1.0

        capture = WindowCapture()
        result = capture.capture()

        # 应自动调用 activate WeChat
        mock_subprocess.assert_any_call(
            ['osascript', '-e', 'tell application "WeChat" to activate'],
            timeout=3, capture_output=True
        )
        # 应 sleep 等待
        mock_sleep.assert_called()
        # 最终结果应为正常窗口
        self.assertEqual(result.window_rect.width, 1760)
        self.assertEqual(result.window_rect.height, 1280)

    @patch('src.capture.window_capture.subprocess.run')
    @patch('src.capture.window_capture.time.sleep')
    @patch('src.capture.window_capture.Quartz')
    @patch('src.capture.window_capture.AppKit')
    def test_persistent_small_window_raises_not_ready(
        self, mock_appkit, mock_quartz, mock_sleep, mock_subprocess
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

        mock_appkit.NSScreen.mainScreen.return_value.backingScaleFactor.return_value = 1.0

        capture = WindowCapture()
        with self.assertRaises(WeChatNotReadyError) as ctx:
            capture.capture()

        self.assertIn("扫码", str(ctx.exception))
        mock_subprocess.assert_any_call(
            ['osascript', '-e', 'tell application "WeChat" to activate'],
            timeout=3, capture_output=True
        )

    @patch('src.capture.window_capture.subprocess.run')
    @patch('src.capture.window_capture.Quartz')
    @patch('src.capture.window_capture.AppKit')
    def test_no_window_raises_window_not_found(
        self, mock_appkit, mock_quartz, mock_subprocess
    ):
        """完全找不到微信窗口时保持原有行为"""
        mock_quartz.CGWindowListCopyWindowInfo.return_value = []
        mock_quartz.kCGWindowListOptionOnScreenOnly = 1
        mock_quartz.kCGWindowListExcludeDesktopElements = 2
        mock_quartz.kCGNullWindowID = 0
        mock_quartz.kCGWindowOwnerName = 'kCGWindowOwnerName'
        mock_quartz.kCGWindowBounds = 'kCGWindowBounds'

        mock_appkit.NSScreen.mainScreen.return_value.backingScaleFactor.return_value = 1.0

        capture = WindowCapture()
        with self.assertRaises(WindowNotFoundError):
            capture.capture()

    @patch('src.capture.window_capture.subprocess.run')
    @patch('src.capture.window_capture.time.sleep')
    @patch('src.capture.window_capture.Quartz')
    @patch('src.capture.window_capture.AppKit')
    def test_small_window_raises_not_ready(
        self, mock_appkit, mock_quartz, mock_sleep, mock_subprocess
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

        mock_appkit.NSScreen.mainScreen.return_value.backingScaleFactor.return_value = 1.0

        capture = WindowCapture()
        with self.assertRaises(WeChatNotReadyError) as ctx:
            capture.capture()

        self.assertIn("扫码登录", str(ctx.exception))
