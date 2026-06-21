#!/usr/bin/env python3
"""Tests for L2 WindowCapture module."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.action.system_automation import SystemAutomation
from src.capture.window_capture import CaptureResult, WindowCapture, WindowNotFoundError
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


class TestWindowCapture(unittest.TestCase):
    """Test WindowCapture with mocked Quartz and SystemAutomation."""

    def setUp(self):
        self.automation = MockCaptureAutomation()
        self.capture = WindowCapture(automation=self.automation)

    def _make_mock_window(self, owner, x, y, width, height, window_id=1):
        """Helper to build a Quartz window info dict."""
        return {
            'kCGWindowOwnerName': owner,
            'kCGWindowBounds': {
                'X': x,
                'Y': y,
                'Width': width,
                'Height': height,
            },
            'kCGWindowNumber': window_id,
        }

    @patch.object(WindowCapture, '_validate_wechat_screenshot', return_value=True)
    @patch('src.capture.window_capture.Quartz')
    @patch('src.capture.window_capture.AppKit')
    def test_capture_success_wechat_en(self, mock_appkit, mock_quartz, mock_validate):
        """Successful capture of English-named WeChat window."""
        mock_quartz.CGWindowListCopyWindowInfo.return_value = [
            self._make_mock_window('Safari', 0, 0, 1200, 800),
            self._make_mock_window('WeChat', 100, 200, 1200, 900),
        ]
        mock_quartz.kCGWindowListOptionOnScreenOnly = 1
        mock_quartz.kCGWindowListExcludeDesktopElements = 2
        mock_quartz.kCGNullWindowID = 0
        mock_quartz.kCGWindowOwnerName = 'kCGWindowOwnerName'
        mock_quartz.kCGWindowBounds = 'kCGWindowBounds'
        mock_quartz.kCGWindowNumber = 'kCGWindowNumber'
        mock_quartz.kCGWindowNumber = 'kCGWindowNumber'

        mock_screen = MagicMock()
        mock_screen.backingScaleFactor.return_value = 1.0
        mock_appkit.NSScreen.mainScreen.return_value = mock_screen

        result = self.capture.capture()

        self.assertIsInstance(result, CaptureResult)
        self.assertEqual(result.image_path, self.capture.output_path)
        self.assertEqual(result.window_rect, Rect(x=100, y=200, width=1200, height=900))
        self.assertEqual(result.scale_factor, 1.0)

        capture_calls = [c for c in self.automation.calls if c[0] == "capture_screen"]
        self.assertEqual(len(capture_calls), 1)
        _, args, kwargs = capture_calls[0]
        self.assertEqual(args[0], Rect(x=100, y=200, width=1200, height=900))
        self.assertEqual(args[1], self.capture.output_path)
        self.assertEqual(kwargs.get("window_id"), 1)

    @patch.object(WindowCapture, '_validate_wechat_screenshot', return_value=True)
    @patch('src.capture.window_capture.Quartz')
    @patch('src.capture.window_capture.AppKit')
    def test_capture_success_wechat_cn(self, mock_appkit, mock_quartz, mock_validate):
        """Successful capture of Chinese-named WeChat window."""
        mock_quartz.CGWindowListCopyWindowInfo.return_value = [
            self._make_mock_window('微信', 50, 100, 1760, 1280),
        ]
        mock_quartz.kCGWindowListOptionOnScreenOnly = 1
        mock_quartz.kCGWindowListExcludeDesktopElements = 2
        mock_quartz.kCGNullWindowID = 0
        mock_quartz.kCGWindowOwnerName = 'kCGWindowOwnerName'
        mock_quartz.kCGWindowBounds = 'kCGWindowBounds'
        mock_quartz.kCGWindowNumber = 'kCGWindowNumber'

        mock_screen = MagicMock()
        mock_screen.backingScaleFactor.return_value = 2.0
        mock_appkit.NSScreen.mainScreen.return_value = mock_screen

        result = self.capture.capture()

        self.assertIsInstance(result, CaptureResult)
        self.assertEqual(result.window_rect, Rect(x=50, y=100, width=1760, height=1280))
        self.assertEqual(result.scale_factor, 2.0)

    @patch('src.capture.window_capture.Quartz')
    def test_window_not_found_no_wechat(self, mock_quartz):
        """Raise WindowNotFoundError when no WeChat window exists."""
        mock_quartz.CGWindowListCopyWindowInfo.return_value = [
            self._make_mock_window('Safari', 0, 0, 1200, 800),
            self._make_mock_window('Finder', 0, 0, 800, 600),
        ]
        mock_quartz.kCGWindowListOptionOnScreenOnly = 1
        mock_quartz.kCGWindowListExcludeDesktopElements = 2
        mock_quartz.kCGNullWindowID = 0
        mock_quartz.kCGWindowOwnerName = 'kCGWindowOwnerName'
        mock_quartz.kCGWindowBounds = 'kCGWindowBounds'
        mock_quartz.kCGWindowNumber = 'kCGWindowNumber'

        with self.assertRaises(WindowNotFoundError):
            self.capture.capture()

    @patch('src.capture.window_capture.Quartz')
    def test_window_not_found_too_small(self, mock_quartz):
        """Raise WindowNotFoundError when WeChat window is too small."""
        mock_quartz.CGWindowListCopyWindowInfo.return_value = [
            self._make_mock_window('WeChat', 0, 0, 199, 199),
        ]
        mock_quartz.kCGWindowListOptionOnScreenOnly = 1
        mock_quartz.kCGWindowListExcludeDesktopElements = 2
        mock_quartz.kCGNullWindowID = 0
        mock_quartz.kCGWindowOwnerName = 'kCGWindowOwnerName'
        mock_quartz.kCGWindowBounds = 'kCGWindowBounds'
        mock_quartz.kCGWindowNumber = 'kCGWindowNumber'

        with self.assertRaises(WindowNotFoundError):
            self.capture.capture()

    @patch.object(WindowCapture, '_validate_wechat_screenshot', return_value=True)
    @patch('src.capture.window_capture.Quartz')
    @patch('src.capture.window_capture.AppKit')
    def test_capture_uses_largest_matching_window(self, mock_appkit, mock_quartz, mock_validate):
        """When multiple WeChat windows exist, the largest one is selected."""
        mock_quartz.CGWindowListCopyWindowInfo.return_value = [
            self._make_mock_window('WeChat', 10, 20, 1200, 900, window_id=1),
            self._make_mock_window('WeChat', 30, 40, 1400, 1000, window_id=2),
        ]
        mock_quartz.kCGWindowListOptionOnScreenOnly = 1
        mock_quartz.kCGWindowListExcludeDesktopElements = 2
        mock_quartz.kCGNullWindowID = 0
        mock_quartz.kCGWindowOwnerName = 'kCGWindowOwnerName'
        mock_quartz.kCGWindowBounds = 'kCGWindowBounds'
        mock_quartz.kCGWindowNumber = 'kCGWindowNumber'

        mock_screen = MagicMock()
        mock_screen.backingScaleFactor.return_value = 1.0
        mock_appkit.NSScreen.mainScreen.return_value = mock_screen

        result = self.capture.capture()

        # Largest match wins (1400x1000 > 1200x900)
        self.assertEqual(result.window_rect, Rect(x=30, y=40, width=1400, height=1000))

    @patch('src.capture.window_capture.Quartz')
    @patch('src.capture.window_capture.AppKit')
    def test_subprocess_failure_raises(self, mock_appkit, mock_quartz):
        """If capture fails, a RuntimeError is raised."""
        mock_quartz.CGWindowListCopyWindowInfo.return_value = [
            self._make_mock_window('WeChat', 100, 200, 1200, 900),
        ]
        mock_quartz.kCGWindowListOptionOnScreenOnly = 1
        mock_quartz.kCGWindowListExcludeDesktopElements = 2
        mock_quartz.kCGNullWindowID = 0
        mock_quartz.kCGWindowOwnerName = 'kCGWindowOwnerName'
        mock_quartz.kCGWindowBounds = 'kCGWindowBounds'
        mock_quartz.kCGWindowNumber = 'kCGWindowNumber'

        mock_screen = MagicMock()
        mock_screen.backingScaleFactor.return_value = 1.0
        mock_appkit.NSScreen.mainScreen.return_value = mock_screen

        self.automation.capture_success = False
        self.automation.capture_error = "mock capture failure"

        with self.assertRaises(RuntimeError):
            self.capture.capture()


if __name__ == '__main__':
    unittest.main()
