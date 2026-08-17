#!/usr/bin/env python3
"""Tests for L2 WindowCapture module."""

import os
import sys
import unittest
from unittest.mock import MagicMock, call, patch

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

    def test_validate_skips_when_pytesseract_import_fails(self):
        original_import = __import__

        def fail_pytesseract(name, *args, **kwargs):
            if name == "pytesseract":
                raise ImportError("broken optional dependency")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fail_pytesseract):
            self.assertTrue(self.capture._validate_wechat_screenshot("unused.png"))

    def test_window_capture_failure_falls_back_to_region(self):
        rect = Rect(x=100, y=200, width=1200, height=900)
        self.automation.capture_screen = MagicMock(
            side_effect=[(False, "window capture failed"), (True, "")]
        )

        with patch("src.capture.window_capture.time.sleep") as mock_sleep:
            self.capture._do_capture(rect, window_id=47)

        self.automation.capture_screen.assert_has_calls([
            call(rect, self.capture.output_path, window_id=47),
            call(rect, self.capture.output_path, window_id=None),
        ])
        self.assertIn(
            ("activate_app", ("WeChat",), {}),
            self.automation.calls,
        )
        mock_sleep.assert_called_once_with(0.5)

    @patch.object(WindowCapture, '_validate_wechat_screenshot', return_value=True)
    @patch.object(WindowCapture, '_get_scale_factor', return_value=1.0)
    @patch.object(WindowCapture, '_find_window', return_value=(Rect(x=100, y=200, width=1200, height=900), 1))
    def test_capture_success_wechat_en(self, mock_find, mock_scale, mock_validate):
        """Successful capture of WeChat window（平台无关）。"""
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
    @patch.object(WindowCapture, '_get_scale_factor', return_value=2.0)
    @patch.object(WindowCapture, '_find_window', return_value=(Rect(x=50, y=100, width=1760, height=1280), 1))
    def test_capture_success_wechat_cn(self, mock_find, mock_scale, mock_validate):
        """Successful capture of WeChat window（平台无关）。"""
        result = self.capture.capture()

        self.assertIsInstance(result, CaptureResult)
        self.assertEqual(result.window_rect, Rect(x=50, y=100, width=1760, height=1280))
        self.assertEqual(result.scale_factor, 2.0)

    @patch.object(WindowCapture, '_find_window', return_value=None)
    def test_window_not_found_no_wechat(self, mock_find):
        """Raise WindowNotFoundError when no WeChat window exists."""
        with self.assertRaises(WindowNotFoundError):
            self.capture.capture()

    @patch.object(WindowCapture, '_find_window', return_value=None)
    def test_window_not_found_too_small(self, mock_find):
        """Raise WindowNotFoundError when WeChat window is too small."""
        with self.assertRaises(WindowNotFoundError):
            self.capture.capture()

    @patch.object(WindowCapture, '_validate_wechat_screenshot', return_value=True)
    @patch.object(WindowCapture, '_get_scale_factor', return_value=1.0)
    @patch.object(WindowCapture, '_find_window', return_value=(Rect(x=30, y=40, width=1400, height=1000), 2))
    def test_capture_uses_largest_matching_window(self, mock_find, mock_scale, mock_validate):
        """When multiple WeChat windows exist, the largest one is selected."""
        result = self.capture.capture()

        # Largest match wins (1400x1000 > 1200x900)
        self.assertEqual(result.window_rect, Rect(x=30, y=40, width=1400, height=1000))

    @patch.object(WindowCapture, '_find_window', return_value=(Rect(x=100, y=200, width=1200, height=900), 1))
    def test_subprocess_failure_raises(self, mock_find):
        """If capture fails, a RuntimeError is raised."""
        self.automation.capture_success = False
        self.automation.capture_error = "mock capture failure"

        with self.assertRaises(RuntimeError):
            self.capture.capture()


if __name__ == '__main__':
    unittest.main()
