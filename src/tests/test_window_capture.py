#!/usr/bin/env python3
"""Tests for L2 WindowCapture module."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.capture.window_capture import CaptureResult, WindowCapture, WindowNotFoundError
from src.models.base import Rect


class TestWindowCapture(unittest.TestCase):
    """Test WindowCapture with mocked Quartz and subprocess."""

    def setUp(self):
        self.capture = WindowCapture()

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
    @patch('src.capture.window_capture.subprocess.run')
    @patch('src.capture.window_capture.Quartz')
    @patch('src.capture.window_capture.AppKit')
    def test_capture_success_wechat_en(self, mock_appkit, mock_quartz, mock_subprocess, mock_validate):
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

        mock_screen = MagicMock()
        mock_screen.backingScaleFactor.return_value = 1.0
        mock_appkit.NSScreen.mainScreen.return_value = mock_screen

        mock_subprocess.return_value = MagicMock(returncode=0)

        result = self.capture.capture()

        self.assertIsInstance(result, CaptureResult)
        self.assertEqual(result.image_path, self.capture.output_path)
        self.assertEqual(result.window_rect, Rect(x=100, y=200, width=1200, height=900))
        self.assertEqual(result.scale_factor, 1.0)

        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        cmd = call_args[0][0]
        self.assertEqual(cmd[0], 'screencapture')
        self.assertEqual(cmd[1], '-R')
        self.assertEqual(cmd[2], '100,200,1200,900')
        self.assertEqual(cmd[3], '-x')
        self.assertEqual(cmd[4], self.capture.output_path)

    @patch.object(WindowCapture, '_validate_wechat_screenshot', return_value=True)
    @patch('src.capture.window_capture.subprocess.run')
    @patch('src.capture.window_capture.Quartz')
    @patch('src.capture.window_capture.AppKit')
    def test_capture_success_wechat_cn(self, mock_appkit, mock_quartz, mock_subprocess, mock_validate):
        """Successful capture of Chinese-named WeChat window."""
        mock_quartz.CGWindowListCopyWindowInfo.return_value = [
            self._make_mock_window('微信', 50, 100, 1760, 1280),
        ]
        mock_quartz.kCGWindowListOptionOnScreenOnly = 1
        mock_quartz.kCGWindowListExcludeDesktopElements = 2
        mock_quartz.kCGNullWindowID = 0
        mock_quartz.kCGWindowOwnerName = 'kCGWindowOwnerName'
        mock_quartz.kCGWindowBounds = 'kCGWindowBounds'

        mock_screen = MagicMock()
        mock_screen.backingScaleFactor.return_value = 2.0
        mock_appkit.NSScreen.mainScreen.return_value = mock_screen

        mock_subprocess.return_value = MagicMock(returncode=0)

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

        with self.assertRaises(WindowNotFoundError):
            self.capture.capture()

    @patch.object(WindowCapture, '_validate_wechat_screenshot', return_value=True)
    @patch('src.capture.window_capture.subprocess.run')
    @patch('src.capture.window_capture.Quartz')
    @patch('src.capture.window_capture.AppKit')
    def test_capture_uses_largest_matching_window(self, mock_appkit, mock_quartz, mock_subprocess, mock_validate):
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

        mock_screen = MagicMock()
        mock_screen.backingScaleFactor.return_value = 1.0
        mock_appkit.NSScreen.mainScreen.return_value = mock_screen

        mock_subprocess.return_value = MagicMock(returncode=0)

        result = self.capture.capture()

        # Largest match wins (1400x1000 > 1200x900)
        self.assertEqual(result.window_rect, Rect(x=30, y=40, width=1400, height=1000))

    @patch('src.capture.window_capture.subprocess.run')
    @patch('src.capture.window_capture.Quartz')
    @patch('src.capture.window_capture.AppKit')
    def test_subprocess_failure_raises(self, mock_appkit, mock_quartz, mock_subprocess):
        """If screencapture fails, the subprocess exception propagates."""
        mock_quartz.CGWindowListCopyWindowInfo.return_value = [
            self._make_mock_window('WeChat', 100, 200, 1200, 900),
        ]
        mock_quartz.kCGWindowListOptionOnScreenOnly = 1
        mock_quartz.kCGWindowListExcludeDesktopElements = 2
        mock_quartz.kCGNullWindowID = 0
        mock_quartz.kCGWindowOwnerName = 'kCGWindowOwnerName'
        mock_quartz.kCGWindowBounds = 'kCGWindowBounds'

        mock_screen = MagicMock()
        mock_screen.backingScaleFactor.return_value = 1.0
        mock_appkit.NSScreen.mainScreen.return_value = mock_screen

        import subprocess as sp
        mock_subprocess.side_effect = sp.CalledProcessError(1, ['screencapture'])

        with self.assertRaises(sp.CalledProcessError):
            self.capture.capture()


if __name__ == '__main__':
    unittest.main()
