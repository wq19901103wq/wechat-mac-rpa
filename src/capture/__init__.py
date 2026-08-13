"""截图模块 - 负责屏幕截图"""
from .window_capture import CaptureResult, WeChatNotReadyError, WindowCapture, WindowNotFoundError

__all__ = ['WindowCapture', 'CaptureResult', 'WindowNotFoundError', 'WeChatNotReadyError']
