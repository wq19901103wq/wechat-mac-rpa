#!/usr/bin/env python3
"""L2 Capture - 窗口截图模块"""

import glob
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.action.system_automation import SystemAutomation, get_system_automation
from src.models.base import Rect

_logger = logging.getLogger("src.window_capture")


class WindowNotFoundError(Exception):
    """未找到目标窗口时抛出"""
    pass


class WeChatNotReadyError(Exception):
    """微信窗口尺寸异常（未登录/需扫码）时抛出"""
    pass


@dataclass
class CaptureResult:
    """窗口截图结果"""
    image_path: str
    window_rect: Rect
    scale_factor: float  # Retina 屏幕为 2.0，普通屏幕为 1.0


class CaptureValidationError(Exception):
    """截图内容验证失败（截到的不是微信窗口）时抛出"""
    pass


class WindowCapture:
    """查找并截图微信主窗口"""

    def __init__(
        self,
        output_path: Optional[str] = None,
        min_effective_width: int = 800,
        min_effective_height: int = 600,
        automation: SystemAutomation | None = None,
    ):
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            pid = os.getpid()
            output_path = os.path.join(
                tempfile.gettempdir(), f"wechat_capture_{ts}_{pid}.png"
            )
        self.output_path = output_path
        self.app_names = ['WeChat', '微信']
        self.min_width = 200
        self.min_height = 200
        self.min_effective_width = min_effective_width
        self.min_effective_height = min_effective_height
        self.automation = automation or get_system_automation()

    def _find_window(self) -> Optional[tuple]:
        """查找微信主窗口，返回面积最大的有效窗口 (Rect, window_id) 或 None"""
        import sys

        if sys.platform == "win32":
            return self._find_window_windows()
        # macOS：使用 Quartz
        import Quartz

        # 先尝试 OnScreenOnly（正常情况）
        result = self._find_window_with_options(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
        )
        # fallback：如果 OnScreenOnly 找不到（如窗口在另一个 Space / 外接显示器），
        # 则不加 OnScreenOnly 限制，只排除桌面元素
        if result is None:
            result = self._find_window_with_options(
                Quartz.kCGWindowListExcludeDesktopElements
            )
        return result

    def _find_window_windows(self) -> Optional[tuple]:
        """Windows：通过 automation 查找微信主窗口（含隐藏窗口）。"""
        import win32gui

        hwnd = self.automation.find_main_window("WeChat")
        if not hwnd:
            return None
        rect = win32gui.GetWindowRect(hwnd)
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        if width <= self.min_width or height <= self.min_height:
            return None
        return (
            Rect(x=rect[0], y=rect[1], width=width, height=height),
            hwnd,
        )

    def _find_window_with_options(self, options: int) -> Optional[tuple]:
        """使用指定 options 查找微信窗口，返回 (Rect, window_id) 或 None"""
        import Quartz

        window_list = Quartz.CGWindowListCopyWindowInfo(
            options, Quartz.kCGNullWindowID
        )

        best_window: Optional[Rect] = None
        best_window_id: Optional[int] = None
        best_area = 0

        for window in window_list:
            owner = window.get(Quartz.kCGWindowOwnerName, '')
            if owner in self.app_names:
                bounds = window.get(Quartz.kCGWindowBounds, {})
                width = int(bounds.get('Width', 0))
                height = int(bounds.get('Height', 0))
                window_id = int(window.get(Quartz.kCGWindowNumber, 0))

                if width > self.min_width and height > self.min_height:
                    area = width * height
                    if area > best_area:
                        best_area = area
                        best_window = Rect(
                            x=int(bounds.get('X', 0)),
                            y=int(bounds.get('Y', 0)),
                            width=width,
                            height=height
                        )
                        best_window_id = window_id
        return (best_window, best_window_id) if best_window else None

    def _is_effective_window(self, rect: Rect) -> bool:
        """判断窗口尺寸是否达到有效主窗口标准"""
        return (
            rect.width >= self.min_effective_width
            and rect.height >= self.min_effective_height
        )

    def _activate_wechat(self) -> None:
        """尝试激活微信应用"""
        self.automation.activate_app("WeChat")

    def _get_scale_factor(self) -> float:
        """获取主屏幕的 Retina 缩放因子"""
        import sys

        if sys.platform == "win32":
            # Windows 侧坐标已按物理像素（DPI aware），无需额外缩放
            return 1.0
        import AppKit

        try:
            screen = AppKit.NSScreen.mainScreen()
            if screen is not None:
                return float(screen.backingScaleFactor())
        except Exception as e:
            _logger.warning(f"获取屏幕缩放因子失败: {e}")
        return 1.0

    def _to_screencapture_region(self, rect: Rect) -> str:
        """将 Rect 转换为 screencapture -R 参数格式"""
        return f"{rect.x},{rect.y},{rect.width},{rect.height}"

    def _do_capture(self, rect: Rect, window_id: int) -> None:
        """执行截图命令。

        优先使用 -l <windowid> 只截取指定窗口（不受其他窗口覆盖影响），
        fallback 到 -R 按坐标截取。
        """
        ok, err = self.automation.capture_screen(
            rect, self.output_path, window_id=window_id if window_id else None
        )
        if ok:
            return

        _logger.warning("[WindowCapture] 按窗口截图失败，降级为区域截图: %s", err)
        self._activate_wechat()
        time.sleep(0.5)
        fallback_ok, fallback_err = self.automation.capture_screen(
            rect, self.output_path, window_id=None
        )
        if not fallback_ok:
            raise RuntimeError(f"截图失败: window={err}; region={fallback_err}")

    def _validate_wechat_screenshot(self, image_path: str) -> bool:
        """验证截图内容确实是微信窗口。

        方法：OCR 截图顶部区域，检查是否有微信特有的 UI 元素
        （如左侧边栏的"搜索"、聊天列表、或标题栏文字）。
        这是一种轻量级的布局验证，不依赖具体聊天内容。

        注：Tesseract 未安装时跳过验证（graceful degrade），
        不阻断主流程，因为主 OCR 使用 qwen-vl-ocr / Vision 框架。
        """
        try:
            import pytesseract
        except ImportError as e:
            _logger.debug("pytesseract 不可用，跳过截图内容验证: %s", e)
            return True

        try:
            from PIL import Image
            img = Image.open(image_path)
            # 截取顶部 80px 区域（标题栏 + 搜索框位置）
            top_region = img.crop((0, 0, min(img.width, 400), min(img.height, 80)))
            text = pytesseract.image_to_string(top_region, lang='chi_sim+eng').strip()
            # 微信窗口顶部通常有"搜索"或当前聊天名
            wechat_indicators = {'搜索', '微信', 'WeChat'}
            if any(ind in text for ind in wechat_indicators):
                return True
            # fallback：检查左侧边栏区域是否有微信图标特征
            left_region = img.crop((0, 0, min(img.width, 60), min(img.height, 200)))
            left_text = pytesseract.image_to_string(left_region, lang='chi_sim+eng').strip()
            return bool(left_text)  # 左侧有文字/图标说明是微信
        except pytesseract.TesseractNotFoundError:
            _logger.debug("Tesseract 未安装，跳过截图内容验证")
            return True
        except Exception as e:
            _logger.warning(f"截图验证异常: {e}")
            return False

    def capture(self) -> CaptureResult:
        """
        查找并截图微信主窗口。

        如果窗口尺寸过小（未登录/浮窗），会先尝试激活微信并等待后重试一次。
        重试后仍无效则抛出 WeChatNotReadyError，提示用户可能需要扫码登录。

        Returns:
            CaptureResult: 包含图片路径和窗口几何信息

        Raises:
            WindowNotFoundError: 未找到任何微信窗口
            WeChatNotReadyError: 窗口尺寸异常，可能需要扫码登录
            CaptureValidationError: 截图内容验证失败
        """
        t_capture_start = time.time()
        # 清理旧截图（超过1小时的临时文件，避免 /tmp 无限累积）
        try:
            cutoff = time.time() - 3600
            for old in glob.glob(os.path.join(tempfile.gettempdir(), "wechat_capture_*.png")):
                try:
                    if os.path.getmtime(old) < cutoff:
                        os.remove(old)
                except OSError as e:
                    _logger.warning("cleanup old screenshot failed: %s", e)
        except Exception as e:
            _logger.debug("[WindowCapture] 清理旧截图失败: %s", e)

        # 每次调用生成新的输出路径，避免覆盖旧截图
        # 这是 SmartPerceptionPipeline 像素 diff 正确工作的前提
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        pid = os.getpid()
        self.output_path = os.path.join(
            tempfile.gettempdir(), f"wechat_capture_{ts}_{pid}.png"
        )

        t_find_start = time.time()
        result = self._find_window()
        t_find_ms = (time.time() - t_find_start) * 1000
        if result is None:
            raise WindowNotFoundError("WeChat window not found")

        window_rect, window_id = result

        if not self._is_effective_window(window_rect):
            # 尝试激活微信并等待恢复
            self._activate_wechat()
            time.sleep(2.0)
            result = self._find_window()

            if result is None:
                raise WindowNotFoundError("WeChat window not found after activation")

            window_rect, window_id = result

            if not self._is_effective_window(window_rect):
                raise WeChatNotReadyError(
                    f"微信窗口尺寸异常 ({window_rect.width}x{window_rect.height})，"
                    "可能需要扫码登录或主窗口未展开"
                )

        t_screenshot_start = time.time()
        self._do_capture(window_rect, window_id)
        t_screenshot_ms = (time.time() - t_screenshot_start) * 1000

        # 验证截图内容
        t_validate_start = time.time()
        is_valid = self._validate_wechat_screenshot(self.output_path)
        t_validate_ms = (time.time() - t_validate_start) * 1000
        if not is_valid:
            raise CaptureValidationError(
                "截图验证失败：截到的内容不像微信窗口，可能有其他窗口覆盖"
            )

        scale_factor = self._get_scale_factor()
        t_total_ms = (time.time() - t_capture_start) * 1000
        _logger.info(
            f"[Perf][Capture] total={t_total_ms:.0f}ms "
            f"find_window={t_find_ms:.0f}ms screenshot={t_screenshot_ms:.0f}ms "
            f"validate={t_validate_ms:.0f}ms"
        )

        return CaptureResult(
            image_path=self.output_path,
            window_rect=window_rect,
            scale_factor=scale_factor
        )
