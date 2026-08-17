#!/usr/bin/env python3
"""L4 Action - 系统级 UI 自动化抽象

将 cliclick、osascript、AppleScript、screencapture、pbcopy/pbpaste 等 macOS 特定调用
封装为统一接口，使 Bot 层和 Action 层可以面向接口编程，便于 mock、测试和跨平台扩展。
"""

import logging
import subprocess  # nosec B404
from abc import ABC, abstractmethod

from src.models.base import Rect

_logger = logging.getLogger("src.system_automation")


class SystemAutomation(ABC):
    """系统级 UI 自动化抽象接口。"""

    @abstractmethod
    def activate_app(self, app_name: str) -> bool:
        """激活指定应用并等待其获得焦点。"""
        pass

    @abstractmethod
    def get_frontmost_app(self, app_name: str) -> tuple[bool, str]:
        """检查指定应用是否为当前 frontmost 应用。

        Returns:
            (is_frontmost, info_or_error)
        """
        pass

    @abstractmethod
    def get_window_rect(self, app_name: str) -> tuple[bool, Rect | None, str]:
        """获取指定应用主窗口的位置和大小。

        Returns:
            (success, rect_or_none, error_message)
        """
        pass

    def find_main_window(self, app_name: str) -> int | None:
        """返回平台窗口标识（Windows: hwnd；macOS: CGWindowID）。

        找不到返回 None；不支持时返回 None。macOS 的 WindowCapture 使用
        Quartz 直接查找，因此无需覆盖。
        """
        return None

    @abstractmethod
    def click_at(self, x: int, y: int) -> bool:
        """在屏幕逻辑坐标 (x, y) 处点击一次。"""
        pass

    @abstractmethod
    def send_keys(self, key_spec: str) -> bool:
        """发送键盘事件。

        Args:
            key_spec: 按键描述，如 "keystroke \\"v\\" using command down"、
                      "key code 53" 等 AppleScript 片段。
        """
        pass

    @abstractmethod
    def run_applescript(self, script: str, timeout: int = 5) -> tuple[int, str, str]:
        """执行 AppleScript。

        Returns:
            (returncode, stdout, stderr)
        """
        pass

    @abstractmethod
    def set_clipboard_text(self, text: str) -> bool:
        """将文本写入系统剪贴板。"""
        pass

    # ------------------------------------------------------------------
    # 平台无关的语义化按键操作
    #
    # 基类默认实现基于 send_keys 的 AppleScript 片段语义（macOS 原生），
    # WindowsSystemAutomation 通过解析同一片段在 Windows 上等价执行；
    # 平台实现可按需 override。现有 mock 无需改动即可继承默认行为。
    # ------------------------------------------------------------------

    def paste(self) -> bool:
        """粘贴剪贴板内容（macOS: Cmd+V；Windows: Ctrl+V）。"""
        return self.send_keys('keystroke "v" using command down')

    def press_enter(self) -> bool:
        """按回车发送消息（macOS 先右箭头取消全选再回车，见 Mac 实现）。"""
        return self.send_keys("keystroke return")

    def clear_input(self) -> bool:
        """清空输入框（全选 + 删除）。"""
        if not self.send_keys('keystroke "a" using command down'):
            return False
        return self.send_keys("key code 51")

    def copy_selection(self) -> bool:
        """全选当前输入框并复制（用于发送前校验输入框内容）。"""
        if not self.send_keys('keystroke "a" using command down'):
            return False
        return self.send_keys('keystroke "c" using command down')

    def set_clipboard_file(self, path: str) -> bool:
        """把文件放入剪贴板（macOS: POSIX file；Windows: CF_HDROP）。

        Returns:
            True 成功；不支持时返回 False。
        """
        return False

    @abstractmethod
    def get_clipboard_text(self) -> tuple[bool, str]:
        """读取系统剪贴板文本。"""
        pass

    @abstractmethod
    def capture_screen(
        self,
        rect: Rect,
        output_path: str,
        window_id: int | None = None,
    ) -> tuple[bool, str]:
        """截取指定屏幕区域或窗口并保存到文件。

        Args:
            rect: 截图区域（window_id 为空时使用）
            output_path: 输出文件路径
            window_id: 可选的窗口 ID，优先使用 screencapture -l

        Returns:
            (success, error_message)
        """
        pass


class MacOSSystemAutomation(SystemAutomation):
    """macOS 实现：基于 AppleScript + cliclick + screencapture。"""

    def __init__(self, cliclick_path: str = "/opt/homebrew/bin/cliclick"):
        self.cliclick_path = cliclick_path

    def activate_app(self, app_name: str) -> bool:
        script = f'tell application "{app_name}" to activate'
        try:
            rc, _, stderr = self.run_applescript(script, timeout=3)
            if rc != 0:
                _logger.warning("activate_app(%s) 失败: %s", app_name, stderr)
            return rc == 0
        except (subprocess.SubprocessError, OSError) as e:
            _logger.warning("activate_app(%s) 异常: %s", app_name, e)
            return False

    def get_frontmost_app(self, app_name: str) -> tuple[bool, str]:
        script = f'''
            tell application "System Events"
                tell process "{app_name}"
                    set frontmost to true
                    delay 0.3
                end tell
                set frontApp to name of first application process whose frontmost is true
                return frontApp
            end tell
        '''
        try:
            rc, stdout, stderr = self.run_applescript(script, timeout=5)
            if rc != 0:
                return False, stderr
            front_app = stdout.strip()
            return front_app == app_name, front_app
        except Exception as e:
            return False, str(e)

    def get_window_rect(self, app_name: str) -> tuple[bool, Rect | None, str]:
        """通过 AppleScript 获取应用主窗口的位置和大小。"""
        script = f'''
            tell application "System Events"
                tell process "{app_name}"
                    tell window 1
                        set winPos to position
                        set winSize to size
                        return ((item 1 of winPos) as text) & "," & ((item 2 of winPos) as text) & "," & ((item 1 of winSize) as text) & "," & ((item 2 of winSize) as text)
                    end tell
                end tell
            end tell
        '''
        try:
            rc, stdout, stderr = self.run_applescript(script, timeout=5)
            if rc != 0:
                return False, None, stderr
            parts = stdout.strip().split(",")
            if len(parts) != 4:
                return False, None, f"无法解析窗口坐标: {stdout!r}"
            x, y, w, h = map(int, map(float, parts))
            return True, Rect(x=x, y=y, width=w, height=h), ""
        except Exception as e:
            return False, None, str(e)

    def click_at(self, x: int, y: int) -> bool:
        try:
            subprocess.run(  # nosec
                [self.cliclick_path, f"c:{x},{y}"],
                check=True,
                timeout=5,
            )
            return True
        except (subprocess.SubprocessError, OSError) as e:
            _logger.warning("click_at(%d,%d) 失败: %s", x, y, e)
            return False

    def send_keys(self, key_spec: str) -> bool:
        script = f'''
            tell application "System Events"
                tell process "WeChat"
                    {key_spec}
                end tell
            end tell
        '''
        try:
            rc, _, stderr = self.run_applescript(script, timeout=5)
            if rc != 0:
                _logger.warning("send_keys 失败: %s", stderr)
            return rc == 0
        except (subprocess.SubprocessError, OSError) as e:
            _logger.warning("send_keys 异常: %s", e)
            return False

    def run_applescript(self, script: str, timeout: int = 5) -> tuple[int, str, str]:
        try:
            r = subprocess.run(  # nosec
                ["osascript", "-e", script],
                timeout=timeout,
                capture_output=True,
            )
            return (
                r.returncode,
                r.stdout.decode("utf-8", errors="replace"),
                r.stderr.decode("utf-8", errors="replace"),
            )
        except Exception as e:
            return -1, "", str(e)

    def set_clipboard_text(self, text: str) -> bool:
        try:
            subprocess.run(  # nosec
                ["pbcopy"],
                input=text.encode("utf-8"),
                timeout=2,
                capture_output=True,
            )
            return True
        except (subprocess.SubprocessError, OSError) as e:
            _logger.warning("set_clipboard_text 失败: %s", e)
            return False

    def get_clipboard_text(self) -> tuple[bool, str]:
        try:
            r = subprocess.run(  # nosec
                ["pbpaste"],
                timeout=2,
                capture_output=True,
            )
            if r.returncode == 0:
                return True, r.stdout.decode("utf-8", errors="replace")
            return False, r.stderr.decode("utf-8", errors="replace")
        except Exception as e:
            return False, str(e)

    def capture_screen(
        self,
        rect: Rect,
        output_path: str,
        window_id: int | None = None,
    ) -> tuple[bool, str]:
        """使用 screencapture 截取指定区域或窗口。

        对输出路径做基本校验，防止路径注入。
        """
        _SHELL_META = "&;|`$()"
        if not output_path or "/" not in output_path or any(c in output_path for c in _SHELL_META):
            return False, f"非法截图输出路径: {output_path}"
        if window_id:
            cmd = [
                "screencapture",
                "-l", str(window_id),
                "-o",  # 排除窗口阴影
                "-x", output_path,
            ]
        else:
            cmd = [
                "screencapture",
                "-R", f"{rect.x},{rect.y},{rect.width},{rect.height}",
                "-x", output_path,
            ]
        try:
            subprocess.run(cmd, check=True, timeout=5)  # nosec
            return True, ""
        except Exception as e:
            return False, str(e)


    def set_clipboard_file(self, path: str) -> bool:
        """通过 AppleScript 把文件放入剪贴板（POSIX file）。"""
        safe_path = str(path).replace('"', '\\"')
        script = f'''
            set the clipboard to (POSIX file "{safe_path}")
        '''
        rc, _, stderr = self.run_applescript(script, timeout=5)
        if rc != 0:
            _logger.warning("set_clipboard_file 失败: %s", stderr)
        return rc == 0

    def press_enter(self) -> bool:
        """macOS 回车发送：先右箭头取消选中文本，再按回车。"""
        if not self.send_keys("key code 124"):
            return False
        return self.send_keys("keystroke return")


class NoOpSystemAutomation(SystemAutomation):
    """空实现，用于测试或禁用 UI 交互的场景。"""

    def __init__(self, window_rect: Rect | None = None):
        self.window_rect = window_rect or Rect(x=0, y=0, width=1200, height=800)

    def activate_app(self, app_name: str) -> bool:
        return True

    def get_frontmost_app(self, app_name: str) -> tuple[bool, str]:
        return True, app_name

    def get_window_rect(self, app_name: str) -> tuple[bool, Rect | None, str]:
        return True, self.window_rect, ""

    def click_at(self, x: int, y: int) -> bool:
        return True

    def send_keys(self, key_spec: str) -> bool:
        return True

    def run_applescript(self, script: str, timeout: int = 5) -> tuple[int, str, str]:
        return 0, "", ""

    def set_clipboard_text(self, text: str) -> bool:
        return True

    def get_clipboard_text(self) -> tuple[bool, str]:
        return True, ""

    def capture_screen(
        self,
        rect: Rect,
        output_path: str,
        window_id: int | None = None,
    ) -> tuple[bool, str]:
        return True, ""


class WindowsSystemAutomation(SystemAutomation):
    """Windows 实现：win32gui + SendInput + win32clipboard + PrintWindow/mss。

    关键点（已在微信 4.1.12.55 实测）：
    - 微信主窗口可能被隐藏到托盘（EnumWindows 看不到），需枚举线程窗口查找；
    - 截图用 PrintWindow(PW_RENDERFULLCONTENT) 走 DWM 离屏渲染，不怕遮挡，
      但隐藏窗口需先 ShowWindow 显示后再截；
    - 剪贴板用原生 win32clipboard（CF_UNICODETEXT），不用 PowerShell 子进程。
    """

    _SHELL_META = "&;|`$()"

    def __init__(self, process_names: tuple[str, ...] = ("Weixin.exe",)):
        self.process_names = tuple(n.lower() for n in process_names)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_dpi_aware() -> None:
        """启用 Per-Monitor DPI aware，保证窗口/鼠标坐标一致。"""
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                import ctypes

                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:  # nosec B110 - DPI aware 是尽力而为，失败不阻塞
                pass

    def _find_main_window(self, app_name: str) -> int | None:
        """找到微信主窗口句柄（含隐藏窗口），按面积取最大。"""
        import psutil
        import win32gui
        import win32process

        self._ensure_dpi_aware()
        pids = {
            p.info["pid"]
            for p in psutil.process_iter(["pid", "name"])
            if p.info["name"] and p.info["name"].lower() in self.process_names
        }
        if not pids:
            return None

        best: int | None = None
        best_area = 0
        best_title_match = False

        def consider(hwnd: int) -> None:
            nonlocal best, best_area, best_title_match
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
            except Exception:
                return
            if pid not in pids:
                return
            rect = win32gui.GetWindowRect(hwnd)
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            if width < 200 or height < 200:
                return
            # 优先标题含微信/Weixin 的主窗口，排除托盘/辅助窗口
            title = win32gui.GetWindowText(hwnd)
            title_match = "微信" in title or "weixin" in title.lower()
            area = width * height
            if title_match and not best_title_match:
                best = hwnd
                best_area = area
                best_title_match = True
                return
            if title_match == best_title_match and area > best_area:
                best = hwnd
                best_area = area

        # 先枚举可见顶层窗口
        win32gui.EnumWindows(lambda hwnd, _param: consider(hwnd), None)
        # 微信可能隐藏到托盘（EnumWindows 看不到），回退枚举各进程线程窗口
        if best is None:
            for pid in pids:
                try:
                    threads = psutil.Process(pid).threads()
                except (psutil.Error, OSError):
                    continue
                for tid_info in threads:
                    acc: list[int] = []
                    win32gui.EnumThreadWindows(
                        tid_info.id, lambda hwnd, _param, acc=acc: acc.append(hwnd), None
                    )
                    for hwnd in acc:
                        consider(hwnd)
        return best

    @staticmethod
    def _parse_key_spec(key_spec: str) -> list[dict]:
        """解析 AppleScript 片段（keystroke/key code）为按键动作序列。

        支持 message_sender 使用的模式：
          keystroke "v" using command down   -> Ctrl+V
          keystroke "a" using command down   -> Ctrl+A
          keystroke "c" using command down   -> Ctrl+C
          keystroke "x" using command down   -> Ctrl+X
          keystroke return                   -> Enter
          key code 51                        -> Delete
          key code 124                       -> Right
        """
        import re

        actions: list[dict] = []
        pattern = re.compile(
            r'keystroke\s+"([^"]*)"(?:\s+using\s+command\s+down)?|keystroke\s+(\w+)|key\s+code\s+(\d+)'
        )
        for m in pattern.finditer(key_spec):
            if m.group(1) is not None:
                ch = m.group(1)
                using_command = "using command down" in key_spec[: m.end()]
                actions.append({"type": "char", "char": ch, "command": using_command})
            elif m.group(2) is not None:
                actions.append({"type": "named", "name": m.group(2)})
            elif m.group(3) is not None:
                actions.append({"type": "code", "code": int(m.group(3))})
        return actions

    @staticmethod
    def _send_vk(vk: int, with_ctrl: bool = False) -> bool:
        """通过 SendInput/keybd_event 发送一个虚拟键。"""
        import ctypes
        import time

        user32 = ctypes.windll.user32
        try:
            if with_ctrl:
                user32.keybd_event(0x11, 0, 0, 0)  # VK_CONTROL down
            user32.keybd_event(vk, 0, 0, 0)  # down
            user32.keybd_event(vk, 0, 2, 0)  # up
            if with_ctrl:
                user32.keybd_event(0x11, 0, 2, 0)  # VK_CONTROL up
            time.sleep(0.05)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # SystemAutomation 实现
    # ------------------------------------------------------------------
    def activate_app(self, app_name: str) -> bool:
        import time

        import win32con
        import win32gui

        hwnd = self._find_main_window(app_name)
        if not hwnd:
            _logger.warning("activate_app(%s) 未找到微信主窗口", app_name)
            return False
        if not win32gui.IsWindowVisible(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            time.sleep(0.3)
        try:
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.3)
            return True
        except Exception as e:
            _logger.warning("activate_app(%s) 异常: %s", app_name, e)
            return False

    def get_frontmost_app(self, app_name: str) -> tuple[bool, str]:
        import psutil
        import win32gui
        import win32process

        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return False, "no foreground window"
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            name = psutil.Process(pid).name().lower()
        except Exception as e:
            return False, f"pid? ({e})"
        return name in self.process_names, name

    def get_window_rect(self, app_name: str) -> tuple[bool, Rect | None, str]:
        import win32gui

        hwnd = self._find_main_window(app_name)
        if not hwnd:
            return False, None, "未找到微信主窗口"
        rect = win32gui.GetWindowRect(hwnd)
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        return True, Rect(x=rect[0], y=rect[1], width=width, height=height), ""

    def find_main_window(self, app_name: str) -> int | None:
        return self._find_main_window(app_name)

    def click_at(self, x: int, y: int) -> bool:
        try:
            import pyautogui

            pyautogui.click(x, y)
            return True
        except Exception as e:
            _logger.warning("click_at(%d,%d) 失败: %s", x, y, e)
            return False

    def send_keys(self, key_spec: str) -> bool:
        """解析 AppleScript 按键片段并在 Windows 上等价执行。"""
        actions = self._parse_key_spec(key_spec)
        if not actions:
            _logger.warning("send_keys 无法解析按键片段: %s", key_spec)
            return False
        for action in actions:
            ok = self._exec_action(action)
            if not ok:
                return False
        return True

    def _exec_action(self, action: dict) -> bool:
        if action["type"] == "char":
            ch = action["char"].lower()
            if len(ch) == 1 and ch.isalpha():
                return self._send_vk(ord(ch.upper()), with_ctrl=action.get("command", False))
            return self._send_vk(0x0D)  # 其他字符按回车兜底
        if action["type"] == "named":
            name = action["name"].lower()
            if name in ("return", "enter"):
                return self._send_vk(0x0D)  # VK_RETURN
            return False
        if action["type"] == "code":
            # macOS key code -> Windows VK
            mac_to_vk = {
                36: 0x0D,  # return -> Enter
                51: 0x2E,  # delete -> Delete
                124: 0x27,  # right arrow -> VK_RIGHT
                53: 0x1B,  # esc -> Escape
            }
            vk = mac_to_vk.get(action["code"])
            if vk is None:
                _logger.warning("不支持的 key code: %d", action["code"])
                return False
            return self._send_vk(vk)
        return False

    def run_applescript(self, script: str, timeout: int = 5) -> tuple[int, str, str]:
        return 1, "", "Windows 不支持 AppleScript"

    def set_clipboard_text(self, text: str) -> bool:
        try:
            import win32clipboard
            import win32con

            win32clipboard.OpenClipboard()
            try:
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardText(text, win32con.CF_UNICODETEXT)
            finally:
                win32clipboard.CloseClipboard()
            return True
        except Exception as e:
            _logger.warning("set_clipboard_text 失败: %s", e)
            return False

    def get_clipboard_text(self) -> tuple[bool, str]:
        try:
            import win32clipboard
            import win32con

            win32clipboard.OpenClipboard()
            try:
                data = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
            finally:
                win32clipboard.CloseClipboard()
            return True, data
        except Exception as e:
            return False, str(e)

    def capture_screen(
        self,
        rect: Rect,
        output_path: str,
        window_id: int | None = None,
    ) -> tuple[bool, str]:
        """截取指定窗口（PrintWindow，不怕遮挡）或屏幕区域（mss）。"""
        import os

        if not output_path or not os.path.isabs(output_path) or any(c in output_path for c in self._SHELL_META):
            return False, f"非法截图输出路径: {output_path}"
        if window_id:
            return self._capture_window(window_id, output_path)
        return self._capture_region(rect, output_path)

    def _capture_window(self, hwnd: int, output_path: str) -> tuple[bool, str]:
        """PrintWindow 截取窗口客户区（隐藏窗口先显示）。"""
        import time

        import win32gui
        import win32ui

        if not win32gui.IsWindowVisible(hwnd):
            win32gui.ShowWindow(hwnd, 5)  # SW_SHOW
            time.sleep(0.5)
        client = win32gui.GetClientRect(hwnd)
        width = client[2] - client[0]
        height = client[3] - client[1]
        if width <= 0 or height <= 0:
            return False, f"窗口客户区无效: {width}x{height}"
        try:
            import ctypes

            from PIL import Image

            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(bitmap)
            pw_client_only = 0x1
            pw_render_full_content = 0x2
            ctypes.windll.user32.PrintWindow(
                hwnd, save_dc.GetSafeHdc(), pw_client_only | pw_render_full_content
            )
            bmp_info = bitmap.GetInfo()
            bmp_str = bitmap.GetBitmapBits(True)
            img = Image.frombuffer(
                "RGB",
                (bmp_info["bmWidth"], bmp_info["bmHeight"]),
                bmp_str,
                "raw",
                "BGRX",
                0,
                1,
            )
            img.save(output_path)
            win32gui.DeleteObject(bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)
            return True, ""
        except Exception as e:
            _logger.warning("PrintWindow 截图失败: %s", e)
            return False, str(e)

    def _capture_region(self, rect: Rect, output_path: str) -> tuple[bool, str]:
        """用 mss 截取屏幕区域（仅窗口区域，不截全屏）。"""
        try:
            import mss

            with mss.mss() as sct:
                monitor = {
                    "left": rect.x,
                    "top": rect.y,
                    "width": rect.width,
                    "height": rect.height,
                }
                sct.shot(output=output_path, monitor=monitor)
            return True, ""
        except Exception as e:
            _logger.warning("mss 截图失败: %s", e)
            return False, str(e)

    def set_clipboard_file(self, path: str) -> bool:
        """Windows 文件剪贴板（CF_HDROP）：构造 DROPFILES + UTF-16 路径。"""
        import os
        import struct

        import win32clipboard
        import win32con

        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            _logger.warning("set_clipboard_file 文件不存在: %s", abs_path)
            return False
        wide = abs_path.encode("utf-16-le") + b"\x00\x00\x00\x00"
        # DROPFILES: DWORD pFiles(20), POINT pt(8), BOOL fNC(4), BOOL fWide(4)
        header = struct.pack("<iiiii", 20, 0, 0, 0, 1)
        data = header + wide
        try:
            win32clipboard.OpenClipboard()
            try:
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32con.CF_HDROP, data)
            finally:
                win32clipboard.CloseClipboard()
            return True
        except Exception as e:
            _logger.warning("set_clipboard_file 失败: %s", e)
            return False


def get_system_automation() -> SystemAutomation:
    """按当前平台返回默认 SystemAutomation 实现。"""
    import sys

    if sys.platform == "win32":
        return WindowsSystemAutomation()
    return MacOSSystemAutomation()
