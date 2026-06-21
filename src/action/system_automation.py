#!/usr/bin/env python3
"""L4 Action - 系统级 UI 自动化抽象

将 cliclick、osascript、AppleScript 等 macOS 特定调用封装为统一接口，
使 Bot 层和 Action 层可以面向接口编程，便于 mock 和跨平台扩展。
"""

import subprocess
from abc import ABC, abstractmethod


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
    def click_at(self, x: int, y: int) -> bool:
        """在屏幕逻辑坐标 (x, y) 处点击一次。"""
        pass

    @abstractmethod
    def send_keys(self, key_spec: str) -> bool:
        """发送键盘事件。

        Args:
            key_spec: 按键描述，如 "keystroke \"v\" using command down"、
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

    @abstractmethod
    def get_clipboard_text(self) -> tuple[bool, str]:
        """读取系统剪贴板文本。"""
        pass


class MacOSSystemAutomation(SystemAutomation):
    """macOS 实现：基于 AppleScript + cliclick。"""

    def __init__(self, cliclick_path: str = "/opt/homebrew/bin/cliclick"):
        self.cliclick_path = cliclick_path

    def activate_app(self, app_name: str) -> bool:
        script = f'tell application "{app_name}" to activate'
        try:
            rc, _, _ = self.run_applescript(script, timeout=3)
            return rc == 0
        except Exception:
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

    def click_at(self, x: int, y: int) -> bool:
        try:
            subprocess.run(
                [self.cliclick_path, f"c:{x},{y}"],
                check=True,
                timeout=5,
            )
            return True
        except Exception:
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
            rc, _, _ = self.run_applescript(script, timeout=5)
            return rc == 0
        except Exception:
            return False

    def run_applescript(self, script: str, timeout: int = 5) -> tuple[int, str, str]:
        try:
            r = subprocess.run(
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
            subprocess.run(
                ["pbcopy"],
                input=text.encode("utf-8"),
                timeout=2,
                capture_output=True,
            )
            return True
        except Exception:
            return False

    def get_clipboard_text(self) -> tuple[bool, str]:
        try:
            r = subprocess.run(
                ["pbpaste"],
                timeout=2,
                capture_output=True,
            )
            if r.returncode == 0:
                return True, r.stdout.decode("utf-8", errors="replace")
            return False, r.stderr.decode("utf-8", errors="replace")
        except Exception as e:
            return False, str(e)


class NoOpSystemAutomation(SystemAutomation):
    """空实现，用于测试或禁用 UI 交互的场景。"""

    def activate_app(self, app_name: str) -> bool:
        return True

    def get_frontmost_app(self, app_name: str) -> tuple[bool, str]:
        return True, app_name

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
