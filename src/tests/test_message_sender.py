#!/usr/bin/env python3
"""MessageSender paste 验证单元测试

通过 Mock SystemAutomation 验证 WeChatMessageSender 的发送逻辑，
不再直接依赖 subprocess.run。
"""

import pytest

from src.action.message_sender import WeChatMessageSender
from src.action.system_automation import SystemAutomation
from src.models.base import Rect


class MockSystemAutomation(SystemAutomation):
    """可编程的 SystemAutomation mock，用于测试发送流程。"""

    def __init__(
        self,
        frontmost: bool = True,
        pasted_texts: list[str] | None = None,
        original_clipboard: str = "original_clipboard",
    ):
        self.frontmost = frontmost
        self.pasted_texts = list(pasted_texts or [])
        self.original_clipboard = original_clipboard
        self._pbpaste_call_index = 0
        self.calls: list[tuple[str, tuple, dict]] = []

    def _log(self, name: str, *args, **kwargs):
        self.calls.append((name, args, kwargs))

    def activate_app(self, app_name: str) -> bool:
        self._log("activate_app", app_name)
        return True

    def get_frontmost_app(self, app_name: str) -> tuple[bool, str]:
        self._log("get_frontmost_app", app_name)
        return self.frontmost, "WeChat" if self.frontmost else "Other"

    def get_window_rect(self, app_name: str) -> tuple[bool, Rect | None, str]:
        self._log("get_window_rect", app_name)
        return True, Rect(x=100, y=100, width=800, height=600), ""

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
        # 第一次调用：读取原始剪贴板；后续调用：读取验证时的输入框内容
        if self._pbpaste_call_index == 0:
            self._pbpaste_call_index += 1
            return True, self.original_clipboard
        idx = self._pbpaste_call_index - 1
        self._pbpaste_call_index += 1
        if idx < len(self.pasted_texts):
            return True, self.pasted_texts[idx]
        return True, ""

    def capture_screen(
        self,
        rect: Rect,
        output_path: str,
        window_id: int | None = None,
    ) -> tuple[bool, str]:
        self._log("capture_screen", rect, output_path, window_id=window_id)
        return True, ""


@pytest.fixture
def sender():
    return WeChatMessageSender()


class TestPasteVerification:
    """修复方向：paste 后验证输入框内容，不匹配则重试"""

    def test_paste_verification_success(self):
        """paste 验证通过，直接发送，无需重试"""
        automation = MockSystemAutomation(pasted_texts=["测试消息"])
        sender = WeChatMessageSender(automation=automation)

        result = sender.send("测试消息")

        assert result.success is True
        assert automation.get_frontmost_app("WeChat")[0] is True
        # 至少应包含 focus、paste、verify、return 相关的 automation 调用
        # （语义化方法 paste/copy_selection 在 mock 上走 send_keys）
        assert any(c[0] == "send_keys" for c in automation.calls)
        assert any(c[0] == "get_clipboard_text" for c in automation.calls)

    def test_paste_verification_fail_then_retry(self):
        """paste 验证失败，重试后成功"""
        automation = MockSystemAutomation(pasted_texts=["", "测试消息"])
        sender = WeChatMessageSender(automation=automation)

        result = sender.send("测试消息")

        assert result.success is True
        # 原始剪贴板 + 验证1 + 验证2 = 至少 3 次 get_clipboard_text
        pbpaste_count = sum(1 for c in automation.calls if c[0] == "get_clipboard_text")
        assert pbpaste_count >= 3

    def test_paste_verification_fail_after_max_retries(self):
        """paste 验证多次失败，最终返回失败"""
        automation = MockSystemAutomation(pasted_texts=["", "", "", ""])
        sender = WeChatMessageSender(automation=automation)

        result = sender.send("测试消息")

        assert result.success is False
        assert "paste" in result.error.lower() or "验证" in result.error.lower() or "发送" in result.error.lower()

    def test_paste_verification_rejects_existing_draft(self):
        """输入框中混有旧草稿时不得发送。"""
        automation = MockSystemAutomation(
            pasted_texts=["前缀测试消息后缀", "", "", "", ""]
        )
        sender = WeChatMessageSender(automation=automation)

        result = sender.send("测试消息")

        assert result.success is False

    def test_silent_mode(self):
        """静默模式下不调用任何 UI 自动化，返回 skipped 标志避免误标 replied"""
        automation = MockSystemAutomation()
        sender = WeChatMessageSender(silent_mode=True, automation=automation)

        result = sender.send("测试消息")

        assert result.success is False
        assert result.skipped is True
        assert result.sent_text == "测试消息"
        assert "静默模式跳过发送" in result.error
        # 静默模式不应触发任何 UI 调用
        assert len(automation.calls) == 0

    def test_silent_whitelist_requires_exact_normalized_name(self, monkeypatch):
        monkeypatch.setenv("SILENT_WHITELIST", "柚子群")
        sender = WeChatMessageSender(silent_mode=True, automation=MockSystemAutomation())

        assert sender.in_silent_whitelist("柚子群 (12)") is True
        assert sender.in_silent_whitelist("旧柚子群备份") is False
