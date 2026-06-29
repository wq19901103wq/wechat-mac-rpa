#!/usr/bin/env python3
"""L4 Action Layer 单元测试"""


from unittest.mock import MagicMock, patch

import pytest

from src.action.message_sender import MessageSender, WeChatMessageSender
from src.action.ui_interactor import PyAutoGUIInteractor, UIInteractor
from src.models.base import ActionResult, ChatListItem, Rect


class TestMessageSenderInterface:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            MessageSender()


class TestWeChatMessageSender:
    def test_send_invokes_pbcopy_and_subprocess(self):
        """发送成功时会调用 pbcopy 与发送子流程"""
        sender = WeChatMessageSender()
        text = "你好，世界"

        with patch.object(sender, "_ensure_wechat_frontmost", return_value=(True, "")), \
             patch.object(sender, "_focus_input", return_value=(0, "")), \
             patch.object(sender, "_pbcopy", return_value=(0, "")), \
             patch.object(sender, "_paste", return_value=(0, "")), \
             patch.object(sender, "_verify", return_value=(text, 0, 0)), \
             patch.object(sender, "_send_return", return_value=(0, "")):
            result = sender.send(text)

        assert result.success is True
        assert result.sent_text == text

    def test_send_includes_wechat_frontmost_check(self):
        """发送时会检查 WeChat 是否为 frontmost"""
        sender = WeChatMessageSender()

        with patch.object(sender, "_ensure_wechat_frontmost", return_value=(True, "")) as mock_frontmost, \
             patch.object(sender, "_focus_input", return_value=(0, "")), \
             patch.object(sender, "_pbcopy", return_value=(0, "")), \
             patch.object(sender, "_paste", return_value=(0, "")), \
             patch.object(sender, "_verify", return_value=("test", 0, 0)), \
             patch.object(sender, "_send_return", return_value=(0, "")):
            sender.send("test")

        mock_frontmost.assert_called()

    def test_send_returns_success_action_result(self):
        sender = WeChatMessageSender()
        text = "hello"

        with patch.object(sender, "_ensure_wechat_frontmost", return_value=(True, "")), \
             patch.object(sender, "_focus_input", return_value=(0, "")), \
             patch.object(sender, "_pbcopy", return_value=(0, "")), \
             patch.object(sender, "_paste", return_value=(0, "")), \
             patch.object(sender, "_verify", return_value=(text, 0, 0)), \
             patch.object(sender, "_send_return", return_value=(0, "")):
            result = sender.send(text)

        assert isinstance(result, ActionResult)
        assert result.success is True
        assert result.sent_text == text
        assert result.error is None

    def test_send_returns_failure_when_frontmost_check_fails(self):
        sender = WeChatMessageSender()

        with patch.object(sender, "_ensure_wechat_frontmost", return_value=(False, "无法激活微信")):
            result = sender.send("hello")

        assert isinstance(result, ActionResult)
        assert result.success is False
        assert result.error is not None
        assert "无法激活微信" in result.error

    def test_send_image_not_implemented(self, tmp_path):
        sender = WeChatMessageSender()
        result = sender.send_image(str(tmp_path / "test.png"))
        assert isinstance(result, ActionResult)
        assert result.success is False
        assert "not implemented" in result.error.lower()

    def test_send_file_file_not_exists(self, tmp_path):
        sender = WeChatMessageSender()
        result = sender.send_file(str(tmp_path / "nonexistent_file_12345.txt"))
        assert isinstance(result, ActionResult)
        assert result.success is False
        assert "不存在" in result.error

    def test_send_file_silent_mode_returns_failure(self, tmp_path):
        # 静默模式跳过发送应返回 success=False（未真实发送），
        # 避免调用方误 mark_replied 导致对方消息被永久跳过
        sender = WeChatMessageSender(silent_mode=True)
        result = sender.send_file(str(tmp_path / "nonexistent_file_12345.txt"))
        assert isinstance(result, ActionResult)
        assert result.success is False
        assert "[文件]" in result.sent_text

    def test_send_text_silent_mode_returns_failure(self):
        # 文本静默跳过同样返回 success=False（核心修复：王勇奇消息不回复的根因）
        sender = WeChatMessageSender(silent_mode=True)
        result = sender.send("测试消息", chat_name="不在白名单的聊天")
        assert isinstance(result, ActionResult)
        assert result.success is False
        assert result.sent_text == "测试消息"

    def test_send_file_invokes_copy_and_paste_scripts(self):
        """文件发送走 frontmost→focus→复制到剪贴板→paste→return 流程"""
        import tempfile

        sender = WeChatMessageSender()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            tmp_path = f.name

        try:
            with patch.object(sender, "_ensure_wechat_frontmost", return_value=(True, "")) as mock_frontmost, \
                 patch.object(sender, "_focus_input", return_value=(0, "")) as mock_focus, \
                 patch.object(sender, "_clear_input"), \
                 patch.object(sender, "automation") as mock_auto, \
                 patch.object(sender, "_paste", return_value=(0, "")) as mock_paste, \
                 patch.object(sender, "_send_return", return_value=(0, "")) as mock_return:
                # automation.run_applescript 用于"复制文件到剪贴板"，返回成功
                mock_auto.run_applescript.return_value = (0, "", "")
                result = sender.send_file(tmp_path)

            assert result.success is True
            assert "[文件]" in result.sent_text
            # 验证流程各步被调用
            mock_frontmost.assert_called_once()
            mock_focus.assert_called_once()
            mock_paste.assert_called_once()
            mock_return.assert_called_once()
            # 复制文件到剪贴板的 applescript 至少调一次
            mock_auto.run_applescript.assert_called()
        finally:
            import os
            os.unlink(tmp_path)


class TestUIInteractorInterface:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            UIInteractor()


class TestPyAutoGUIInteractor:
    def test_click_chat_item_calculates_correct_center_from_rect(self):
        interactor = PyAutoGUIInteractor()
        item = ChatListItem(
            nickname="小王",
            last_message_preview="在吗",
            unread_count="1",
            timestamp="12:34",
            rect=Rect(x=10, y=20, width=100, height=60),
        )

        with patch("src.action.ui_interactor.pyautogui") as mock_pyautogui:
            mock_pyautogui.click.return_value = None
            result = interactor.click_chat_item(item)

        assert result is True
        # center x = 10 + 100/2 = 60, center y = 20 + 60/2 = 50
        mock_pyautogui.click.assert_called_once_with(60, 50)

    def test_click_input_box_returns_bool(self):
        interactor = PyAutoGUIInteractor()

        with patch("src.action.ui_interactor.pyautogui") as mock_pyautogui:
            mock_pyautogui.click.return_value = None
            result = interactor.click_input_box()

        assert isinstance(result, bool)
        assert result is True
        mock_pyautogui.click.assert_called_once()

    def test_click_chat_item_returns_false_on_exception(self):
        interactor = PyAutoGUIInteractor()
        item = ChatListItem(
            nickname="小王",
            last_message_preview="在吗",
            unread_count="1",
            timestamp="12:34",
            rect=Rect(x=10, y=20, width=100, height=60),
        )

        with patch("src.action.ui_interactor.pyautogui") as mock_pyautogui:
            mock_pyautogui.click.side_effect = Exception("click failed")
            result = interactor.click_chat_item(item)

        assert result is False
