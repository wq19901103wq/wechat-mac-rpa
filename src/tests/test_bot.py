#!/usr/bin/env python3
"""L5 Bot Orchestrator 单元测试"""

from unittest.mock import Mock

import pytest

from src.bot.wechat_bot import WeChatBot
from src.layout.profile import PROFILE_WECHAT_MAC_1760X1280
from src.models.base import ActionResult, ChatMessage, PerceptionResult, SenderType


class TestWeChatBot:
    @pytest.fixture
    def bot(self):
        return WeChatBot(PROFILE_WECHAT_MAC_1760X1280)

    def test_tick_no_new_messages(self, bot, tmp_path):
        """没有新消息时不发送回复"""
        mock_result = PerceptionResult(
            chat_name="测试群",
            messages=[ChatMessage(text="旧消息", sender="A", sender_type=SenderType.OTHER, chat_name="测试群")],
            chat_list_items=[],
            screenshot_path=str(tmp_path / "1.png")
        )
        bot.perception = Mock()
        bot.perception.perceive.return_value = mock_result

        # mock global_store 返回空未回复列表
        bot.global_store = Mock()
        bot.global_store.merge_tick.return_value = (Mock(), [])

        bot.tick()

        bot.perception.perceive.assert_called_once()
        bot.global_store.merge_tick.assert_called_once()
        bot.global_store.mark_replied.assert_not_called()

    def test_tick_replies_to_new_message(self, bot, tmp_path):
        """有新消息且 policy 允许时发送回复"""
        msg = ChatMessage(text="在吗", sender="A", sender_type=SenderType.OTHER, chat_name="测试群")
        mock_result = PerceptionResult(
            chat_name="测试群",
            messages=[msg],
            chat_list_items=[],
            screenshot_path=str(tmp_path / "2.png")
        )
        bot.perception = Mock()
        bot.perception.perceive.return_value = mock_result

        bot.policy = Mock()
        bot.policy.should_reply.return_value = True

        bot.generator = Mock()
        bot.generator.generate.return_value = ["在的"]

        bot.sender = Mock()
        bot.sender.send.return_value = ActionResult(success=True, sent_text="在的")

        mock_state = Mock()
        mock_state.pending_self_messages = []
        bot.global_store = Mock()
        bot.global_store.merge_tick.return_value = (mock_state, [msg])
        bot.global_store.chats = {"测试群": mock_state}

        bot.tick()

        bot.generator.generate.assert_called_once()
        bot.sender.send.assert_called_once_with("在的", chat_name="测试群")
        bot.global_store.mark_replied.assert_called_once()

    def test_tick_perception_none(self, bot):
        """perceive 返回 None 时直接跳过"""
        bot.perception = Mock()
        bot.perception.perceive.return_value = None

        bot.tick()

        bot.perception.perceive.assert_called_once()

    def test_tick_policy_declines(self, bot, tmp_path):
        """policy 返回 False 时不生成回复"""
        msg = ChatMessage(text="在吗", sender="A", sender_type=SenderType.OTHER, chat_name="测试群")
        mock_result = PerceptionResult(
            chat_name="测试群",
            messages=[msg],
            chat_list_items=[],
            screenshot_path=str(tmp_path / "3.png")
        )
        bot.perception = Mock()
        bot.perception.perceive.return_value = mock_result

        bot.policy = Mock()
        bot.policy.should_reply.return_value = False

        bot.generator = Mock()
        bot.sender = Mock()

        mock_state = Mock()
        bot.global_store = Mock()
        bot.global_store.merge_tick.return_value = (mock_state, [msg])

        bot.tick()

        bot.generator.generate.assert_not_called()
        bot.sender.send.assert_not_called()

    def test_on_message_callback(self, bot, tmp_path):
        """on_message 回调被正确触发"""
        callback = Mock()
        bot.on_message = callback

        msg = ChatMessage(text="hi", sender="A", sender_type=SenderType.OTHER, chat_name="测试群")
        mock_result = PerceptionResult(
            chat_name="测试群",
            messages=[msg],
            chat_list_items=[],
            screenshot_path=str(tmp_path / "4.png")
        )
        bot.perception = Mock()
        bot.perception.perceive.return_value = mock_result

        mock_state = Mock()
        bot.global_store = Mock()
        bot.global_store.merge_tick.return_value = (mock_state, [msg])

        bot.tick()

        callback.assert_called_once_with(msg, mock_state)

    def test_send_to_chat(self, bot):
        """send_to_chat 主动发送接口"""
        bot.sender = Mock()
        bot.sender.send.return_value = ActionResult(success=True, sent_text="hello")

        result = bot.send_to_chat("测试群", "hello")

        assert result.success is True
        bot.sender.send.assert_called_once_with("hello", chat_name="测试群")
