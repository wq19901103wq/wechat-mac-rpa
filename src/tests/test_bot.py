#!/usr/bin/env python3
"""L5 Bot Orchestrator 单元测试"""

from unittest.mock import Mock

import pytest

from src.bot.wechat_bot import WeChatBot
from src.layout.profile import PROFILE_WECHAT_MAC_1760X1280
from src.models.base import ActionResult, ChatMessage, PerceptionResult, SenderType


class TestWeChatBot:
    @pytest.fixture
    def bot(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("scripts.sync_knowledge.sync", lambda: False)
        monkeypatch.setattr("src.bot.wechat_bot.GlobalStore", Mock)
        monkeypatch.setattr("src.bot.wechat_bot.MemoryEngine", Mock)
        db = Mock()
        db._get_conn.return_value = Mock()
        monkeypatch.setattr("src.badcase.case_db.get_db", lambda: db)
        bot = WeChatBot(PROFILE_WECHAT_MAC_1760X1280, llm_client=Mock(), perception=Mock())
        bot.global_store.chats = {}
        bot.memory_engine.get_user_memory.return_value = ""
        bot.debug_logger = Mock(current=None)
        return bot

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
        bot.policy = Mock()
        bot.policy.should_reply.return_value = False

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

        bot.policy = Mock()
        bot.policy.should_reply.return_value = False
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

    def test_run_auto_keeps_fixed_interval(self, monkeypatch):
        """tick 耗时应从轮询间隔中扣除，避免周期持续漂移。"""
        bot = WeChatBot.__new__(WeChatBot)
        bot._tick_id = 1
        bot.logger = Mock()
        bot.tick = Mock(side_effect=lambda: setattr(bot, "running", False))
        monotonic = Mock(side_effect=[10.0, 12.0])
        sleep = Mock()
        monkeypatch.setattr("src.bot.wechat_bot.time.monotonic", monotonic)
        monkeypatch.setattr("src.bot.wechat_bot.time.sleep", sleep)

        bot.run_auto(interval=5.0)

        sleep.assert_called_once_with(3.0)

    def test_no_reply_chat_skips_llm_and_marks_processed(self, bot, tmp_path):
        msg = ChatMessage(text="推送", sender="公众号", sender_type=SenderType.OTHER, chat_name="公众号")
        bot.perception = Mock()
        bot.perception.perceive.return_value = PerceptionResult(
            chat_name="公众号", messages=[msg], chat_list_items=[], screenshot_path=str(tmp_path / "skip.png")
        )
        bot.policy = Mock()
        bot.policy.should_reply.return_value = True
        bot.generator = Mock()
        bot.sender = Mock(silent_mode=False)
        state = Mock(messages=[msg])
        bot.global_store = Mock()
        bot.global_store.merge_tick.return_value = (state, [msg])

        bot.tick()

        bot.generator.generate.assert_not_called()
        bot.global_store.mark_replied.assert_called_once_with("公众号", msg, "")

    def test_partial_send_failure_does_not_retry_delivered_reply(self, bot, tmp_path, monkeypatch):
        bot._tick_id = 1
        msg = ChatMessage(text="分两条回", sender="A", sender_type=SenderType.OTHER, chat_name="测试群")
        bot.perception = Mock()
        bot.perception.perceive.return_value = PerceptionResult(
            chat_name="测试群", messages=[msg], chat_list_items=[], screenshot_path=str(tmp_path / "partial.png")
        )
        bot.policy = Mock()
        bot.policy.should_reply.return_value = True
        bot.generator = Mock()
        bot.generator.generate.return_value = ["第一条", "第二条"]
        bot.generator.text_for_logging.side_effect = lambda text: text
        bot.generator.messages_for_logging.side_effect = lambda messages: messages
        bot.sender = Mock(silent_mode=False)
        bot.sender.send.side_effect = [
            ActionResult(success=True, sent_text="第一条"),
            ActionResult(success=False, error="发送失败"),
        ]
        state = Mock(messages=[msg], pending_self_messages=[])
        bot.global_store = Mock()
        bot.global_store.merge_tick.return_value = (state, [msg])
        bot.global_store.chats = {"测试群": state}
        bot._update_tick_send_result = Mock()
        monkeypatch.setattr("src.bot.wechat_bot.time.sleep", Mock())

        bot.tick()

        bot.global_store.mark_replied.assert_called_once_with("测试群", msg, "第一条")
        bot._update_tick_send_result.assert_called_once_with(2, ["第一条"], success=False)

    def test_generation_failure_keeps_message_unreplied(self, bot, tmp_path):
        msg = ChatMessage(
            text="在吗",
            sender="A",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        bot.perception = Mock()
        bot.perception.perceive.return_value = PerceptionResult(
            chat_name="测试群",
            messages=[msg],
            chat_list_items=[],
            screenshot_path=str(tmp_path / "failed.png"),
        )
        bot.policy = Mock()
        bot.policy.should_reply.return_value = True
        bot.generator = Mock(last_generation_failed=True)
        bot.generator.generate.return_value = []
        bot.generator.text_for_logging.side_effect = lambda text: text
        bot.generator.messages_for_logging.side_effect = lambda messages: messages
        bot.sender = Mock(silent_mode=False)
        state = Mock(messages=[msg])
        bot.global_store = Mock()
        bot.global_store.merge_tick.return_value = (state, [msg])
        bot.global_store.chats = {"测试群": state}

        bot.tick()

        bot.global_store.mark_replied.assert_not_called()

    def test_explicit_no_reply_marks_message_replied(self, bot, tmp_path):
        msg = ChatMessage(
            text="不用回",
            sender="A",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        bot.perception = Mock()
        bot.perception.perceive.return_value = PerceptionResult(
            chat_name="测试群",
            messages=[msg],
            chat_list_items=[],
            screenshot_path=str(tmp_path / "no_reply.png"),
        )
        bot.policy = Mock()
        bot.policy.should_reply.return_value = True
        bot.generator = Mock(last_generation_failed=False)
        bot.generator.generate.return_value = []
        bot.generator.text_for_logging.side_effect = lambda text: text
        bot.generator.messages_for_logging.side_effect = lambda messages: messages
        bot.sender = Mock(silent_mode=False)
        state = Mock(messages=[msg])
        bot.global_store = Mock()
        bot.global_store.merge_tick.return_value = (state, [msg])
        bot.global_store.chats = {"测试群": state}

        bot.tick()

        bot.global_store.mark_replied.assert_called_once_with(
            "测试群", msg, "(未回复)"
        )
