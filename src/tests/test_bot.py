#!/usr/bin/env python3
"""L5 Bot Orchestrator 单元测试"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.action.login_recovery import LoginRecoveryResult, LoginRecoveryStatus
from src.benchmarks.production_replay_archive import (
    DecisionPointCutoff,
    ProductionReplayArchive,
    ReplaySender,
    ReplaySideEffectSink,
)
from src.bot.wechat_bot import WeChatBot
from src.capture.window_capture import WeChatNotReadyError
from src.layout.profile import PROFILE_WECHAT_MAC_1760X1280
from src.models.base import ActionResult, ChatMessage, PerceptionResult, SenderType
from src.reply.digital_human import DigitalHumanError, DigitalHumanReplyGenerator
from src.session.global_store import ChatState


class TestWeChatBot:
    @pytest.fixture(autouse=True)
    def digital_human_client(self, monkeypatch):
        monkeypatch.setattr('src.bot.wechat_bot.load_digital_human_config',
                            lambda: dict(repository=Path('/example/dh'), instance='test', timeout=10))
        factory = Mock()
        factory.return_value.review.return_value = dict(decision='allow', issues=[])
        monkeypatch.setattr('src.reply.digital_human.DigitalHumanDraftClient', factory)
        return factory

    @pytest.fixture
    def bot(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("WECHAT_GROUP_REPLY_ENABLED", raising=False)
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

    def test_generator_uses_digital_human_even_with_injected_llm(self, bot, digital_human_client):
        assert isinstance(bot.generator, DigitalHumanReplyGenerator)
        assert bot.generator.llm_client is None
        assert bot.generator._client is digital_human_client.return_value
        digital_human_client.assert_called_once()

    def test_default_bot_uses_digital_human(self, bot):
        default_bot = WeChatBot(PROFILE_WECHAT_MAC_1760X1280, perception=Mock())
        assert isinstance(default_bot.generator, DigitalHumanReplyGenerator)
        assert default_bot.generator.llm_client is None

    @pytest.mark.parametrize('setting,is_group,should_generate', [
        (None, True, False), ('0', True, False), ('1', True, True), ('0', False, True),
    ])
    def test_group_reply_switch(self, bot, monkeypatch, setting, is_group, should_generate):
        if setting is not None:
            monkeypatch.setenv('WECHAT_GROUP_REPLY_ENABLED', setting)
        bot = WeChatBot(PROFILE_WECHAT_MAC_1760X1280, perception=Mock())
        target = ChatMessage('在吗', 'A', SenderType.OTHER, 'example', is_at_me=True)
        state = ChatState('source-test', 'example', is_group, [target])
        bot.perception.perceive.return_value = PerceptionResult('example', [target], [], '', is_group=is_group)
        bot.global_store.merge_tick.return_value = (state, [target])
        bot.global_store.chats = {'example': state}
        bot.policy.should_reply = Mock(return_value=True)
        bot.generator = Mock()
        bot.generator.generate.return_value = ['回复']
        bot.sender = Mock(silent_mode=False)
        bot.sender.send.return_value = ActionResult(success=True, sent_text='回复')
        bot.on_message = Mock()
        bot.debug_logger = Mock(current=None)
        bot._try_switch_to_unread_chat = Mock(return_value='')

        bot.tick()

        bot.global_store.merge_tick.assert_called_once_with('example', [target], is_group=is_group)
        if should_generate:
            bot.generator.generate.assert_called_once_with([target], [target], is_group=is_group,
                                                          tick_id=bot._tick_id)
            bot.sender.send.assert_called_once_with('回复', chat_name='example')
        else:
            bot.generator.generate.assert_not_called()
            bot.sender.send.assert_not_called()
            bot.policy.should_reply.assert_not_called()
            bot.on_message.assert_not_called()
            bot.global_store.mark_replied.assert_not_called()
            bot._try_switch_to_unread_chat.assert_called_once()

    @pytest.mark.parametrize('is_group', [False, True])
    @pytest.mark.parametrize('outcome', ['draft', 'silence', 'failure', 'missing_time', 'blocked', 'review_failure'])
    def test_tick_routes_through_digital_human(self, bot, is_group, outcome):
        bot.group_reply_enabled = True
        target = ChatMessage('在吗', 'A', SenderType.OTHER, 'example',
                             chatroom_id='source-test', create_time=None if outcome == 'missing_time' else 100)
        state = ChatState('source-test', 'example', is_group, [target])
        bot.perception.perceive.return_value = PerceptionResult('example', [target], [], '', is_group=is_group)
        bot.global_store.merge_tick.return_value = (state, [target])
        bot.global_store.chats = {'example': state}
        bot.policy.should_reply = Mock(return_value=True)
        bot.sender = Mock(silent_mode=False)
        bot.sender.send.return_value = ActionResult(success=True, sent_text='DH 回复')
        client = Mock()
        client.draft.return_value = dict(replies=[] if outcome == 'silence' else ['DH 回复'],
                                        binding=dict(instance='test', generator='g-test', data='d-test'))
        client.review.return_value = dict(decision='block' if outcome == 'blocked' else 'allow',
            issues=[dict(category='hallucination', reason='无事实依据', quote='DH 回复', reply_index=0)]
                   if outcome == 'blocked' else [])
        if outcome == 'failure':
            client.draft.side_effect = DigitalHumanError('DH unavailable')
        if outcome == 'review_failure':
            client.review.side_effect = TimeoutError('review unavailable')
        bot.generator._client = client

        bot.tick()

        if outcome == 'missing_time':
            client.draft.assert_not_called()
        else:
            assert client.draft.call_args.args[0]['chat_type'] == ('group' if is_group else 'private')
        if outcome == 'draft':
            bot.sender.send.assert_called_once_with('DH 回复', chat_name='example')
            bot.global_store.mark_replied.assert_called_once_with('example', target, 'DH 回复')
        else:
            bot.sender.send.assert_not_called()
            if outcome in ('silence', 'blocked'):
                bot.global_store.mark_replied.assert_called_once_with('example', target, '(未回复)')
            else:
                bot.global_store.mark_replied.assert_not_called()
                assert bot.generator.last_generation_failed is True
                if outcome == 'review_failure':
                    client.close.assert_not_called()
                    assert bot.generator.last_generation_trace[-1]['type'] == 'reply_review_error'
                else:
                    client.close.assert_called_once()
        if outcome in ('silence', 'failure', 'missing_time'):
            client.review.assert_not_called()
        else:
            client.review.assert_called_once()

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
        bot.logger = Mock()

        bot.tick()

        bot.perception.perceive.assert_called_once()
        warning = bot.logger.warning.call_args.args[0]
        assert "实际启动 Bot 的应用" in warning
        assert "屏幕录制权限" in warning
        assert "自动化开发环境" in warning

    def test_tick_recovers_when_perception_reports_not_ready(self, bot, tmp_path):
        result = PerceptionResult(
            chat_name="测试群",
            messages=[],
            chat_list_items=[],
            screenshot_path=str(tmp_path / "recovered.png"),
        )
        bot.perception = Mock()
        bot.perception.perceive.side_effect = [WeChatNotReadyError("需要登录"), result]
        bot._login_handler = Mock()
        bot._login_handler.handle.return_value = LoginRecoveryResult(
            LoginRecoveryStatus.SUCCESS, "已恢复"
        )
        bot.global_store = Mock()
        bot.global_store.merge_tick.return_value = (Mock(), [])

        bot.tick()

        assert bot.perception.perceive.call_count == 2
        bot._login_handler.handle.assert_called_once()

    def test_chat_switch_can_be_disabled(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("scripts.sync_knowledge.sync", lambda: False)
        monkeypatch.setattr("src.bot.wechat_bot.GlobalStore", Mock)
        monkeypatch.setattr("src.bot.wechat_bot.MemoryEngine", Mock)
        disabled_bot = WeChatBot(
            PROFILE_WECHAT_MAC_1760X1280,
            llm_client=Mock(),
            perception=Mock(),
            enable_chat_switch=False,
        )

        assert disabled_bot.enable_chat_switch is False
        assert disabled_bot._try_switch_to_unread_chat(Mock()) == ""

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

    def test_replay_tick_archives_then_stops_before_post_decision_side_effects(
        self, bot, tmp_path
    ):
        bot.group_reply_enabled = True
        msg = ChatMessage("在吗", "A", SenderType.OTHER, "测试群")
        state = ChatState("group-1", "测试群", True, [msg])
        pending_before = list(state.pending_self_messages)
        bot.perception.perceive.return_value = PerceptionResult(
            chat_name="测试群",
            messages=[msg],
            chat_list_items=[],
            screenshot_path="",
            is_group=True,
        )
        bot.policy = Mock()
        bot.policy.should_reply.return_value = True
        bot.generator = SimpleNamespace(
            generate=Mock(return_value=["在的"]),
            text_for_logging=lambda text: text,
            tool_registry=SimpleNamespace(to_openai_schemas=lambda: []),
            session_memory=SimpleNamespace(_sessions={}),
            last_system_prompt="system",
            last_user_prompt="user",
            last_raw_response="在的",
            last_llm_calls=[],
            last_tool_calls=[],
            last_generation_trace=[],
            last_loaded_skills=[],
            last_skill_injected_content="",
            last_active_llm="fake",
        )
        original_sender = Mock()
        bot.sender = ReplaySender(
            ReplaySideEffectSink(tmp_path / "blocked.jsonl"),
            original_sender,
        )
        bot.global_store = Mock(max_messages=1000)
        bot.global_store.chats = {"测试群": state}
        bot.global_store.merge_tick.return_value = (state, [msg])
        bot.replay_archive = ProductionReplayArchive(
            tmp_path / "archive", bot.session_id, {"mode": "test"}
        )
        bot._replay_cutoff_reached = False
        bot.debug_logger = Mock(current=SimpleNamespace(reply_generation_trace=[]))
        bot._update_tick_send_result = Mock()
        bot.memory_engine = Mock()
        bot.on_message = Mock()

        with pytest.raises(DecisionPointCutoff):
            bot.tick()

        original_sender.send.assert_not_called()
        bot.global_store.mark_replied.assert_not_called()
        bot.global_store.save.assert_not_called()
        bot.global_store.save_screenshot.assert_not_called()
        bot._update_tick_send_result.assert_not_called()
        bot.memory_engine.update_group_wiki.assert_not_called()
        bot.memory_engine.update_user_wiki.assert_not_called()
        bot.on_message.assert_not_called()
        assert state.pending_self_messages == pending_before
        tick_dir = tmp_path / "archive" / bot.session_id / "ticks" / str(bot._tick_id)
        meta = json.loads((tick_dir / "tick.json").read_text(encoding="utf-8"))
        assert meta["decision_point_captured"] is True

    def test_replay_run_auto_stops_after_first_cutoff(self, monkeypatch):
        bot = WeChatBot.__new__(WeChatBot)
        bot._tick_id = 1
        bot.logger = Mock()
        bot.tick = Mock(side_effect=DecisionPointCutoff("stop"))
        monkeypatch.setattr("src.bot.wechat_bot.time.sleep", Mock())

        bot.run_auto(interval=0)

        bot.tick.assert_called_once()
        assert bot.running is False

    def test_replay_save_sessions_is_always_suppressed(self, bot, tmp_path):
        bot.replay_archive = ProductionReplayArchive(
            tmp_path / "archive", bot.session_id, {"mode": "test"}
        )

        bot.save_sessions()

        bot.global_store.save.assert_not_called()

    def test_replay_not_ready_stops_without_login_recovery(self, bot, tmp_path):
        bot.perception.perceive.side_effect = WeChatNotReadyError("需要登录")
        bot._login_handler = Mock()
        bot.replay_archive = ProductionReplayArchive(
            tmp_path / "archive", bot.session_id, {"mode": "test"}
        )

        with pytest.raises(DecisionPointCutoff):
            bot.tick()

        bot._login_handler.handle.assert_not_called()
        bot.global_store.save.assert_not_called()

    def test_replay_archive_failure_stops_before_send(self, bot, tmp_path):
        msg = ChatMessage("在吗", "A", SenderType.OTHER, "测试群")
        state = ChatState("group-1", "测试群", True, [msg])
        bot.perception.perceive.return_value = PerceptionResult(
            chat_name="测试群",
            messages=[msg],
            chat_list_items=[],
            screenshot_path="",
            is_group=True,
        )
        bot.global_store = Mock(max_messages=1000, chats={"测试群": state})
        bot.global_store.merge_tick.return_value = (state, [msg])
        bot.policy = Mock()
        bot.policy.should_reply.return_value = True
        bot.generator = Mock()
        bot.sender = Mock()
        bot.replay_archive = Mock()
        bot.replay_archive.capture_perception.side_effect = OSError("disk full")

        with pytest.raises(DecisionPointCutoff):
            bot.tick()

        bot.generator.generate.assert_not_called()
        bot.sender.send.assert_not_called()
        bot.global_store.save.assert_not_called()

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
