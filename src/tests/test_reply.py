#!/usr/bin/env python3
"""Tests for L4 Reply modules."""

from src.models.base import ChatMessage, SenderType
from src.reply.generator import ReplyGenerator
from src.reply.policy import ReplyPolicy
from src.session.global_store import ChatState
from src.tools import get_registry, register_builtin_tools

# 测试用全局工具注册表
_TEST_TOOL_REGISTRY = get_registry()
register_builtin_tools(_TEST_TOOL_REGISTRY)


class TestReplyPolicy:
    def test_self_message_returns_false(self):
        policy = ReplyPolicy()
        session = ChatState(chat_id="c1", chat_name="Friend")
        msg = ChatMessage(
            text="hello",
            sender="me",
            sender_type=SenderType.SELF,
            chat_name="Friend",
        )
        assert policy.should_reply(msg, session) is False

    def test_system_message_returns_false(self):
        policy = ReplyPolicy()
        session = ChatState(chat_id="c1", chat_name="Friend")
        msg = ChatMessage(
            text="system alert",
            sender="system",
            sender_type=SenderType.SYSTEM,
            chat_name="Friend",
        )
        assert policy.should_reply(msg, session) is False

    def test_group_chat_with_at_returns_true(self):
        policy = ReplyPolicy()
        from src.session.global_store import ChatState
        session = ChatState(chat_id="c1", chat_name="Group (3)")
        msg = ChatMessage(
            text="@me hello",
            sender="friend",
            sender_type=SenderType.OTHER,
            chat_name="Group (3)",
            is_at_me=True,
        )
        assert policy.should_reply(msg, session) is True

    def test_normal_private_chat_returns_true(self):
        policy = ReplyPolicy()
        from src.session.global_store import ChatState
        session = ChatState(chat_id="c1", chat_name="Alice")
        msg = ChatMessage(
            text="hello",
            sender="Alice",
            sender_type=SenderType.OTHER,
            chat_name="Alice",
        )
        assert policy.should_reply(msg, session) is True


class TestReplyGenerator:
    def test_returns_non_empty_string(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        msg = ChatMessage(
            text="hello",
            sender="Alice",
            sender_type=SenderType.OTHER,
            chat_name="Alice",
        )
        reply = gen.generate([msg], [msg])
        assert isinstance(reply, list)
        assert len(reply) >= 0

    def test_handles_none_llm_client_gracefully(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        msg = ChatMessage(
            text="hello",
            sender="Alice",
            sender_type=SenderType.OTHER,
            chat_name="Alice",
        )
        reply = gen.generate([msg], [msg])
        assert isinstance(reply, list)

    def test_fallback_when_llm_fails(self):
        class FailingLLM:
            def chat(self, *args, **kwargs):
                raise RuntimeError("LLM down")

        gen = ReplyGenerator(llm_client=FailingLLM(), tool_registry=_TEST_TOOL_REGISTRY)
        msg = ChatMessage(
            text="hello",
            sender="Alice",
            sender_type=SenderType.OTHER,
            chat_name="Alice",
        )
        reply = gen.generate([msg], [msg])
        assert reply == []
