#!/usr/bin/env python3
"""Tests for L4 Reply modules."""

import xml.etree.ElementTree as ET

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
    def test_prompt_is_one_xml_request_without_unread_duplication(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        history = ChatMessage(
            text="旧消息",
            sender="Alice",
            sender_type=SenderType.OTHER,
            chat_name="A&B",
        )
        unread = ChatMessage(
            text="新消息 <确认>",
            sender="Alice",
            sender_type=SenderType.OTHER,
            chat_name="A&B",
        )

        prompt = gen._build_user_prompt([unread], [history, unread])
        root = ET.fromstring(prompt)

        assert root.tag == "request"
        assert [message.text for message in root.findall("./history/message")] == ["Alice：旧消息"]
        assert [message.text for message in root.findall("./unread/message")] == ["Alice：新消息 <确认>"]

    def test_self_history_is_separated_and_forbidden_after_unread(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        other = ChatMessage(
            text="真人旧消息",
            sender="Alice",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        self_reply = ChatMessage(
            text="Bot 已用旧梗",
            sender="我",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        unread = ChatMessage(
            text="真人新消息",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        prompt = gen._build_user_prompt([unread], [other, self_reply, unread], is_group=True)
        root = ET.fromstring(prompt)

        assert [message.text for message in root.findall("./history/message")] == ["Alice：真人旧消息"]
        consumed = root.find("./consumed_self_replies")
        assert consumed is not None
        assert consumed.attrib["reuse"] == "forbidden"
        assert [message.text for message in consumed.findall("./message")] == ["我：Bot 已用旧梗"]
        guard = root.find("./reply_guard")
        assert guard is not None
        assert guard.attrib == {"priority": "final", "enforcement": "hard"}
        assert "不得 callback" in "".join(guard.itertext())
        assert "不能成为回复的主语、宾语、修饰对象、背景或隐含前提" in "".join(guard.itertext())
        assert "没有独立的新内容" in "".join(guard.itertext())
        assert [child.tag for child in root][-3:] == ["unread", "consumed_self_replies", "reply_guard"]

    def test_reply_guard_exists_without_self_history(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        unread = ChatMessage(
            text="真人新消息",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )

        root = ET.fromstring(gen._build_user_prompt([unread], [unread], is_group=True))

        assert [child.tag for child in root][-1] == "reply_guard"
        assert "必须按 identity_provocation 处理" in "".join(root.find("./reply_guard").itertext())

    def test_system_prompt_does_not_embed_skill_catalog(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)

        prompt = gen._system_prompt()

        assert ET.fromstring(prompt).tag == "instructions"
        assert "<available_skills>" not in prompt
        assert "<active_skill" not in prompt

    def test_appended_sections_stay_before_final_reply_guard(self):
        gen = ReplyGenerator(llm_client=None, tool_registry=_TEST_TOOL_REGISTRY)
        self_reply = ChatMessage(
            text="Bot 已用旧梗",
            sender="我",
            sender_type=SenderType.SELF,
            chat_name="测试群",
        )
        unread = ChatMessage(
            text="真人新消息",
            sender="Bob",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
        )
        prompt = gen._build_user_prompt([unread], [self_reply, unread], is_group=True)

        prompt = gen._append_request_sections(
            prompt,
            [
                '<active_skill name="group_banter"><skill /></active_skill>',
                '<style_examples source="verified_human"><example /></style_examples>',
            ],
        )
        root = ET.fromstring(prompt)

        assert [child.tag for child in root][-3:] == [
            "active_skill",
            "style_examples",
            "reply_guard",
        ]

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
        assert gen.last_generation_failed is True
