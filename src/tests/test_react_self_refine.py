#!/usr/bin/env python3
"""ReAct + Self-Refine 单元测试。

仅验证 ReplyGenerator 内部的 ReAct 循环、
Self-Refine（Feedback + Iterate）开关及 max_tool_calls 降级逻辑，
不调用真实 LLM API。
"""

import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from src.models.base import ChatMessage, SenderType
from src.reply.generator import MAX_TOOL_CALLS, ReplyGenerator


@dataclass
class MockFunction:
    name: str
    arguments: str


@dataclass
class MockToolCall:
    id: str
    type: str
    function: MockFunction

    def __init__(self, name: str, arguments: str, id: str = "tc_1", type: str = "function"):
        self.id = id
        self.type = type
        self.function = MockFunction(name=name, arguments=arguments)


@dataclass
class MockResponse:
    content: str = ""
    tool_calls: List[MockToolCall] = field(default_factory=list)
    reasoning_content: str = ""


class MockLLM:
    """按顺序返回预设响应的 Mock LLM。

    支持两种用法：
    1. 传入 `responses` 列表，每次 chat() 按顺序弹出；
    2. 传入 `response_func(messages, tools, **kwargs)`，在列表耗尽后动态生成响应。
    """

    def __init__(
        self,
        responses: Optional[List[Any]] = None,
        response_func: Optional[Any] = None,
    ):
        self.responses = list(responses) if responses else []
        self.response_func = response_func
        self.calls: List[Dict[str, Any]] = []
        self.index = 0

    def chat(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})

        if self.index < len(self.responses):
            resp = self.responses[self.index]
            self.index += 1
        elif self.response_func is not None:
            resp = self.response_func(messages, tools=tools, **kwargs)
        else:
            resp = '{"replies": ["fallback"]}'

        if isinstance(resp, str):
            return MockResponse(content=resp)
        return resp


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def sample_message():
    return ChatMessage(
        text="在吗",
        sender="Alice",
        sender_type=SenderType.OTHER,
        chat_name="Alice",
    )


def _make_generator(mock_llm_instance, enable_self_refine: bool = True) -> ReplyGenerator:
    """构造已关闭环境变量影响的 ReplyGenerator。"""
    gen = ReplyGenerator(llm_client=mock_llm_instance)
    gen.enable_self_refine = enable_self_refine
    gen.enable_react_tools = True
    return gen


class TestSelfRefine:
    def test_self_refine_pass_skips_iterate(self, mock_llm, sample_message):
        mock_llm.responses = [
            '{"skills": []}',                       # skill router
            '{"replies": ["test reply"]}',          # ReAct 生成
            '{"decision": "pass", "issues": []}',   # Feedback
        ]
        gen = _make_generator(mock_llm, enable_self_refine=True)
        replies = gen.generate([sample_message], [sample_message])

        assert replies == ["test reply"]
        assert gen.last_self_refine_applied is True
        assert gen.last_feedback_decision == "pass"
        assert gen.last_iterate_count == 0
        feedback_kwargs = mock_llm.calls[2]["kwargs"]
        assert feedback_kwargs["max_retries"] == 0
        assert feedback_kwargs["max_tokens"] == 300
        assert feedback_kwargs["response_format"] == {"type": "json_object"}
        feedback_messages = mock_llm.calls[2]["messages"]
        assert feedback_messages[0]["role"] == "system"
        assert ET.fromstring(feedback_messages[0]["content"]).tag == "reply_audit"
        assert feedback_messages[2] == {"role": "assistant", "content": '{"replies": ["test reply"]}'}
        assert feedback_messages[3]["content"] == "现在执行回复质量审计，只输出规定的 JSON。"
        # 消息流应包含 system + user + generation assistant + feedback prompt + feedback assistant
        assert len(gen.last_llm_messages) == 5
        assert gen.last_llm_messages[0]["role"] == "system"
        assert gen.last_llm_messages[2]["role"] == "assistant"
        assert gen.last_llm_messages[3]["role"] == "user"
        assert gen.last_llm_messages[4]["role"] == "assistant"

    def test_self_refine_fail_triggers_iterate(self, mock_llm, sample_message):
        mock_llm.responses = [
            '{"skills": []}',                       # skill router
            '{"replies": ["bad reply"]}',           # ReAct 生成
            '{"decision": "fail", "issues": ["太正式"]}',  # Feedback
            '{"replies": ["好的吧"]}',              # Iterate
        ]
        gen = _make_generator(mock_llm, enable_self_refine=True)
        replies = gen.generate([sample_message], [sample_message])

        assert replies == ["好的吧"]
        assert gen.last_self_refine_applied is True
        assert gen.last_feedback_decision == "fail"
        assert gen.last_iterate_count == 1
        # 消息流应包含 system + user + generation assistant + feedback prompt + feedback assistant + iterate prompt + iterate assistant
        assert len(gen.last_llm_messages) == 7
        assert gen.last_llm_messages[0]["role"] == "system"
        assert gen.last_llm_messages[2]["role"] == "assistant"
        assert gen.last_llm_messages[4]["role"] == "assistant"
        assert gen.last_llm_messages[6]["role"] == "assistant"

    def test_self_refine_disabled(self, mock_llm, sample_message):
        mock_llm.responses = [
            '{"skills": []}',                       # skill router
            '{"replies": ["reply"]}',               # 单次推理
        ]
        gen = _make_generator(mock_llm, enable_self_refine=False)
        replies = gen.generate([sample_message], [sample_message])

        assert replies == ["reply"]
        assert gen.last_self_refine_applied is False
        assert gen.last_feedback_decision == ""
        assert gen.last_iterate_count == 0
        # 消息流应包含 system + user + generation assistant
        assert len(gen.last_llm_messages) == 3
        assert gen.last_llm_messages[2]["role"] == "assistant"
        # 关闭 Self-Refine 后不应调用 Feedback/Iterate，只应有 2 次 LLM 调用
        assert len(mock_llm.calls) == 2

    def test_self_refine_feedback_error_returns_error_decision(self, mock_llm, sample_message):
        """Feedback 返回空 text 时 decision 应为 error，而非 pass。"""
        mock_llm.responses = [
            '{"skills": []}',                       # skill router
            '{"replies": ["test reply"]}',          # ReAct 生成
            MockResponse(content=""),               # Feedback 返回空
        ]
        gen = _make_generator(mock_llm, enable_self_refine=True)
        replies = gen.generate([sample_message], [sample_message])

        assert replies == ["test reply"]            # 保留原回复，不丢数据
        assert gen.last_self_refine_applied is True
        assert gen.last_feedback_decision == "error"
        assert gen.last_iterate_count == 0

    @pytest.mark.parametrize(
        "feedback",
        [
            '{"decision": "pass"}',
            '{"decision": "unknown", "issues": []}',
            '{"decision": "pass", "issues": ["不应存在"]}',
            '{"decision": "fail", "issues": []}',
            '{"decision": "fail", "issues": "太正式"}',
        ],
    )
    def test_self_refine_rejects_invalid_feedback_schema(self, feedback, sample_message):
        mock_llm = MockLLM(responses=[
            '{"skills": []}',
            '{"replies": ["test reply"]}',
            feedback,
        ])
        gen = _make_generator(mock_llm, enable_self_refine=True)

        assert gen.generate([sample_message], [sample_message]) == ["test reply"]
        assert gen.last_feedback_decision == "error"
        assert gen.last_iterate_count == 0

    def test_fact_contradiction_triggers_iterate_before_feedback(self, mock_llm, sample_message):
        mock_llm.responses = [
            '{"skills": []}',
            '{"replies": ["物价高"]}',
            '{"replies": ["物价挺低的"]}',
        ]
        fact_llm = MockLLM(responses=[
            '{"claims": [{"claim": "物价高", "verdict": "contradicted", "reason": "上下文明确说物价低"}]}'
        ])
        gen = _make_generator(mock_llm, enable_self_refine=True)
        gen._fact_check_client = fact_llm

        replies = gen.generate([sample_message], [sample_message])

        assert replies == ["物价挺低的"]
        assert gen.last_feedback_decision == "fail"
        assert gen.last_feedback_issues == ["事实矛盾：物价高；上下文明确说物价低"]
        assert gen.last_iterate_count == 1
        assert len(fact_llm.calls) == 1
        assert len(mock_llm.calls) == 3

    def test_directional_claim_requires_independent_support(self, mock_llm, sample_message):
        mock_llm.responses = [
            '{"skills": []}',
            '{"replies": ["物价高"]}',
            '{"replies": ["物价挺低的"]}',
        ]
        fact_llm = MockLLM(responses=[
            '{"claims": [{"claim": "物价高", "verdict": "entailed", "reason": "初步判断"}]}',
            '{"claims": [{"claim": "物价高", "verdict": "unknown", "reason": "没有直接证据"}]}',
        ])
        gen = _make_generator(mock_llm, enable_self_refine=True)
        gen._fact_check_client = fact_llm

        replies = gen.generate([sample_message], [sample_message])

        assert replies == ["物价挺低的"]
        assert gen.last_feedback_decision == "fail"
        assert gen.last_feedback_issues == ["事实无依据：物价高；没有直接证据"]
        assert gen.last_iterate_count == 1
        assert len(fact_llm.calls) == 2

    def test_iterate_issues_are_structured_json(self, mock_llm, sample_message):
        mock_llm.responses = [
            '{"skills": []}',
            '{"replies": ["bad reply"]}',
            '{"decision": "fail", "issues": ["称呼错误", "包含 <tag>"]}',
            '{"replies": ["改好了"]}',
        ]
        gen = _make_generator(mock_llm, enable_self_refine=True)

        assert gen.generate([sample_message], [sample_message]) == ["改好了"]
        revision = mock_llm.calls[3]["messages"][-2]["content"]
        root = ET.fromstring(revision)
        assert root.tag == "reply_revision"
        assert root.findtext("issues_json") == '["称呼错误", "包含 <tag>"]'
        assert "反馈问题：" not in revision


class TestForceSkill:
    """FORCE_SKILL 环境变量的测试。"""

    def test_force_skill_valid_bypasses_router(self, mock_llm, sample_message):
        """有效 skill 名 → 跳过路由，强制注入。"""
        mock_llm.responses = [
            '{"skills": ["answering_questions"]}',  # 路由应被跳过
            '{"replies": ["test reply"]}',
        ]
        gen = _make_generator(mock_llm, enable_self_refine=False)
        with patch.dict(os.environ, {"FORCE_SKILL": "handling_vent"}):
            gen.generate([sample_message], [sample_message])
        assert "handling_vent" in gen.last_loaded_skills
        assert "answering_questions" not in gen.last_loaded_skills

    def test_force_skill_invalid_ignored(self, mock_llm, sample_message):
        """不存在 skill 名 → 忽略，走正常路由。"""
        mock_llm.responses = [
            '{"skills": ["answering_questions"]}',
            '{"replies": ["test reply"]}',
        ]
        gen = _make_generator(mock_llm, enable_self_refine=False)
        with patch.dict(os.environ, {"FORCE_SKILL": "nonexistent_skill"}):
            gen.generate([sample_message], [sample_message])
        assert "answering_questions" in gen.last_loaded_skills
        assert "nonexistent_skill" not in gen.last_loaded_skills

    def test_force_skill_empty_ignored(self, mock_llm, sample_message):
        """FORCE_SKILL 为空字符串 → 正常路由。"""
        mock_llm.responses = [
            '{"skills": ["casual_chat"]}',
            '{"replies": ["test reply"]}',
        ]
        gen = _make_generator(mock_llm, enable_self_refine=False)
        with patch.dict(os.environ, {"FORCE_SKILL": ""}):
            gen.generate([sample_message], [sample_message])
        assert "casual_chat" in gen.last_loaded_skills


class TestParseReplies:
    """_parse_replies 方法的单元测试。"""

    def test_valid_json_returns_replies(self, mock_llm):
        gen = _make_generator(mock_llm, enable_self_refine=False)
        result = gen._parse_replies('{"replies": ["hello", "world"]}')
        assert result == ["hello", "world"]

    def test_broken_json_returns_empty(self, mock_llm):
        """不完整 JSON（如 {"replies": ["确实气人。"）→ 返回 []，不兜底输出原始串。"""
        gen = _make_generator(mock_llm, enable_self_refine=False)
        result = gen._parse_replies('{"replies": ["确实气人。"')
        assert result == []

    @pytest.mark.parametrize(
        "raw",
        [
            '{"replies": "你好"}',
            '{"replies": {"text": "你好"}}',
            '{"replies": [null, 123, {"text": "你好"}]}',
        ],
    )
    def test_invalid_replies_type_returns_empty(self, mock_llm, raw):
        gen = _make_generator(mock_llm, enable_self_refine=False)
        assert gen._parse_replies(raw) == []

    def test_plain_text_fallback(self, mock_llm):
        """纯文本（无 JSON 特征）→ 正常文本切分回退。"""
        gen = _make_generator(mock_llm, enable_self_refine=False)
        result = gen._parse_replies("第一行\n\n第二行")
        assert result == ["第一行", "第二行"]

    def test_empty_text_returns_empty(self, mock_llm):
        gen = _make_generator(mock_llm, enable_self_refine=False)
        assert gen._parse_replies("") == []
        assert gen._parse_replies("  ") == []


class TestReActLoop:
    def test_broken_json_retries_instead_of_becoming_no_reply(self, sample_message):
        mock_llm = MockLLM(responses=[
            '{"skills": []}',
            '{"replies": ["被截断的回复"',
            '{"replies": ["重试成功"]}',
        ])
        gen = _make_generator(mock_llm, enable_self_refine=False)

        with patch("src.reply.generator.time.sleep"):
            replies = gen.generate([sample_message], [sample_message])

        assert replies == ["重试成功"]
        assert gen.last_generation_failed is False

    def test_skill_router_uses_end_to_end_deadline(self, sample_message):
        mock_llm = MockLLM(responses=['{"skills": []}', '{"replies": ["ok"]}'])
        gen = _make_generator(mock_llm, enable_self_refine=False)

        assert gen.generate([sample_message], [sample_message]) == ["ok"]
        router_timeout = mock_llm.calls[0]["kwargs"]["timeout"]
        generation_timeout = mock_llm.calls[1]["kwargs"]["timeout"]
        assert 0 < generation_timeout <= router_timeout <= 20.0

    def test_reserves_time_for_final_response(self):
        executed = []

        def _response_func(messages, tools=None, **kwargs):
            if tools:
                return MockResponse(tool_calls=[MockToolCall(name="dummy_loop", arguments="{}")])
            return MockResponse(content='{"replies": ["final"]}')

        mock_llm = MockLLM(response_func=_response_func)
        gen = _make_generator(mock_llm, enable_self_refine=False)
        gen.tool_registry.register(
            name="dummy_loop",
            description="测试用工具",
            parameters={"type": "object", "properties": {}},
            func=lambda: executed.append(1) or "ok",
        )

        replies, _ = gen._react_generate(
            [{"role": "user", "content": "test"}],
            gen.tool_registry.to_openai_schemas(),
            deadline=time.time() + 2.0,
        )

        assert replies == ["final"]
        assert executed == []

    def test_max_tool_calls_counts_individual_calls(self, sample_message):
        executed = []
        llm_call_count = 0

        def _response_func(messages, tools=None, **kwargs):
            nonlocal llm_call_count
            llm_call_count += 1
            if llm_call_count == 1:
                return MockResponse(content='{"skills": []}')
            if tools:
                return MockResponse(tool_calls=[
                    MockToolCall(name="dummy_loop", arguments="{}", id=f"tc_{llm_call_count}_{i}")
                    for i in range(3)
                ])
            return MockResponse(content='{"replies": ["forced final"]}')

        mock_llm = MockLLM(response_func=_response_func)
        gen = _make_generator(mock_llm, enable_self_refine=False)
        gen.tool_registry.register(
            name="dummy_loop",
            description="测试用工具",
            parameters={"type": "object", "properties": {}},
            func=lambda: executed.append(1) or "ok",
        )

        assert gen.generate([sample_message], [sample_message]) == ["forced final"]
        assert len(executed) == MAX_TOOL_CALLS

    def test_max_tool_calls_limit(self, sample_message):
        """达到 MAX_TOOL_CALLS 后应强制禁用 tools 并返回 JSON，避免无限循环。"""
        call_count = 0

        def _response_func(messages, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # skill router
                return MockResponse(content='{"skills": []}')
            if tools:
                # 持续返回 dummy 工具调用，迫使进入下一轮 ReAct
                return MockResponse(
                    tool_calls=[
                        MockToolCall(
                            name="dummy_loop",
                            arguments='{"query": "loop"}',
                            id=f"tc_{call_count}",
                        )
                    ]
                )
            # tools 被强制禁用后，必须输出最终 JSON
            return MockResponse(content='{"replies": ["forced final"]}')

        mock_llm = MockLLM(response_func=_response_func)
        gen = _make_generator(mock_llm, enable_self_refine=False)
        # 注册一个 dummy 工具用于测试循环
        gen.tool_registry.register(
            name="dummy_loop",
            description="测试用循环工具",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            func=lambda query: "ok",
        )
        replies = gen.generate([sample_message], [sample_message])

        assert replies == ["forced final"]

        # 调用次数应为：1 次 skill router + MAX_TOOL_CALLS 次 tool 调用 + 1 次强制 JSON
        assert call_count == 1 + MAX_TOOL_CALLS + 1

        # 验证确实存在 dummy_loop 工具调用记录
        assert any(tc["tool_name"] == "dummy_loop" for tc in gen.last_tool_calls)
