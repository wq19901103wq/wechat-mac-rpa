#!/usr/bin/env python3
"""ReAct + Self-Refine 真实 API 集成测试。

运行前需要设置 DEEPSEEK_API_KEY 环境变量（或已由 .env 文件提供）。
真实 API 测试会产生 token 费用，默认通过 skipif 在未设置 key 时跳过。
"""

import os
import sys
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

# 将项目根目录加入 Python 路径，以便 import src 包
sys.path.insert(0, str(Path(__file__).parent.parent))

# 优先从项目根目录 .env 加载环境变量
load_dotenv(Path(__file__).parent.parent / ".env")

from src.models.base import ChatMessage, SenderType  # noqa: E402
from src.reply.generator import ReplyGenerator  # noqa: E402
from src.utils.qwen_client import QwenClient  # noqa: E402


# 只有在设置了真实 API key 时才执行，避免无 key 时失败或产生费用。
HAS_DEEPSEEK_API_KEY = bool(os.environ.get("DEEPSEEK_API_KEY"))


# 简单投资类问题，用于触发 think 工具调用。
_INVESTMENT_QUESTION = "拼多多还能拿吗"


def _make_message(text: str) -> ChatMessage:
    return ChatMessage(
        text=text,
        sender="示例用户申",
        sender_type=SenderType.OTHER,
        chat_name="测试群",
    )


@pytest.mark.skipif(
    not HAS_DEEPSEEK_API_KEY,
    reason="DEEPSEEK_API_KEY not set",
)
def test_investment_question_triggers_think():
    """投资类问题应触发 think 工具调用，并在 20s 内产生回复。"""
    client = QwenClient("deepseek-v4-flash")
    gen = ReplyGenerator(llm_client=client)

    replies = gen.generate(
        [_make_message(_INVESTMENT_QUESTION)],
        [],
        is_group=True,
    )

    assert replies, "应产生非空回复"
    tool_names = [tc.get("tool_name", "") for tc in gen.last_tool_calls]
    assert "think" in tool_names, f"投资类问题应调用 think 工具，实际调用: {tool_names}"


@pytest.mark.skipif(
    not HAS_DEEPSEEK_API_KEY,
    reason="DEEPSEEK_API_KEY not set",
)
def test_total_timeout_under_20s(monkeypatch):
    """整个生成流程（含 Self-Refine）应在 20s 内完成。"""
    monkeypatch.setenv("ENABLE_SELF_REFINE", "1")
    client = QwenClient("deepseek-v4-flash")
    gen = ReplyGenerator(llm_client=client)

    start = time.time()
    gen.generate(
        [_make_message(_INVESTMENT_QUESTION)],
        [],
        is_group=True,
    )
    elapsed = time.time() - start

    assert elapsed < 20.0, f"总耗时 {elapsed:.2f}s 超过 20s 预算"


@pytest.mark.skipif(
    not HAS_DEEPSEEK_API_KEY,
    reason="DEEPSEEK_API_KEY not set",
)
def test_self_refine_produces_decision(monkeypatch):
    """启用 Self-Refine 后，应产生非空的 feedback decision 记录。"""
    monkeypatch.setenv("ENABLE_SELF_REFINE", "1")
    client = QwenClient("deepseek-v4-flash")
    gen = ReplyGenerator(llm_client=client)

    replies = gen.generate(
        [_make_message(_INVESTMENT_QUESTION)],
        [],
        is_group=True,
    )

    assert replies, "应产生非空回复"
    assert gen.last_self_refine_applied is True, "应标记 Self-Refine 已应用"
    assert gen.last_feedback_decision, f"feedback decision 不应为空，原始输出: {gen.last_feedback_raw!r}"
