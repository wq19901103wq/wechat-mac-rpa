#!/usr/bin/env python3
"""QwenClient 单元测试"""

import os
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import pytest

from src.utils.qwen_client import QwenClient


class TestQwenClientInit:
    def test_dashscope_default(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
            client = QwenClient()
            assert client.model == "deepseek-v4-flash"
            assert not client.is_deepseek_official
            assert "aliyuncs" in str(client.client.base_url)

    def test_deepseek_official(self):
        with patch.dict(os.environ, {
            "USE_DEEPSEEK_OFFICIAL": "true",
            "DEEPSEEK_API_KEY": "ds-key"
        }, clear=True):
            client = QwenClient()
            assert client.is_deepseek_official
            parsed = urlparse(str(client.client.base_url))
            assert parsed.hostname == "api.deepseek.com"

    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
                QwenClient()


class TestQwenClientChat:
    @patch("src.utils.qwen_client.OpenAI")
    def test_chat_with_messages(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="hello", tool_calls=None))]
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "key"}, clear=True):
            client = QwenClient()
            result = client.chat(messages=[{"role": "user", "content": "hi"}], temperature=0.3, max_tokens=500)
            assert result == "hello"
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["temperature"] == 0.3
            assert call_kwargs["max_tokens"] == 500

    @patch("src.utils.qwen_client.OpenAI")
    def test_chat_with_tools_returns_tool_calls(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_tool_call = {"name": "search_memory", "arguments": '{"query": "test"}'}
        mock_response = MagicMock()
        mock_msg = MagicMock(content="", tool_calls=[mock_tool_call])
        mock_response.choices = [MagicMock(message=mock_msg)]
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "key"}, clear=True):
            client = QwenClient()
            result = client.chat(messages=[{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
            assert result == mock_msg

    @patch("src.utils.qwen_client.OpenAI")
    def test_chat_error_returns_empty(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "key"}, clear=True):
            client = QwenClient()
            result = client.chat(messages=[{"role": "user", "content": "hi"}])
            assert result == ""

    @patch("src.utils.qwen_client.OpenAI")
    def test_deepseek_thinking_enabled(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok", tool_calls=None))]
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(os.environ, {
            "USE_DEEPSEEK_OFFICIAL": "true",
            "DEEPSEEK_API_KEY": "key"
        }, clear=True):
            client = QwenClient()
            client.chat(messages=[{"role": "user", "content": "hi"}])
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["extra_body"] == {"thinking": {"type": "enabled"}}

    @patch("src.utils.qwen_client.OpenAI")
    def test_chat_legacy_interface(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="legacy reply", tool_calls=None))]
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "key"}, clear=True):
            client = QwenClient()
            result = client.chat(user_id="u1", message="hello", system_prompt="sys")
            assert result == "legacy reply"
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            msgs = call_kwargs["messages"]
            assert msgs[0]["role"] == "system"
            assert msgs[0]["content"] == "sys"
            assert msgs[1]["role"] == "user"
            assert msgs[1]["content"] == "hello"
