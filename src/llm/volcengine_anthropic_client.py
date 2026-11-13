"""QwenClient-compatible client for Volcengine's Anthropic endpoint."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import requests


DEFAULT_VOLCENGINE_CONFIG = Path.home() / ".cc-connect/config.toml"


def load_volcengine_credentials(config_path: Path = DEFAULT_VOLCENGINE_CONFIG) -> tuple[str, str]:
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        import tomli as tomllib
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    project = next(row for row in config.get("projects", []) if row.get("name") == "feishu-connect")
    agent = project["agent"]
    provider = next(row for row in agent.get("providers", []) if row.get("name") == agent.get("provider"))
    credential = provider.get("api_key") or provider.get("auth_token")
    base_url = provider.get("base_url")
    if not credential or not base_url:
        raise ValueError("feishu-connect provider is missing its credential or base_url")
    return str(credential), str(base_url).rstrip("/")


class VolcengineAnthropicClient:
    def __init__(self, api_key: str, base_url: str, model: str, session: Any = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.resolved_model = ""
        self.last_thinking = ""
        self.last_request_body: dict[str, Any] = {}
        self.last_response_payload: dict[str, Any] = {}
        self.last_status_code: int | None = None
        self.last_elapsed_ms: int | None = None
        if session is None:
            session = requests.Session()
            session.trust_env = False
        self.session = session

    @staticmethod
    def _tool_definitions(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        return [{
            "name": row.get("function", {}).get("name", ""),
            "description": row.get("function", {}).get("description", ""),
            "input_schema": row.get("function", {}).get("parameters") or {"type": "object", "properties": {}},
        } for row in tools or []]

    @staticmethod
    def _messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []

        def append(role: str, blocks: list[dict[str, Any]]) -> None:
            if converted and converted[-1]["role"] == role:
                converted[-1]["content"].extend(blocks)
            else:
                converted.append({"role": role, "content": blocks})

        for row in messages:
            role = row.get("role")
            content = row.get("content") or ""
            if role == "system":
                system_parts.append(str(content))
                continue
            if role == "tool":
                append("user", [{"type": "tool_result", "tool_use_id": row.get("tool_call_id", ""), "content": str(content)}])
                continue
            blocks = [{"type": "text", "text": str(content)}] if content else []
            if role == "assistant":
                for tool_call in row.get("tool_calls") or []:
                    function = tool_call.get("function", {})
                    arguments = function.get("arguments") or "{}"
                    try:
                        tool_input = json.loads(arguments) if isinstance(arguments, str) else arguments
                    except json.JSONDecodeError:
                        tool_input = {"raw": arguments}
                    blocks.append({"type": "tool_use", "id": tool_call.get("id", ""), "name": function.get("name", ""), "input": tool_input})
            append("assistant" if role == "assistant" else "user", blocks or [{"type": "text", "text": ""}])
        return "\n\n".join(system_parts), converted

    def chat(self, messages=None, user_id=None, message=None, system_prompt=None, tools=None,
             temperature=None, max_tokens=None, timeout=None, response_format=None,
             raise_on_error=False, max_retries=None):
        del user_id, response_format
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": message or ""})
        system, anthropic_messages = self._messages(messages)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 1000,
            "temperature": 0.7 if temperature is None else temperature,
        }
        if system:
            body["system"] = system
        converted_tools = self._tool_definitions(tools)
        if converted_tools:
            body["tools"] = converted_tools
        attempts = max(1, (max_retries or 0) + 1)
        try:
            response = None
            started = time.monotonic()
            self.last_request_body = body
            for attempt in range(attempts):
                response = self.session.post(
                    f"{self.base_url}/v1/messages",
                    headers={"content-type": "application/json", "x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
                    json=body,
                    timeout=timeout or 500,
                )
                if response.status_code < 500 or attempt + 1 >= attempts:
                    break
            assert response is not None
            self.last_elapsed_ms = round((time.monotonic() - started) * 1000)
            self.last_status_code = response.status_code
            response.raise_for_status()
            payload = response.json()
            self.last_response_payload = payload
            self.resolved_model = str(payload.get("model") or "")
            text_parts, thinking_parts, tool_calls = [], [], []
            for block in payload.get("content") or []:
                if block.get("type") == "text":
                    text_parts.append(str(block.get("text") or ""))
                elif block.get("type") == "thinking":
                    thinking_parts.append(str(block.get("thinking") or ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append(SimpleNamespace(
                        id=str(block.get("id") or ""), type="function",
                        function=SimpleNamespace(name=str(block.get("name") or ""), arguments=json.dumps(block.get("input") or {}, ensure_ascii=False)),
                    ))
            self.last_thinking = "\n".join(thinking_parts)
            text = "\n".join(part for part in text_parts if part)
            return SimpleNamespace(content=text, tool_calls=tool_calls, reasoning_content=self.last_thinking) if tool_calls else text
        except Exception:
            if raise_on_error:
                raise
            return ""
