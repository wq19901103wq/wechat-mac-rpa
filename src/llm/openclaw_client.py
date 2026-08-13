#!/usr/bin/env python3
"""OpenClaw LLM 客户端

通过 OpenClaw Gateway / Kimi Code 代理调用大模型。
默认连接本地代理 http://127.0.0.1:18790，模型 kimi-for-coding。
"""

import logging
import os
import time
from typing import Optional


def _get_openai_client():
    """延迟导入 openai，避免未安装时崩溃"""
    try:
        from openai import OpenAI
        return OpenAI
    except ImportError:
        raise ImportError(
            "使用 OpenClawClient 需要安装 openai 库: pip install openai"
        )


class OpenClawClient:
    """OpenClaw / Kimi Code 代理客户端"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:18790",
        api_key: Optional[str] = None,
        model: str = "kimi-for-coding",
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        timeout: float = 30.0,
    ):
        OpenAI = _get_openai_client()
        # openai v2.x 不会自动加 /v1 前缀，需手动处理
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = base_url + "/v1"
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENCLAW_API_KEY", "openclaw-local"),
            timeout=timeout,
        )
        self.model = model
        # 注意：kimi-for-coding 的 reasoning_content 也占用 max_tokens 配额。
        # 1024 足够覆盖典型思考过程（~300 chars）+ 回复内容（~50 chars）。
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or (
            "你是一个友好、简洁的微信助手。回复要求："
            "1. 友好自然；"
            "2. 简洁（不超过50字）；"
            "3. 群聊中被@时直接回答问题。"
        )

    def chat(self, messages=None, tools=None, temperature=None, max_tokens=None, timeout=None):
        """
        调用 OpenClaw 生成回复。接口与 QwenClient 兼容。

        Args:
            messages: OpenAI 格式的消息列表
            tools: function calling 工具列表（可选）
            temperature: 采样温度（可选）
            max_tokens: 最大 token 数（可选，默认 self.max_tokens）
            timeout: 超时秒数（可选）

        Returns:
            str: 普通文本回复
            或 message 对象（当模型返回 tool_calls 时，与 QwenClient 行为一致）
        """
        if messages is None:
            messages = []
        # 如果没有 system prompt，注入默认的
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            messages = [{"role": "system", "content": self.system_prompt}] + messages

        try:
            kwargs = {
                "model": self.model,
                "messages": messages,  # type: ignore[arg-type]
                "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if tools:
                kwargs["tools"] = tools
            if timeout is not None:
                kwargs["timeout"] = timeout

            _logger = logging.getLogger("src.llm.openclaw")
            _logger.info("[OpenClaw] request start: model=%s tools=%s timeout=%s",
                         kwargs.get("model"), bool(kwargs.get("tools")), kwargs.get("timeout"))
            t_req_start = time.time()
            resp = self.client.chat.completions.create(**kwargs)
            t_req_ms = (time.time() - t_req_start) * 1000
            _logger.info("[OpenClaw] request end: duration=%.0fms model=%s",
                         t_req_ms, kwargs.get("model"))
            choice = resp.choices[0]
            msg = choice.message

            # 如果模型返回 tool_calls，返回 message 对象（与 QwenClient 行为一致）
            if getattr(msg, "tool_calls", None):
                return msg

            content = msg.content or ""
            finish_reason = getattr(choice, "finish_reason", "")

            # kimi-for-coding 的思考过程（reasoning_content）也占用 max_tokens。
            # 当 max_tokens 不足时 content 为空，finish_reason="length"。
            # 此时绝不能从 reasoning_content 取内容——那是思考过程，不是回复。
            if not content.strip():
                reasoning = getattr(msg, "reasoning_content", "")
                if reasoning:
                    # 记录诊断信息，便于排查 token 是否足够
                    logging.getLogger("src.openclaw").warning(
                        f"LLM content 为空 (finish_reason={finish_reason!r}, "
                        f"reasoning_len={len(reasoning)}, max_tokens={kwargs['max_tokens']})"
                    )
                # 返回空字符串让上层 generator 知道调用失败，绝不能返回"收到"
                raise RuntimeError(f"LLM content empty, max_tokens={kwargs['max_tokens']} insufficient")

            return content.strip()
        except Exception as exc:
            raise RuntimeError(f"OpenClaw 调用失败: {exc}") from exc

    @classmethod
    def from_openclaw_config(cls, config_path: Optional[str] = None) -> "OpenClawClient":
        """从 OpenClaw 配置文件自动读取 base_url 和模型配置"""
        import json
        from pathlib import Path

        if config_path is None:
            config_path = str(Path.home() / ".openclaw" / "openclaw.json")

        if not os.path.exists(config_path):
            # 使用默认值
            return cls()

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        providers = cfg.get("models", {}).get("providers", {})
        openai_cfg = providers.get("openai", {})
        base_url = openai_cfg.get("baseUrl", "http://127.0.0.1:18790")
        models = openai_cfg.get("models", [])
        model = models[0]["id"] if models else "kimi-for-coding"

        return cls(base_url=base_url, model=model)
