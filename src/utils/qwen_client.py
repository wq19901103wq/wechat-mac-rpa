"""
Qwen LLM 客户端 - 用于回复生成
接口与 KimiClient 兼容
"""

import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("pip install openai")


class QwenClient:
    """LLM API 客户端（支持 DeepSeek 官方 / DashScope 双平台）"""

    # 类级 token 统计：按 model 汇总 prompt/completion/total/calls
    _token_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0, "calls": 0})

    def __init__(self, model: str = "deepseek-v4-flash"):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未设置")
        base_url = "https://api.deepseek.com/v1"
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.is_deepseek_official = True
        self.last_thinking = ""  # 最近一次 LLM 推理过程

    @classmethod
    def get_token_stats(cls) -> Dict[str, Dict[str, int]]:
        """返回按模型汇累计 token 消耗统计。"""
        return dict(cls._token_stats)

    @classmethod
    def reset_token_stats(cls):
        """清空累计 token 统计。"""
        cls._token_stats.clear()

    @classmethod
    def log_token_stats(cls, logger: logging.Logger):
        """输出累计 token 消耗到日志。"""
        stats = cls.get_token_stats()
        if not stats:
            logger.info("[TokenStats] 暂无 DeepSeek API token 消耗记录")
            return
        for model, s in stats.items():
            logger.info(
                "[TokenStats] model=%s calls=%d prompt=%d completion=%d total=%d",
                model, s["calls"], s["prompt"], s["completion"], s["total"]
            )

    def chat(self, messages=None, user_id=None, message=None, system_prompt=None, tools=None, temperature=None, max_tokens=None, timeout=None, response_format=None) -> str:
        """生成回复，支持 tools（function calling）

        支持两种调用方式：
        1. 新接口: chat(messages=[...], tools=[...], temperature=0.3, max_tokens=2000)
        2. 旧接口: chat(user_id="xxx", message="...", system_prompt="...")
        """
        if messages is not None:
            return self._chat_with_messages(messages, tools=tools, temperature=temperature, max_tokens=max_tokens, timeout=timeout, response_format=response_format)
        return self._chat_with_user_id(user_id, message, system_prompt)

    def _chat_with_messages(self, messages: List[dict], tools=None, temperature=None, max_tokens=None, timeout=None, response_format=None) -> str:
        """直接透传 messages 列表调用大模型，支持 tools 和自定义参数"""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else 0.7,
                "max_tokens": max_tokens if max_tokens is not None else 1000,
                "timeout": timeout if timeout is not None else 500,
            }
            if tools:
                kwargs["tools"] = tools
            if response_format:
                kwargs["response_format"] = response_format
            # DeepSeek 官方平台：thinking 仅 reasoner 支持，flash/pro 开了会空回复
            _logger = logging.getLogger("src.llm.qwen")
            _logger.info("[Qwen] request start: model=%s tools=%s timeout=%s",
                         kwargs.get("model"), bool(kwargs.get("tools")), kwargs.get("timeout"))
            t_req_start = time.time()
            response = self.client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
            t_req_ms = (time.time() - t_req_start) * 1000
            # 记录 token 消耗
            usage = getattr(response, "usage", None)
            if usage:
                model_key = str(kwargs.get("model", self.model))
                self._token_stats[model_key]["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
                self._token_stats[model_key]["completion"] += getattr(usage, "completion_tokens", 0) or 0
                self._token_stats[model_key]["total"] += getattr(usage, "total_tokens", 0) or 0
                self._token_stats[model_key]["calls"] += 1
                _logger.info(
                    "[Qwen] request end: duration=%.0fms model=%s prompt=%d completion=%d total=%d",
                    t_req_ms, model_key,
                    getattr(usage, "prompt_tokens", 0) or 0,
                    getattr(usage, "completion_tokens", 0) or 0,
                    getattr(usage, "total_tokens", 0) or 0,
                )
            else:
                _logger.info("[Qwen] request end: duration=%.0fms model=%s",
                             t_req_ms, kwargs.get("model"))
            msg = response.choices[0].message
            # 日志记录思考过程（在 tool_calls 判断之前）
            reasoning = getattr(msg, "reasoning_content", None) or ""
            if reasoning:
                self.last_thinking = reasoning
                _logger.info("[Qwen] thinking: %s", reasoning[:800])
            # 日志记录内容
            text = msg.content or ""
            if text:
                _logger.info("[Qwen] output: %s", text[:500])
            # 如果模型返回 tool_calls，也返回（让上层处理）
            if getattr(msg, "tool_calls", None):
                return msg  # type: ignore[return-value]
            # DeepSeek 某些模型（如 v4-pro）在长 prompt 下会把输出放在 reasoning_content 而非 content
            if not text and reasoning:
                text = reasoning
            return text
        except Exception as e:
            print(f"Qwen LLM 错误: {e}")
            return ""

    def _chat_with_user_id(self, user_id: str, message: str, system_prompt: Optional[str] = None) -> str:
        """简单封装"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        return self._chat_with_messages(messages)
