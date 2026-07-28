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
    """LLM API 客户端（支持 DeepSeek 官方 / Zhipu / 自定义 OpenAI 兼容端点）"""

    # 类级 token 统计：按 model 汇总 prompt/completion/total/calls
    _token_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0, "calls": 0})

    def __init__(self, model: str = "", api_key: str = "", base_url: str = ""):
        model = model or os.environ.get("LLM_MODEL", "deepseek-v4-flash")
        api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 或 LLM_API_KEY 未设置")
        base_url = base_url or os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        from urllib.parse import urlparse
        hostname = urlparse(base_url).hostname or ""
        self.is_deepseek_official = hostname == "api.deepseek.com" or hostname.endswith(".deepseek.com")
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

    def chat(self, messages=None, user_id=None, message=None, system_prompt=None, tools=None, temperature=None, max_tokens=None, timeout=None, response_format=None, raise_on_error=False, max_retries=None) -> str:
        """生成回复，支持 tools（function calling）

        支持两种调用方式：
        1. 新接口: chat(messages=[...], tools=[...], temperature=0.3, max_tokens=2000)
        2. 旧接口: chat(user_id="xxx", message="...", system_prompt="...")
        """
        if messages is not None:
            return self._chat_with_messages(messages, tools=tools, temperature=temperature, max_tokens=max_tokens, timeout=timeout, response_format=response_format, raise_on_error=raise_on_error, max_retries=max_retries)
        return self._chat_with_user_id(user_id, message, system_prompt)

    def _chat_with_messages(self, messages: List[dict], tools=None, temperature=None, max_tokens=None, timeout=None, response_format=None, raise_on_error=False, max_retries=None) -> str:
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
            # DeepSeek 官方端点显示启用 thinking 可提升 reasoning_content 质量
            # JSON 模式/路由等需要稳定输出格式时，关闭 thinking 避免模型输出分析性文字
            is_json_mode = isinstance(response_format, dict) and response_format.get("type") == "json_object"
            if self.is_deepseek_official and "deepseek" in self.model.lower():
                thinking_type = "disabled" if is_json_mode else "enabled"
                kwargs["extra_body"] = {"thinking": {"type": thinking_type}}
            elif is_json_mode and "qwen" in self.model.lower():
                kwargs["extra_body"] = {"enable_thinking": False}
            # DeepSeek v4-flash 实测已支持 reasoning_content，默认就会输出 thinking
            _logger = logging.getLogger("src.llm.qwen")
            _logger.info("[Qwen] request start: model=%s tools=%s timeout=%s",
                         kwargs.get("model"), bool(kwargs.get("tools")), kwargs.get("timeout"))
            t_req_start = time.time()
            request_client = self.client.with_options(max_retries=max_retries) if max_retries is not None else self.client
            response = request_client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
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
                    "[Qwen] request end: duration=%.0fms model=%s prompt=%d cache_hit=%d cache_miss=%d completion=%d total=%d",
                    t_req_ms, model_key,
                    getattr(usage, "prompt_tokens", 0) or 0,
                    getattr(usage, "prompt_cache_hit_tokens", 0) or 0,
                    getattr(usage, "prompt_cache_miss_tokens", 0) or 0,
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
            if not text and not reasoning and not getattr(msg, "tool_calls", None):
                _logger.warning("[Qwen] 模型返回空 content 且空 reasoning_content，可能为 endpoint 异常或内容过滤，msg=%s", msg)
            return text
        except Exception as e:
            _logger.warning("Qwen LLM 错误: %s", e, exc_info=True)
            if raise_on_error:
                raise
            return ""

    def _chat_with_user_id(self, user_id: str, message: str, system_prompt: Optional[str] = None) -> str:
        """简单封装"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        return self._chat_with_messages(messages)
