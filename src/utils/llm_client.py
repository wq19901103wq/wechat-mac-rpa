"""
Kimi LLM 客户端
兼容之前的配置
⚠️ DEPRECATED: 此客户端已废弃，新项目请使用 QwenClient（src.utils.qwen_client）
"""

import os
from typing import Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("pip install openai")


class KimiClient:
    """Kimi Coding Agent LLM 客户端"""

    def __init__(self):
        agent_name = os.getenv("CODING_AGENT_NAME", "claude-code")

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.kimi.com/coding/v1"),
            default_headers={
                "User-Agent": f"{agent_name}/0.1.39",
                "X-Coding-Agent": agent_name,
                "X-Client-Name": agent_name
            }
        )
        self.model = os.getenv("LLM_MODEL", "kimi-for-coding")
        self.conversations: Dict[str, List[dict]] = {}

    def chat(
        self,
        user_id: Optional[str] = None,
        message: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """生成回复。

        兼容两种调用方式：
        - 旧方式: chat(user_id, message, system_prompt)
        - 新方式: chat(messages=[{"role": "user", "content": "..."}], temperature=0.3)
        """
        # 新接口：直接传入 messages 列表
        if messages is not None:
            conv_messages = list(messages)
        else:
            # 旧接口：使用 user_id + message
            if user_id is None or message is None:
                raise ValueError("KimiClient.chat 需要传入 messages= 或 (user_id, message)")
            if user_id not in self.conversations:
                self.conversations[user_id] = []
                if system_prompt:
                    self.conversations[user_id].append(
                        {"role": "system", "content": system_prompt}
                    )
            self.conversations[user_id].append(
                {"role": "user", "content": message}
            )
            # 限制历史
            if len(self.conversations[user_id]) > 21:
                self.conversations[user_id] = [
                    self.conversations[user_id][0],  # system
                    *self.conversations[user_id][-20:]
                ]
            conv_messages = self.conversations[user_id]

        temp = temperature if temperature is not None else 0.7
        tokens = max_tokens if max_tokens is not None else 1000

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=conv_messages,
                temperature=temp,
                max_tokens=tokens,
                timeout=30,
            )
            reply = response.choices[0].message.content

            # 旧接口模式下保存对话历史
            if messages is None and user_id in self.conversations:
                self.conversations[user_id].append(
                    {"role": "assistant", "content": reply}
                )
            return reply

        except Exception as e:
            _logger = __import__("logging").getLogger("src.llm_client")
            _logger.error(f"LLM 错误: {e}")
            return "抱歉，服务暂时不可用"

    def clear_history(self, user_id: str):
        if user_id in self.conversations:
            del self.conversations[user_id]


if __name__ == "__main__":
    # 测试
    client = KimiClient()
    reply = client.chat("test_user", "你好")
    print(f"回复: {reply}")
