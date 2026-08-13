#!/usr/bin/env python3
"""测试：用自定义 prompt 让 mem0 提取群聊结构化信息"""

import os, sys
from pathlib import Path
from datetime import datetime
import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.session.global_store import GlobalStore
from mem0 import Memory

_logger = logging.getLogger(__name__)

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "deepseek-v4-flash",
            "api_key": DASHSCOPE_API_KEY,
            "openai_base_url": os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-v3",
            "api_key": DASHSCOPE_API_KEY,
            "openai_base_url": os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        }
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "path": "data/mem0_custom_prompt_test"
        }
    }
}

store = GlobalStore()
target_name = "打工人和退休干部们"
if target_name not in store.chats:
    print(f"找不到群: {target_name}")
    sys.exit(1)

state = store.chats[target_name]
messages = state.messages[-100:]

lines = []
for m in messages:
    ts = m.create_time
    tstr = ""
    if ts:
        try:
            tstr = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
        except Exception as e:
            _logger.warning("timestamp conversion failed: %s", e)
    lines.append(f"[{tstr}] {m.sender}: {m.text}")

full_context = "\n".join(lines)
print(f"群: {target_name} | 消息: {len(messages)} | 字符: {len(full_context)}")

# 自定义提取指令 - 通用化，不针对特定主题
custom_prompt = """这是群聊记录，不是个人对话。请提取所有有价值的结构化信息，包括但不限于：
- 人物身份信息（职业、职级、公司、关系）
- 观点态度（对事物、公司、趋势、政策的看法）
- 行为操作（买卖、决策、行动）
- 计划意图（目标、安排、将要做什么）
- 事实陈述（已发生的事件、已知信息）
- 人物关系（谁和谁是什么关系）

每条提取必须包含：具体人物 + 时间（如有）+ 具体内容。
不要遗漏任何有信息量的内容。"""

print(f"\n自定义 prompt:\n{custom_prompt}\n")

# 初始化 mem0
print("初始化 mem0...")
memory = Memory.from_config(config)

# 用自定义 prompt 调用 add()
print("调用 memory.add() 提取...")
try:
    result = memory.add(full_context, user_id=target_name, prompt=custom_prompt)
    print(f"成功！返回 {len(result['results'])} 条结果\n")
    
    for i, r in enumerate(result['results'], 1):
        print(f"{i}. [{r['event']}] {r['memory']}")
        
except Exception as e:
    print(f"失败: {type(e).__name__}: {e}")

# 搜索验证
print("\n搜索验证:")
for query in ["谁", "什么", "怎么样", "买了", "去了"]:
    try:
        results = memory.search(query, filters={"user_id": target_name})
        print(f"\n🔍 '{query}':")
        for r in results.get("results", [])[:3]:
            print(f"   [{r.get('score', 0):.2f}] {r.get('memory', '')}")
    except Exception as e:
        print(f"\n🔍 '{query}': 搜索失败 {e}")
