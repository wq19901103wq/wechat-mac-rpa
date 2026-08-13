#!/usr/bin/env python3
"""测试 Mem0 事实提取能力"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mem0 import Memory

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
            "path": "data/mem0_chroma"
        }
    }
}

print("初始化 Mem0...")
memory = Memory.from_config(config)

# 测试数据：模拟微信对话
test_messages = [
    "示例用户辰说：我今天清仓阿里了，加了腾讯",
    "示例用户巳说：阿里我也早走了，现在只持有拼多多和小米",
    "阿杰说：MSTR leap call 继续持有，看好比特币",
]

print("\n添加记忆...")
for msg in test_messages:
    result = memory.add(msg, user_id="test_user", metadata={"source": "投资交流群"})
    print(f"  ✓ {msg[:40]}...")

print("\n搜索记忆：'谁持有腾讯？'")
results = memory.search("谁持有腾讯？", filters={"user_id": "test_user"})
for r in results.get("results", [])[:3]:
    print(f"  [{r['score']:.2f}] {r['memory']}")

print("\n搜索记忆：'谁清仓了阿里？'")
results = memory.search("谁清仓了阿里？", filters={"user_id": "test_user"})
for r in results.get("results", [])[:3]:
    print(f"  [{r['score']:.2f}] {r['memory']}")

print("\n全部提取的记忆片段:")
all_memories = memory.get_all(filters={"user_id": "test_user"})
for m in all_memories.get("memories", []):
    print(f"  - [{m['created_at'][:10]}] {m['memory']}")
