#!/usr/bin/env python3
"""验证 deepseek-v4-flash 在 ReAct round 1 是否稳定返回空 content"""

import os
from openai import OpenAI

# 加载 .env
with open(".env") as f:
    for line in f:
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k, v)

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-v4-flash"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

tools = [{
    "type": "function",
    "function": {
        "name": "search_memory",
        "description": "搜索本地记忆",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    }
}]

def test_round0():
    """round 0: 带 tools，让模型决定调不调工具"""
    messages = [
        {"role": "system", "content": "你是一个微信助手，简短回复。"},
        {"role": "user", "content": "示例用户壬是谁？"},
    ]
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, tools=tools, max_tokens=2000, timeout=30
    )
    msg = resp.choices[0].message
    print(f"[round 0] content={msg.content!r} tool_calls={getattr(msg, 'tool_calls', None)}")
    return msg

def test_round1_with_tool_result():
    """round 1: 模拟 tool 结果已回注，再传 tools"""
    messages = [
        {"role": "system", "content": "你是一个微信助手，简短回复。"},
        {"role": "user", "content": "示例用户壬是谁？"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_abc123", "type": "function", "function": {"name": "search_memory", "arguments": '{"query": "示例用户壬"}'}}
        ]},
        {"role": "tool", "tool_call_id": "call_abc123", "content": "示例用户壬是示例用户甲的朋友，2023年一起旅游认识的。"},
    ]
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, tools=tools, max_tokens=2000, timeout=30
    )
    msg = resp.choices[0].message
    content = msg.content or ""
    print(f"[round 1] content={content!r} empty={not content.strip()} tool_calls={getattr(msg, 'tool_calls', None)}")
    return content

def test_round1_no_tools():
    """round 1: 模拟 tool 结果已回注，但不再传 tools"""
    messages = [
        {"role": "system", "content": "你是一个微信助手，简短回复。"},
        {"role": "user", "content": "示例用户壬是谁？"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_abc123", "type": "function", "function": {"name": "search_memory", "arguments": '{"query": "示例用户壬"}'}}
        ]},
        {"role": "tool", "tool_call_id": "call_abc123", "content": "示例用户壬是示例用户甲的朋友，2023年一起旅游认识的。"},
    ]
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=2000, timeout=30
    )
    msg = resp.choices[0].message
    content = msg.content or ""
    print(f"[round 1 no-tools] content={content!r} empty={not content.strip()}")
    return content

if __name__ == "__main__":
    print(f"模型: {MODEL}\n")

    print("=== 测试1: round 0（带 tools）===")
    for _ in range(3):
        msg = test_round0()
        if getattr(msg, "tool_calls", None):
            print(f"  → 返回了 tool_calls")
        else:
            print(f"  → 直接返回 text")

    print("\n=== 测试2: round 1（tool 结果回注 + 再传 tools）===")
    empty_count = 0
    for _ in range(5):
        content = test_round1_with_tool_result()
        if not content.strip():
            empty_count += 1
    print(f"\n  空返回率: {empty_count}/5")

    print("\n=== 测试3: round 1（tool 结果回注 + 不传 tools）===")
    empty_count = 0
    for _ in range(3):
        content = test_round1_no_tools()
        if not content.strip():
            empty_count += 1
    print(f"\n  空返回率: {empty_count}/3")
