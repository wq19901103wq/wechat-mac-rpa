#!/usr/bin/env python3
"""测试：跳过 Phase 1 后，mem0.add() 能否处理长文本"""

import os, json, sys
from pathlib import Path
from datetime import datetime
import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.session.global_store import GlobalStore
from mem0 import Memory
from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT, generate_additive_extraction_prompt
from mem0.memory.main import _build_session_scope
from mem0.memory.utils import parse_messages

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
            "path": "data/mem0_skip_phase1_test"
        }
    }
}

store = GlobalStore()
target_name = next((n for n in store.chats if "共同富裕" in n), None)
if not target_name:
    print("找不到群")
    sys.exit(1)

state = store.chats[target_name]
messages = state.messages[-500:]

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

# 1. 初始化 mem0
print("\n初始化 mem0...")
memory = Memory.from_config(config)

# 2. 模拟 Phase 0，构造 prompt（跳过 last_messages）
print("\n构造提示词...")
session_scope = _build_session_scope({"user_id": target_name})
last_messages = []  # 批量导入，跳过 last_messages

parsed_messages = parse_messages([{"role": "user", "content": full_context}])

user_prompt = generate_additive_extraction_prompt(
    existing_memories=[],
    new_messages=parsed_messages,
    last_k_messages=last_messages,
    custom_instructions=None,
)

print("\n" + "=" * 60)
print("【系统提示词】")
print("=" * 60)
print(ADDITIVE_EXTRACTION_PROMPT)

print("\n" + "=" * 60)
print("【用户提示词 - 前 3000 字符】")
print("=" * 60)
print(user_prompt[:3000])
print(f"\n... (共 {len(user_prompt)} 字符，已截断)")

# 3. 调 LLM 提取事实
print("\n调 LLM 提取事实...")
import openai
client = openai.OpenAI(api_key=DASHSCOPE_API_KEY, base_url=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))

resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": ADDITIVE_EXTRACTION_PROMPT},
        {"role": "user", "content": user_prompt},
    ],
    response_format={"type": "json_object"},
    temperature=0.1,
)

raw_output = resp.choices[0].message.content
print("\n" + "=" * 60)
print("【LLM 原始输出】")
print("=" * 60)
print(raw_output)

# 4. 解析事实
try:
    facts = json.loads(raw_output, strict=False).get("memory", [])
except Exception as e:
    print(f"解析失败: {e}")
    facts = []

print(f"\n提取到 {len(facts)} 条事实")

# 5. 用改过的 mem0.add() 入库（跳过 Phase 1）
print("\n" + "=" * 60)
print("用 memory.add() 入库（Phase 1 已跳过）...")
print("=" * 60)

try:
    result = memory.add(full_context, user_id=target_name)
    print(f"成功！返回 {len(result['results'])} 条结果")
    for r in result['results'][:10]:
        print(f"  [{r['event']}] {r['memory'][:80]}")
    if len(result['results']) > 10:
        print(f"  ... 还有 {len(result['results']) - 10} 条")
except Exception as e:
    print(f"失败: {type(e).__name__}: {e}")

# 6. 搜索验证
print("\n" + "=" * 60)
print("搜索验证")
print("=" * 60)
for query in ["谁持有腾讯", "清仓阿里", "看好比特币", "加仓"]:
    try:
        results = memory.search(query, filters={"user_id": target_name})
        print(f"\n🔍 '{query}':")
        for r in results.get("results", [])[:3]:
            print(f"   [{r.get('score', 0):.2f}] {r.get('memory', '')}")
    except Exception as e:
        print(f"\n🔍 '{query}': 搜索失败 {e}")
