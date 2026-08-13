#!/usr/bin/env python3
"""
单群 Mem0 事实提取测试 - 2000条分chunk导入
- 选示例交流群
- 2000条消息按embedding限制切分，分块导入
"""
import os
import sys
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
            "path": "data/mem0_chroma_2000batch"
        }
    }
}

store = GlobalStore()

target_name = None
for name in store.chats:
    if "共同富裕" in name:
        target_name = name
        break

if not target_name:
    print("找不到示例交流群")
    sys.exit(1)

state = store.chats[target_name]
msg_count = len(state.messages)
print(f"群: {target_name} | 总消息: {msg_count}", flush=True)

limit = min(msg_count, 2000)
messages = state.messages[-limit:]
print(f"导入最近 {limit} 条消息...", flush=True)

# 格式化为行
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

# 按embedding限制切分 chunk（每 chunk < 4000 字符，留安全余量）
MAX_CHUNK = 4000
chunks = []
current = []
current_len = 0
for line in lines:
    if current_len + len(line) + 1 > MAX_CHUNK and current:
        chunks.append("\n".join(current))
        current = [line]
        current_len = len(line)
    else:
        current.append(line)
        current_len += len(line) + 1
if current:
    chunks.append("\n".join(current))

print(f"切成 {len(chunks)} 个 chunk，每块约 {MAX_CHUNK} 字符", flush=True)

memory = Memory.from_config(config)

for i, chunk in enumerate(chunks, 1):
    print(f"  导入 chunk {i}/{len(chunks)} ({len(chunk)} 字符)...", flush=True)
    memory.add(chunk, user_id=target_name, metadata={"chunk": i, "total_chunks": len(chunks)})
    print(f"    完成", flush=True)

print(f"\n提取的记忆片段:", flush=True)
all_memories = memory.get_all(filters={"user_id": target_name})
mem_list = all_memories.get("memories", [])
print(f"共 {len(mem_list)} 条记忆\n", flush=True)

for m in mem_list[:50]:
    print(f"  - {m['memory']}", flush=True)

if len(mem_list) > 50:
    print(f"  ... 还有 {len(mem_list) - 50} 条", flush=True)

print(f"\n搜索测试:", flush=True)
for query in ["谁持有", "清仓", "加仓", "推荐", "看好", "字节", "职级"]:
    print(f"\n  Query: '{query}'", flush=True)
    results = memory.search(query, filters={"user_id": target_name})
    for r in results.get("results", [])[:3]:
        print(f"    [{r['score']:.2f}] {r['memory']}", flush=True)
