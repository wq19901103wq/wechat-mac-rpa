#!/usr/bin/env python3
"""
批量导入历史记录到 mem0：
1. 一次性 LLM 提取所有事实
2. 直接调 mem0._create_memory() 写入（跳过 Phase 1/2）
"""

import os, json, sys
from pathlib import Path
from datetime import datetime
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

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
            "path": "data/mem0_bulk_import"
        }
    }
}

store = GlobalStore()
target_name = next((n for n in store.chats if "共同富裕" in n), None)
if not target_name:
    print("找不到群")
    sys.exit(1)

state = store.chats[target_name]
messages = state.messages[-500:]  # 先测 500 条

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

# 1. 一次性 LLM 提取所有事实
prompt = f"""从以下微信聊天记录中提取所有关键事实。
要求：
1. 每人每条观点/行为独立成句
2. 包含具体对象（股票/公司/观点/决策）
3. 不同人的观点不要合并
4. 时间信息如有则保留
5. 返回严格 JSON: {{"facts": ["事实1", "事实2", ...]}}

聊天记录：
{full_context}
"""

print("\nLLM 提取事实...")
import openai
client = openai.OpenAI(api_key=DASHSCOPE_API_KEY, base_url=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
    temperature=0.1,
)
facts = json.loads(resp.choices[0].message.content).get("facts", [])
print(f"提取到 {len(facts)} 条事实")

# 2. 初始化 mem0（会清掉旧数据，重新建 collection）
print("\n初始化 mem0...")
memory = Memory.from_config(config)

# 3. 直接用 _create_memory 写入（跳过 Phase 1/2）
print(f"写入 {len(facts)} 条事实到 mem0...")
for i, fact in enumerate(facts, 1):
    memory._create_memory(
        data=fact,
        existing_embeddings={},
        metadata={"user_id": target_name}
    )
    if i % 20 == 0:
        print(f"  已写入 {i}/{len(facts)}")

print(f"\n完成！共写入 {len(facts)} 条事实")

# 4. 用 mem0 搜索验证
print("\n搜索测试（用 mem0.search）：")
for query in ["谁持有腾讯", "清仓阿里", "看好比特币", "加仓", "字节"]:
    results = memory.search(query, filters={"user_id": target_name})
    print(f"\n🔍 '{query}':")
    for r in results.get("results", [])[:3]:
        print(f"   [{r.get('score', 0):.2f}] {r.get('memory', '')}")
