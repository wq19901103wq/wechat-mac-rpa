#!/usr/bin/env python3
"""批量灌库最优版：LLM 提取只做 1 次"""

import os, json, uuid, sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.session.global_store import GlobalStore
from openai import OpenAI
import chromadb
import logging

_logger = logging.getLogger(__name__)

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))

store = GlobalStore()
target_name = next((n for n in store.chats if "共同富裕" in n), None)
if not target_name:
    print("找不到群")
    sys.exit(1)

state = store.chats[target_name]
messages = state.messages[-2000:]

# 1. 格式化成行（不分 chunk，全部保留）
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
print(f"群: {target_name} | 总消息: {len(messages)} | 总字符: {len(full_context)}")

# 2. 一次性调 LLM 提取所有事实（1 次调用！）
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

print("\n调 LLM 提取事实...")
resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
)
facts = json.loads(resp.choices[0].message.content).get("facts", [])
print(f"提取到 {len(facts)} 条事实")

# 3. 批量 embed（1 次调用）
print("批量 embed...")
embed_resp = client.embeddings.create(input=facts, model="text-embedding-v3")
embeddings = [e.embedding for e in embed_resp.data]

# 4. 批量入库（1 次调用）
chroma = chromadb.PersistentClient(path="data/wechat_bulk_optimal")
collection = chroma.get_or_create_collection(target_name)
collection.add(
    ids=[str(uuid.uuid4()) for _ in facts],
    embeddings=embeddings,
    documents=facts,
    metadatas=[{"source": target_name} for _ in facts],
)
print(f"完成！入库 {len(facts)} 条事实\n")

# 展示提取的事实
print("=" * 60)
print("提取的事实样例（前 50 条）：")
print("=" * 60)
for i, fact in enumerate(facts[:50], 1):
    print(f"{i}. {fact}")

if len(facts) > 50:
    print(f"\n... 还有 {len(facts) - 50} 条")

# 搜索测试
print("\n" + "=" * 60)
print("搜索测试：")
print("=" * 60)
for query in ["谁持有腾讯", "清仓阿里", "看好比特币", "推荐", "职级"]:
    q_embed = client.embeddings.create(input=[query], model="text-embedding-v3").data[0].embedding
    results = collection.query(query_embeddings=[q_embed], n_results=3)
    print(f"\n🔍 '{query}':")
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        print(f"   [{dist:.3f}] {doc}")
