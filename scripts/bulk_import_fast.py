#!/usr/bin/env python3
"""批量灌库：LLM 1 次 + embed 分批(每批<=10) + 入库"""

import os, json, uuid, sys, re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

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

# 1. LLM 一次性提取
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

print("LLM 提取中...")
resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
    temperature=0.1,
)
facts = json.loads(resp.choices[0].message.content).get("facts", [])
print(f"提取到 {len(facts)} 条事实\n")

# 2. Embed 分批（DashScope 限制 batch<=10）
print("Embed 分批处理...")
BATCH = 10
embeddings = []
for i in range(0, len(facts), BATCH):
    batch = facts[i : i + BATCH]
    r = client.embeddings.create(input=batch, model="text-embedding-v3")
    embeddings.extend([e.embedding for e in r.data])
    print(f"  批次 {i // BATCH + 1}/{(len(facts) - 1) // BATCH + 1} 完成")

# 3. 入库
coll_name = re.sub(r"[^a-zA-Z0-9_-]", "_", target_name)
chroma = chromadb.PersistentClient(path="data/wechat_bulk_optimal")
try:
    chroma.delete_collection(coll_name)
except Exception as e:
    _logger.warning("delete collection failed: %s", e)
collection = chroma.get_or_create_collection(coll_name)
collection.add(
    ids=[str(uuid.uuid4()) for _ in facts],
    embeddings=embeddings,
    documents=facts,
    metadatas=[{"source": target_name} for _ in facts],
)
print(f"\n入库 {len(facts)} 条完成！")

# 4. 展示全部事实
print("\n" + "=" * 60)
print(f"全部 {len(facts)} 条事实：")
print("=" * 60)
for i, fact in enumerate(facts, 1):
    print(f"{i}. {fact}")

# 5. 搜索测试
print("\n" + "=" * 60)
print("搜索测试：")
print("=" * 60)
for query in ["谁持有腾讯", "清仓阿里", "看好比特币", "加仓", "字节", "职级"]:
    qe = client.embeddings.create(input=[query], model="text-embedding-v3").data[0].embedding
    r = collection.query(query_embeddings=[qe], n_results=3)
    print(f"\n🔍 '{query}':")
    for doc, dist in zip(r["documents"][0], r["distances"][0]):
        print(f"   [{dist:.3f}] {doc}")
