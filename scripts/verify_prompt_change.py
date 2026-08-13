#!/usr/bin/env python3
"""验证 persona.md 修改后的回复效果 — 带工具调用"""

import json, os, sys, sqlite3
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from src.utils.qwen_client import QwenClient
from src.tools.tool_registry import get_registry
from src.tools.builtin_tools import register_builtin_tools
from src.memory import MemoryEngine

_logger = logging.getLogger(__name__)

llm = QwenClient()

# 注册工具（与线上一致）
registry = get_registry()
register_builtin_tools()
mem = MemoryEngine()
def _search_memory(query: str = "") -> str:
    return mem.search_keyword(query)
registry.register(
    name="search_memory",
    description="搜索本地长期记忆。当你不确定某个人是谁、某件事的背景、或者某个关系时，调用此工具查询本地 wiki 记忆库。",
    parameters={"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"]},
    func=_search_memory,
)
tools = registry.to_openai_schemas()

# 加载当前 persona.md
persona_path = PROJECT_ROOT / "data" / "persona.md"
with open(persona_path, encoding="utf-8") as f:
    system_prompt = f.read()

# 替换模板变量
tools_desc = "\n".join(f"- {t.name}：{t.description}" for t in registry._tools.values())
system_prompt = system_prompt.replace("{tools_description}", tools_desc)
system_prompt = system_prompt.replace("{dynamic_few_shot}", "（无相关历史对话）")

# 重点验证有幻觉/猜错/啰嗦问题的 case
TICK_IDS = [
    1780132842,   # 同义反复：我是真人（你关注的）
    1780120216,   # 确认堆叠：《后来》（改善最好的）
    1780119222,   # 人物幻觉：二姨→大姨
    1780120180,   # 猜错歌名：光亮→起风了
    1780120188,   # 信息缺失：没给正确答案
    1780120224,   # 正常case：拉窗帘
]

conn = sqlite3.connect(PROJECT_ROOT / "data" / "cases.db")
cursor = conn.cursor()
cursor.execute('''
    SELECT tick_id, chat_name, user_prompt, replies_sent_json, judge_score, judge_is_badcase
    FROM tick_log 
    WHERE tick_id IN ({placeholders})
'''.format(placeholders=','.join('?' * len(TICK_IDS))), tuple(TICK_IDS))
cases = {row[0]: row for row in cursor.fetchall()}
conn.close()

print("=" * 80)
print(f"System prompt: {len(system_prompt)} 字符 | 带工具调用")
print(f"验证 {len(TICK_IDS)} 条 case")
print("=" * 80)

for tick_id in TICK_IDS:
    row = cases.get(tick_id)
    if not row:
        print(f"\n⚠️ tick {tick_id} 未找到")
        continue
    
    tick_id, chat_name, user_prompt, old_replies_json, old_score, old_bad = row
    old_replies = json.loads(old_replies_json) if old_replies_json else []
    
    print(f"\n{'─' * 80}")
    print(f"【tick {tick_id}】Chat: {chat_name} | 旧score={old_score} | 旧bad={old_bad}")
    print(f">>> 旧回复 ({len(old_replies)}条): {old_replies}")
    print(f">>> 新回复:")
    
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    try:
        # 支持工具调用的多轮对话
        tool_log = []
        final_reply = ""
        max_rounds = 3
        for round_i in range(max_rounds):
            raw = llm.chat(messages=msgs, tools=tools, max_tokens=500, timeout=60)
            
            # 检查是否返回了 tool_calls
            if hasattr(raw, "tool_calls") and raw.tool_calls:
                msgs.append({"role": "assistant", "content": raw.content or "",
                             "tool_calls": [{"id": tc.id, "type": tc.type, 
                                             "function": {"name": tc.function.name, "arguments": tc.function.arguments}} 
                                            for tc in raw.tool_calls]})
                for tc in raw.tool_calls:
                    name = tc.function.name
                    args_str = tc.function.arguments
                    if registry.has(name):
                        result = registry.get(name).execute(args_str)
                    else:
                        result = f"工具 {name} 不存在"
                    tool_log.append({"name": name, "args": args_str, "result": str(result)[:300]})
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                continue
            else:
                final_reply = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
                break
        
        if not final_reply:
            final_reply = "(空)"
        
        # 显示工具调用
        if tool_log:
            print(f"   [工具调用] {len(tool_log)} 次:")
            for t in tool_log:
                print(f"      → {t['name']}({t['args'][:80]}) = {t['result'][:100]}")
        
        # 解析 JSON
        if final_reply.strip().startswith("{"):
            try:
                data = json.loads(final_reply.strip())
                new_replies = data.get("replies", [])
            except Exception as e:
                _logger.warning("parse final reply failed: %s", e)
                new_replies = [final_reply.strip()]
        else:
            new_replies = [final_reply.strip()]
        
        for i, r in enumerate(new_replies, 1):
            print(f"   [{i}] {r}")
        
    except Exception as e:
        print(f"   ❌ 生成失败: {e}")

print(f"\n{'=' * 80}")
print("验证完成")
