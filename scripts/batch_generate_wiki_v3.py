#!/usr/bin/env python3
"""
批量生成/更新 wiki（v3：deepseek-v4-flash + 分片存储 + 500 条消息）。
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# 加载 .env
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

from src.memory.engine import MemoryEngine
from src.session.global_store import GlobalStore
from src.utils.qwen_client import QwenClient


def main():
    print("=" * 50)
    print("批量生成 Wiki v3 (deepseek-v4-flash)")
    print("=" * 50)

    store = GlobalStore()
    print(f"[Store] 共 {len(store.chats)} 个聊天")

    # 检查现有 wiki
    wiki_users = {p.stem for p in Path("data/memory/wiki/users").glob("*.md")}
    wiki_groups = {p.stem for p in Path("data/memory/wiki/groups").glob("*.md")}
    has_wiki = wiki_users | wiki_groups

    # 排序：有 wiki 的先更新，然后按消息量降序
    chats = []
    for name, state in store.chats.items():
        msgs = len(state.messages)
        if msgs < 20:
            continue
        priority = (1 if name in has_wiki else 0, msgs)
        chats.append((name, state, priority))

    chats.sort(key=lambda x: (-x[2][0], -x[2][1]))

    print(f"[Wiki] 将处理 {len(chats)} 个聊天（已有 wiki: {sum(1 for _,_,p in chats if p[0])} 个）\n")

    print("[LLM] 初始化 QwenClient (deepseek-v4-flash)...")
    try:
        llm = QwenClient(model="deepseek-v4-flash")
    except Exception as e:
        print(f"[LLM] 初始化失败: {e}")
        sys.exit(1)
    engine = MemoryEngine(llm_client=llm)

    success = 0
    failed = 0
    start_all = time.time()

    for idx, (name, state, _) in enumerate(chats, 1):
        # 根据消息量动态调整取多少条
        msg_count = len(state.messages)
        if msg_count <= 200:
            limit = msg_count  # 小聊天全量
        elif msg_count <= 1000:
            limit = 1000
        else:
            limit = 200  # 大聊天取最近 200

        msgs = state.messages[-limit:]
        is_group = name.endswith("@chatroom")
        chat_type = "群聊" if is_group else "私聊"
        has = "有" if name in has_wiki else "无"

        print(f"\n[{idx}/{len(chats)}] {name} ({chat_type}, {msg_count} 条, wiki={has}, 传入={len(msgs)})")

        try:
            if is_group:
                task = {
                    "type": "group",
                    "group_name": name,
                    "chat_name": name,
                    "messages": msgs,
                    "bot_replies": [],
                    "timestamp": time.time(),
                }
                engine._do_update_group(task)
            else:
                task = {
                    "type": "user",
                    "user_name": name,
                    "chat_name": name,
                    "messages": msgs,
                    "bot_replies": [],
                    "timestamp": time.time(),
                }
                engine._do_update_user(task)
            success += 1
            print(f"    ✅ 成功")
        except Exception as e:
            failed += 1
            print(f"    ❌ 失败: {e}")

        if idx % 20 == 0:
            elapsed = time.time() - start_all
            print(f"  ... 进度 {idx}/{len(chats)}, 已用 {elapsed:.0f}s")

    print("\n" + "=" * 50)
    print(f"完成！总计 {len(chats)} 个聊天")
    print(f"  ✅ 成功: {success}")
    print(f"  ❌ 失败: {failed}")
    print(f"  ⏭️  跳过(<20条): {len(store.chats) - len(chats)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
