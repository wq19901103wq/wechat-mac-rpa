#!/usr/bin/env python3
"""
批量生成/更新 wiki（v2：适配分片存储，减少消息量防 timeout）。
用法:
    python3 scripts/batch_generate_wiki_v2.py [--limit 100]
"""

import argparse

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.engine import MemoryEngine
from src.session.global_store import GlobalStore
from src.utils.llm_client import KimiClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--msg-limit", type=int, default=100, help="每个聊天取最近 N 条消息")
    parser.add_argument("--max-chats", type=int, default=200, help="最多处理 N 个聊天")
    args = parser.parse_args()

    print("=" * 50)
    print("批量生成 Wiki v2")
    print(f"消息上限: {args.msg_limit} 条/聊天")
    print(f"聊天上限: {args.max_chats} 个")
    print("=" * 50)

    # 加载分片存储
    print("[Store] 加载分片...")
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
        if msgs < 10:
            continue
        priority = (1 if name in has_wiki else 0, msgs)
        chats.append((name, state, priority))

    chats.sort(key=lambda x: (-x[2][0], -x[2][1]))
    chats = chats[:args.max_chats]

    print(f"[Wiki] 将处理 {len(chats)} 个聊天（已有 wiki: {sum(1 for _,_,p in chats if p[0])} 个）\n")

    # 初始化 LLM
    print("[LLM] 初始化 KimiClient...")
    try:
        llm = KimiClient()
    except Exception as e:
        print(f"[LLM] 初始化失败: {e}")
        sys.exit(1)
    engine = MemoryEngine(llm_client=llm)

    success = 0
    failed = 0
    start_all = time.time()

    for idx, (name, state, _) in enumerate(chats, 1):
        msgs = state.messages[-args.msg_limit:]
        is_group = name.endswith("@chatroom")
        chat_type = "群聊" if is_group else "私聊"
        has = "有" if name in has_wiki else "无"

        print(f"\n[{idx}/{len(chats)}] {name} ({chat_type}, {len(state.messages)} 条, wiki={has})")

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
    print(f"  ⏭️  跳过(<10条): {len(store.chats) - len(chats)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
