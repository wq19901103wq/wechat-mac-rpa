#!/usr/bin/env python3
"""
批量为 global_state.json 中的所有聊天生成/更新 wiki。
用法:
    python3 scripts/batch_generate_wiki.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.engine import MemoryEngine
from src.utils.llm_client import KimiClient


def is_group_chat(chat_name: str) -> bool:
    return chat_name.endswith("@chatroom")


def main():
    print("=" * 50)
    print("批量生成 Wiki")
    print("=" * 50)

    # 加载 global_state
    state_file = Path("data/global_state.json")
    if not state_file.exists():
        print("错误: data/global_state.json 不存在")
        sys.exit(1)

    with open(state_file) as f:
        state = json.load(f)

    print(f"[Store] 共 {len(state)} 个聊天")

    # 初始化 LLM + Engine
    print("[LLM] 初始化 KimiClient...")
    try:
        llm = KimiClient()
    except Exception as e:
        print(f"[LLM] 初始化失败: {e}")
        sys.exit(1)

    engine = MemoryEngine(llm_client=llm)

    # 统计
    total = 0
    success = 0
    failed = 0
    skipped = 0

    # 遍历所有聊天
    for chat_name, chat_data in state.items():
        messages = chat_data.get("messages", [])
        if not messages:
            skipped += 1
            continue

        total += 1
        msg_count = len(messages)

        # 取最近 300 条消息（避免超出 LLM 上下文）
        recent_msgs = messages[-300:]

        # 判断群聊/私聊
        is_group = is_group_chat(chat_name)
        chat_type = "群聊" if is_group else "私聊"

        # 检查是否已有 wiki
        if is_group:
            wiki_path = engine._group_wiki_path(chat_name)
        else:
            wiki_path = engine._user_wiki_path(chat_name)
        has_wiki = wiki_path.exists()

        print(f"\n[{total}] {chat_name} ({chat_type}, {msg_count} 条, wiki={'有' if has_wiki else '无'})")

        try:
            if is_group:
                task = {
                    "type": "group",
                    "group_name": chat_name,
                    "chat_name": chat_name,
                    "messages": recent_msgs,
                    "bot_replies": [],
                    "timestamp": time.time(),
                }
                engine._do_update_group(task)
            else:
                # 私聊：user_name 用聊天名（对方昵称）
                task = {
                    "type": "user",
                    "user_name": chat_name,
                    "chat_name": chat_name,
                    "messages": recent_msgs,
                    "bot_replies": [],
                    "timestamp": time.time(),
                }
                engine._do_update_user(task)
            success += 1
            print(f"    ✅ 成功")
        except Exception as e:
            failed += 1
            print(f"    ❌ 失败: {e}")

    print("\n" + "=" * 50)
    print(f"完成！总计 {total} 个聊天")
    print(f"  ✅ 成功: {success}")
    print(f"  ❌ 失败: {failed}")
    print(f"  ⏭️  跳过(空): {skipped}")
    print("=" * 50)


if __name__ == "__main__":
    main()
