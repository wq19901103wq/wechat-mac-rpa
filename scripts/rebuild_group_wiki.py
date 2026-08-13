#!/usr/bin/env python3
"""
批量重建群聊 wiki。
不启动 Bot，直接读取 global_state.json，调用 LLM 为每个群聊生成 wiki。

用法:
    python3 scripts/rebuild_group_wiki.py         # 重建所有群聊
    python3 scripts/rebuild_group_wiki.py --dry-run  # 只列出群聊，不调用 LLM
    python3 scripts/rebuild_group_wiki.py --group "群名"  # 只重建指定群聊
"""

import argparse
import json

import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.engine import _DEFAULT_GROUP_WIKI, _UPDATE_GROUP_PROMPT
from src.models.base import ChatMessage, SenderType
from src.utils.llm_client import KimiClient


def is_group_chat(name: str, msgs: list) -> bool:
    """判断是否为群聊：名字包含 chatroom 或最近 100 条消息中有 2+ 个不同 sender。"""
    if "chatroom" in name.lower():
        return True
    senders = set(
        m.get("sender", "")
        for m in msgs[-100:]
        if m.get("sender") and m.get("sender_type") != "self"
    )
    return len(senders) >= 2


def load_global_state(path: str = "data/global_state.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_chat_messages(raw_msgs: list, chat_name: str) -> list:
    """把 JSON dict 列表转为 ChatMessage 对象列表。"""
    result = []
    for m in raw_msgs:
        try:
            msg = ChatMessage(
                text=m.get("text", ""),
                sender=m.get("sender", ""),
                sender_type=SenderType(m.get("sender_type", "other")),
                chat_name=m.get("chat_name", chat_name),
                is_at_me=m.get("is_at_me", False),
                replied=m.get("replied", False),
                reply_text=m.get("reply_text", ""),
                reply_time=m.get("reply_time"),
                message_type=m.get("message_type", "text"),
                image_description=m.get("image_description", ""),
                image_text=m.get("image_text", ""),
            )
            result.append(msg)
        except Exception:
            continue
    return result


def format_conversation(messages: list, bot_replies: list) -> str:
    """格式化对话为文本（兼容 ChatMessage 和 dict）。"""
    lines = []
    for msg in messages:
        st = getattr(msg, "sender_type", None)
        is_self = st and (
            st.value == "self" if hasattr(st, "value") else str(st) == "self"
        )
        sender = "我" if is_self else getattr(msg, "sender", "")
        text = getattr(msg, "text", "")
        if text:
            lines.append(f"{sender}：{text}")
    for reply in bot_replies:
        lines.append(f"Bot：{reply}")
    return "\n".join(lines)


def update_group_wiki_direct(
    llm_client,
    group_name: str,
    messages: list,
    bot_replies: list = None,
) -> str:
    """
    直接调用 LLM 生成/更新群聊 wiki，返回新 wiki 内容。
    不依赖 MemoryEngine 的队列，适合批量重建。
    """
    if bot_replies is None:
        bot_replies = []

    wiki_dir = Path("data/memory/wiki/groups")
    wiki_dir.mkdir(parents=True, exist_ok=True)
    wiki_path = wiki_dir / f"{group_name}.md"

    current_wiki = (
        wiki_path.read_text(encoding="utf-8")
        if wiki_path.exists()
        else _DEFAULT_GROUP_WIKI.format(group_name=group_name)
    )

    conversation = format_conversation(messages, bot_replies)
    if not conversation.strip():
        print(f"  [{group_name}] 无有效对话，跳过")
        return current_wiki

    now = time.strftime("%Y-%m-%d %H:%M")
    prompt = _UPDATE_GROUP_PROMPT.format(
        current_wiki=current_wiki,
        chat_name=group_name,
        current_time=now,
        conversation=conversation,
        identity_context="",
    )

    try:
        response = llm_client.client.chat.completions.create(
            model=llm_client.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=10000,
            timeout=120,
        )
        new_wiki = response.choices[0].message.content
        new_wiki = new_wiki.strip()
        if new_wiki and len(new_wiki) > 50:
            wiki_path.write_text(new_wiki, encoding="utf-8")
            print(f"  ✓ 已保存: {wiki_path}")
            return new_wiki
        else:
            print(f"  ✗ LLM 返回内容过短，未保存")
            return current_wiki
    except Exception as e:
        print(f"  ✗ LLM 调用失败: {e}")
        return current_wiki


def main():
    parser = argparse.ArgumentParser(description="批量重建群聊 wiki")
    parser.add_argument("--dry-run", action="store_true", help="只列出群聊，不调用 LLM")
    parser.add_argument("--group", type=str, default=None, help="只重建指定群聊")
    parser.add_argument("--max-msgs", type=int, default=200, help="每次取最近 N 条消息 (默认 200)")
    parser.add_argument("--state", type=str, default="data/global_state.json", help="global_state.json 路径")
    args = parser.parse_args()

    state = load_global_state(args.state)
    print(f"已加载 global_state.json，共 {len(state)} 个聊天\n")

    # 识别群聊
    groups = []
    for name, info in state.items():
        msgs = info.get("messages", [])
        if not msgs:
            continue
        if is_group_chat(name, msgs):
            groups.append((name, msgs))

    if not groups:
        print("未找到群聊")
        return

    print(f"发现 {len(groups)} 个群聊:")
    for name, msgs in groups:
        print(f"  {name}: {len(msgs)} 条消息")

    if args.dry_run:
        return

    # 如果指定了群名，只处理该群聊
    if args.group:
        groups = [(n, m) for n, m in groups if n == args.group]
        if not groups:
            print(f"未找到群聊: {args.group}")
            return

    # 初始化 LLM 客户端
    print("\n初始化 LLM 客户端...")
    llm_client = KimiClient()

    # 逐个群聊重建
    for name, msgs in groups:
        print(f"\n▶ 重建群聊 wiki: {name} ({len(msgs)} 条消息)")

        # 分批处理：如果消息很多，分多轮从旧到新喂给 LLM，让 wiki 逐步完善
        batch_size = args.max_msgs
        if len(msgs) <= batch_size:
            batches = [msgs]
        else:
            # 从旧到新分批次
            batches = []
            for i in range(0, len(msgs), batch_size):
                batches.append(msgs[i : i + batch_size])

        for idx, batch in enumerate(batches):
            chat_msgs = to_chat_messages(batch, name)
            print(f"  批次 {idx + 1}/{len(batches)}: {len(chat_msgs)} 条消息")
            update_group_wiki_direct(llm_client, name, chat_msgs)
            if idx < len(batches) - 1:
                time.sleep(2)  # 避免 API 限流

    print("\n✅ 全部群聊 wiki 重建完成")


if __name__ == "__main__":
    main()
