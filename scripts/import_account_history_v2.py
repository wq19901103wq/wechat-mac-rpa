#!/usr/bin/env python3
"""
并发版本：导入另一个微信账号的历史聊天记录到 global_state.json。
用法:
    python3 scripts/import_account_history_v2.py --account b --workers 10
"""

import argparse

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base import ChatMessage, SenderType
from src.session.global_store import GlobalStore, _msg_id
from src.perception.weflow_client import WeFlowClient


def fetch_one_contact(weflow: WeFlowClient, contact, account: str, max_msgs: int):
    """拉取单个联系人的全部消息，返回 (name, msgs) 或 None。"""
    talker = contact.username
    name = contact.nickname or contact.display_name or talker
    msgs = []
    offset = 0
    limit = 5000  # 小页分批，避免 offset 性能问题

    for _ in range(1000):  # 最多 1000 页
        try:
            batch, has_more = weflow.get_messages(talker, limit=limit, offset=offset)
            if not batch:
                break
            for m in batch:
                sender = m.sender_username or ""
                sender_type = SenderType.SELF if m.is_send else SenderType.OTHER
                if not sender and "@chatroom" in talker:
                    sender = "对方"
                msg = ChatMessage(
                    text=m.content or "",
                    sender=sender,
                    sender_type=sender_type,
                    chat_name=name,
                    is_at_me=False,
                    replied=False,
                    reply_text="",
                    reply_time=None,
                    message_type="text",
                    image_description="",
                    image_text="",
                    account=account,
                    local_id=m.local_id,
                    server_id=m.server_id,
                    create_time=m.create_time,
                    raw_type=m.local_type,
                    sender_wxid=m.sender_username,
                )
                msgs.append(msg)
            if not has_more:
                break
            if len(msgs) >= max_msgs:
                break
            offset += len(batch)
        except Exception as e:
            return (name, None, f"拉取失败: {e}")

    if not msgs:
        return (name, None, "无消息")
    return (name, msgs, f"{len(msgs)} 条")


def merge_into_global_store(store: GlobalStore, new_chats: dict, account: str):
    """把新账号的历史合并到 global_store，同名聊天合并消息。"""
    print(f"\n[Merge] 开始合并到 global_state.json...")
    for name, msgs in new_chats.items():
        if name in store.chats:
            existing = store.chats[name]
            before = len(existing.messages)
            existing.messages.extend(msgs)
            for m in msgs:
                existing._msg_ids.add(_msg_id(name, m))
            print(f"  {name}: {before} → {len(existing.messages)} 条")
        else:
            from src.session.global_store import ChatState
            state = ChatState(chat_id=name, chat_name=name)
            state.messages = msgs
            for m in msgs:
                state._msg_ids.add(_msg_id(name, m))
            store.chats[name] = state
            print(f"  {name}: 新建 {len(msgs)} 条")
    store.save()
    print(f"[Merge] 已保存到 {store._state_file}")


def main():
    parser = argparse.ArgumentParser(description="并发导入另一个微信账号的历史聊天记录")
    parser.add_argument("--account", type=str, required=True, help="账号标识，如 b")
    parser.add_argument("--weflow-port", type=int, default=5031, help="WeFlow HTTP API 端口")
    parser.add_argument("--workers", type=int, default=10, help="并发线程数，默认10")
    parser.add_argument("--max-msgs-per-chat", type=int, default=0, help="单聊天上限，0=无限制")
    parser.add_argument("--dry-run", action="store_true", help="只列出联系人，不拉取")
    args = parser.parse_args()

    account = args.account.strip()
    max_msgs = args.max_msgs_per_chat if args.max_msgs_per_chat > 0 else 999999999

    print(f"=" * 50)
    print(f"并发账号导入: account='{account}'")
    print(f"WeFlow API: http://127.0.0.1:{args.weflow_port}")
    print(f"并发 workers: {args.workers}")
    print(f"=" * 50)

    weflow = WeFlowClient(host="127.0.0.1", port=args.weflow_port, timeout=300)
    try:
        if not weflow.health_check():
            print("[WeFlow] 健康检查失败")
            sys.exit(1)
    except Exception as e:
        print(f"[WeFlow] 连接失败: {e}")
        sys.exit(1)
    print("[WeFlow] 已连接\n")

    # 加载联系人
    print("[WeFlow] 加载联系人列表...")
    contacts = weflow.get_contacts()
    print(f"[WeFlow] 共 {len(contacts)} 个联系人/群聊")

    if args.dry_run:
        print("\n[Dry Run] 联系人列表:")
        for c in contacts:
            print(f"  [{c.type:8s}] {c.name[:40]:40s}")
        return

    # 加载现有 global_store
    store = GlobalStore()
    print(f"[Store] 现有 {len(store.chats)} 个聊天\n")

    # 并发拉取
    print(f"[Fetch] 开始并发拉取消息 (workers={args.workers})...")
    start_time = time.time()
    new_chats = {}
    total_msgs = 0
    skipped = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_one_contact, weflow, c, account, max_msgs): c
            for c in contacts
        }
        for future in as_completed(futures):
            contact = futures[future]
            try:
                name, msgs, status = future.result()
                if msgs is None:
                    skipped += 1
                    if "无消息" not in status:
                        errors += 1
                        print(f"  [{contact.name}] {status}")
                else:
                    new_chats[name] = msgs
                    total_msgs += len(msgs)
                    if len(msgs) > 100:
                        print(f"  {name}: {len(msgs)} 条")
            except Exception as e:
                errors += 1
                print(f"  [{contact.name}] 异常: {e}")

    elapsed = time.time() - start_time
    print(f"\n[Fetch] 完成！{len(new_chats)} 个聊天有消息，共 {total_msgs} 条")
    print(f"        跳过 {skipped} 个，错误 {errors} 个，耗时 {elapsed:.1f}s")

    if not new_chats:
        print("没有拉取到任何消息，退出")
        sys.exit(1)

    # 合并
    merge_into_global_store(store, new_chats, account)
    print(f"\n✅ 导入完成！")


if __name__ == "__main__":
    main()
