#!/usr/bin/env python3
"""
导入另一个微信账号的历史聊天记录到 global_state.json。
不隔离 wiki，只在消息中标记 account 来源，LLM 提炼 wiki 时会自动标注场景。

用法:
    # 1. 确保 WeFlow 已启动，且另一个号已登录微信
    # 2. 运行导入脚本
    python3 scripts/import_account_history.py --account work

    # 3. 导入完成后，Bot 启动时会自动加载，wiki 更新时 LLM 会自动标注 [work]

参数:
    --account      账号标识，如 work / personal / b 等（建议简短英文）
    --weflow-port  WeFlow HTTP API 端口，默认 5031
    --max-rounds   分页拉取最大轮数，默认 1000
    --dry-run      只列出将要导入的联系人/群聊，不实际写入
"""

import argparse

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base import ChatMessage, SenderType
from src.session.global_store import GlobalStore
from src.perception.weflow_client import WeFlowClient


def fetch_all_history(weflow: WeFlowClient, account: str, start_index: int = 0, end_index: int = 9999, max_msgs_per_chat: int = 50000):
    """从 WeFlow 拉取当前登录微信的全部历史，返回 {chat_name: [ChatMessage]}。"""
    print(f"[WeFlow] 加载联系人列表...")
    try:
        contacts = weflow.get_contacts()
    except Exception as e:
        print(f"[WeFlow] 获取联系人失败: {e}")
        return {}

    if not contacts:
        print("[WeFlow] 联系人列表为空，请确认微信已登录")
        return {}

    print(f"[WeFlow] 共 {len(contacts)} 个联系人/群聊")

    result = {}
    total_msgs = 0

    contacts = contacts[start_index:end_index]
    print(f"[WeFlow] 处理联系人范围 [{start_index}:{end_index}]，共 {len(contacts)} 个")
    for idx, contact in enumerate(contacts, start_index + 1):
        talker = contact.username
        name = contact.nickname or contact.display_name or talker
        msgs = []
        offset = 0
        max_rounds = 1000
        limit = 100000000

        for _ in range(max_rounds):
            try:
                batch, has_more = weflow.get_messages(talker, limit=limit, offset=offset)
                if not batch:
                    break
                if len(msgs) >= max_msgs_per_chat:
                    print(f"    达到上限 {max_msgs_per_chat} 条，截断")
                    has_more = False
                    break
                for m in batch:
                    # 转换 WeFlowMessage → ChatMessage，标记 account
                    sender = m.sender_username or ""
                    sender_type = SenderType.SELF if m.is_send else SenderType.OTHER
                    # 群聊中 sender 为空时标记为"对方"
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
                offset += len(batch)
            except Exception as e:
                print(f"  [{name}] 拉取失败: {e}")
                break

        if msgs:
            result[name] = msgs
            total_msgs += len(msgs)
            print(f"  [{idx}/{len(contacts)}] {name}: {len(msgs)} 条消息")

    print(f"\n[WeFlow] 共拉取 {len(result)} 个聊天，{total_msgs} 条消息")
    return result


def merge_into_global_store(store: GlobalStore, new_chats: dict, account: str, dry_run: bool = False):
    """把新账号的历史合并到 global_store，同名聊天合并消息。"""
    if dry_run:
        print(f"\n[Dry Run] 将要合并的聊天:")
        for name, msgs in new_chats.items():
            existing = store.chats.get(name)
            if existing:
                print(f"  {name}: 现有 {len(existing.messages)} 条 + 新增 {len(msgs)} 条（account={account}）")
            else:
                print(f"  {name}: 新建，{len(msgs)} 条（account={account}）")
        return

    print(f"\n[Merge] 开始合并到 global_state.json...")
    for name, msgs in new_chats.items():
        if name in store.chats:
            existing = store.chats[name]
            before = len(existing.messages)
            # 直接追加（依赖 global_store 的去重机制处理重复）
            existing.messages.extend(msgs)
            # 重新计算 msg_ids
            from src.session.global_store import _msg_id
            for m in msgs:
                existing._msg_ids.add(_msg_id(name, m))
            print(f"  {name}: {before} → {len(existing.messages)} 条")
        else:
            from src.session.global_store import ChatState
            state = ChatState(chat_id=name, chat_name=name)
            state.messages = msgs
            from src.session.global_store import _msg_id
            for m in msgs:
                state._msg_ids.add(_msg_id(name, m))
            store.chats[name] = state
            print(f"  {name}: 新建 {len(msgs)} 条")

    store.save()
    print(f"[Merge] 已保存到 {store._state_file}")


def main():
    parser = argparse.ArgumentParser(description="导入另一个微信账号的历史聊天记录")
    parser.add_argument("--account", type=str, required=True, help="账号标识，如 work / personal / b")
    parser.add_argument("--weflow-port", type=int, default=5031, help="WeFlow HTTP API 端口，默认 5031")
    parser.add_argument("--dry-run", action="store_true", help="只列出将要导入的聊天，不实际写入")
    parser.add_argument("--start-index", type=int, default=0, help="起始联系人索引（0-based），用于分批导入")
    parser.add_argument("--end-index", type=int, default=9999, help="结束联系人索引（不包含）")
    parser.add_argument("--max-msgs-per-chat", type=int, default=30000, help="单个聊天最大导入消息数，默认50000（防超大群超时）")
    args = parser.parse_args()

    account = args.account.strip()
    if not account:
        print("错误: --account 不能为空")
        sys.exit(1)

    print(f"=" * 50)
    print(f"账号导入: account='{account}'")
    print(f"WeFlow API: http://127.0.0.1:{args.weflow_port}")
    print(f"=" * 50)

    # 初始化 WeFlow 客户端
    weflow = WeFlowClient(host="127.0.0.1", port=args.weflow_port, timeout=300)
    try:
        ok = weflow.health_check()
        if ok:
            print(f"[WeFlow] 已连接\n")
        else:
            print(f"[WeFlow] 健康检查失败")
            sys.exit(1)
    except Exception as e:
        print(f"[WeFlow] 连接失败: {e}")
        print("请确保 WeFlow 已启动（port {}）且微信已登录".format(args.weflow_port))
        sys.exit(1)

    # 加载现有 global_store
    store = GlobalStore()
    print(f"[Store] 现有 {len(store.chats)} 个聊天\n")

    # 拉取新账号历史
    new_chats = fetch_all_history(weflow, account, start_index=args.start_index, end_index=args.end_index, max_msgs_per_chat=args.max_msgs_per_chat)
    if not new_chats:
        print("没有拉取到任何消息，退出")
        sys.exit(1)

    # 合并
    merge_into_global_store(store, new_chats, account, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"\n✅ 导入完成！")
        print(f"   下次 Bot 启动时会自动加载这些消息")
        print(f"   当 Bot 回复相关聊天时，MemoryEngine 会自动更新 wiki")
        print(f"   wiki 中会标注 [account] 来源")
    else:
        print(f"\n[Dry Run] 未实际写入，去掉 --dry-run 后执行")


if __name__ == "__main__":
    main()
