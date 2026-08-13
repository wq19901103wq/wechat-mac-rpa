#!/usr/bin/env python3
"""
从 data/chats/*.json 批量生成正确的 LLM Wiki。

解决以下问题：
1. 基于结构字段区分私聊、群聊和系统消息
2. 时间戳缺失（利用 engine.py 新增的时间戳支持）
3. 同一用户跨聊天信息聚合（按 sender_wxid 聚合）
4. @chatroom 名称映射（支持 chatroom_names.json）

用法:
    python3 scripts/bulk_import_from_chats.py [--dry-run] [--users-only] [--groups-only]

参数:
    --dry-run       只统计和分类，不调用 LLM
    --users-only    只生成用户 wiki
    --groups-only   只生成群聊 wiki
    --min-user-msgs 用户消息量阈值，默认 20
    --min-group-msgs 群聊消息量阈值，默认 30
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.engine import MemoryEngine
from src.utils.qwen_client import QwenClient

_logger = logging.getLogger(__name__)


CHATS_DIR = Path("data/chats")
WIKI_DIR = Path("data/memory/wiki")
UNMAPPED_FILE = CHATS_DIR / "_unmapped_chatrooms.json"
CHATROOM_NAMES_FILE = CHATS_DIR / "chatroom_names.json"


# ── 简单消息包装（兼容 engine._format_conversation） ──

@dataclass
class SimpleMsg:
    sender: str
    sender_type: str  # "self" 或 "other"
    text: str
    create_time: int | None = None
    account: str = ""
    chat_name: str = ""  # 消息来源的聊天名称，用于 conversation 中分隔不同聊天


# ── 群聊名称映射 ──

def load_chatroom_names() -> dict:
    """加载 @chatroom 显示名称映射。"""
    if CHATROOM_NAMES_FILE.exists():
        with open(CHATROOM_NAMES_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def resolve_chat_name(stem: str, data: dict, mapping: dict) -> str:
    """解析群聊/私聊的显示名称。"""
    chat_name = data.get("chat_name", "") or stem
    # 如果是纯数字@chatroom，尝试映射
    if "@chatroom" in chat_name and chat_name.replace("@chatroom", "").isdigit():
        mapped = mapping.get(stem) or mapping.get(chat_name)
        if mapped:
            return mapped
    return chat_name


# ── 群聊/私聊分类 ──

def is_system_message(msg: dict) -> bool:
    """按持久化的发送者类型判断系统消息。"""
    return msg.get("sender_type") == "system"


def classify_chat(stem: str, data: dict) -> str:
    """正确分类群聊/私聊。返回 'group' 或 'private'。"""
    chat_name = data.get("chat_name", "") or stem
    msgs = data.get("messages", [])

    # 强信号1: 文件名或名称包含 @chatroom
    if "@chatroom" in stem.lower() or "@chatroom" in chat_name.lower():
        return "group"

    # 分析消息中的真实 sender（排除 self、系统消息、无 wxid）
    wxid_set = set()
    for m in msgs:
        if m.get("sender_type") == "self":
            continue
        wxid = m.get("sender_wxid", "")
        if not wxid:
            continue
        if is_system_message(m):
            continue
        wxid_set.add(wxid)

    # 强信号2: 2+ 个不同 wxid（真实多人在聊天）
    if len(wxid_set) >= 2:
        return "group"

    return "private"


# ── 全局用户索引 ──

def build_wxid_index(all_chats: dict) -> dict:
    """
    建立全局 wxid → 用户信息索引。
    返回: {wxid: {main_name, all_names, total_msgs, chats: {stem: [msgs]}}}
    """
    raw: dict[str, dict] = defaultdict(
        lambda: {"names": defaultdict(int), "chats": defaultdict(list)}
    )

    for stem, data in all_chats.items():
        for m in data.get("messages", []):
            if m.get("sender_type") == "self":
                continue
            wxid = m.get("sender_wxid", "")
            sender = m.get("sender", "")
            if not wxid or not sender or sender in ("对方", "[未知]", ""):
                continue
            if is_system_message(m):
                continue
            if wxid.endswith("@chatroom"):
                continue  # 排除群聊系统消息（以群聊名称作为 sender 的情况）

            raw[wxid]["names"][sender] += 1
            raw[wxid]["chats"][stem].append(m)

    # 为每个 wxid 确定主昵称
    result = {}
    for wxid, info in raw.items():
        if not info["names"]:
            continue
        main_name = max(info["names"].items(), key=lambda x: x[1])[0]
        total = sum(info["names"].values())
        chat_count = len(info["chats"])
        result[wxid] = {
            "main_name": main_name,
            "all_names": dict(info["names"]),
            "total_msgs": total,
            "chat_count": chat_count,
            "chats": dict(info["chats"]),
        }
    return result


# ── 别名管理 ──

def load_aliases() -> tuple[dict, dict]:
    """加载现有 aliases.json，返回 {昵称: 主名} 反向映射。"""
    aliases_path = Path("data/memory/overrides/aliases.json")
    existing = {}
    name_to_main = {}
    if aliases_path.exists():
        try:
            with open(aliases_path, encoding="utf-8") as f:
                data = json.load(f)
            existing = data.get("users", {})
            for main_name, cfg in existing.items():
                name_to_main[main_name] = main_name
                for alias in cfg.get("aliases", []):
                    name_to_main[alias] = main_name
        except Exception as e:
            _logger.warning("load alias config failed: %s", e)
    return existing, name_to_main


def resolve_main_name(wxid: str, info: dict, name_to_main: dict, existing_wikis: set) -> str:
    """为 wxid 确定稳定的主昵称。
    优先级：
    1. 现有 aliases.json 中已记录的 → 保持原主名
    2. 已有 wiki 文件匹配该 wxid 的某个昵称 → 使用已有 wiki 名
    3. 消息量最多的昵称
    """
    all_names = list(info["all_names"].keys())

    # 优先级1：aliases.json 中已映射
    for name in all_names:
        if name in name_to_main:
            return name_to_main[name]

    # 优先级2：已有 wiki 文件
    for name in all_names:
        if name in existing_wikis:
            return name

    # 优先级3：消息量最多
    return info["main_name"]


def update_aliases_json(wxid_index: dict, name_to_main: dict, dry_run: bool = False):
    """自动发现新别名并更新 aliases.json。保留原有 notes。使用 unique_name 作为键。"""
    aliases_path = Path("data/memory/overrides/aliases.json")
    existing = {}
    if aliases_path.exists():
        try:
            with open(aliases_path, encoding="utf-8") as f:
                existing = json.load(f).get("users", {})
        except Exception as e:
            _logger.warning("load aliases failed: %s", e)

    added = 0
    for wxid, info in wxid_index.items():
        main_name = info.get("unique_name", info.get("resolved_name", info["main_name"]))
        all_names = info["all_names"]

        # 收集该用户的新别名（出现次数≥3，且不在现有 aliases 中，且不归属他人）
        new_aliases = []
        for name, count in sorted(all_names.items(), key=lambda x: -x[1]):
            if name == main_name or name == info.get("resolved_name", info["main_name"]):
                continue
            if count < 3:
                continue
            if main_name in existing and name in existing[main_name].get("aliases", []):
                continue
            # 跨人冲突：若该名字已映射到别的用户，跳过
            owner = name_to_main.get(name)
            if owner and owner != main_name:
                _logger.warning(
                    f"bulk_import: 别名『{name}』已属于『{owner}』，"
                    f"跳过分配给『{main_name}』"
                )
                continue
            new_aliases.append(name)

        if new_aliases:
            if main_name not in existing:
                existing[main_name] = {"aliases": [], "notes": ""}
            for name in new_aliases:
                if name not in existing[main_name]["aliases"]:
                    existing[main_name]["aliases"].append(name)
                    added += 1

    if not dry_run:
        aliases_path.parent.mkdir(parents=True, exist_ok=True)
        aliases_path.write_text(
            json.dumps({"users": existing}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return added


# ── Token 估算与分轮次 ──

def estimate_tokens(text: str) -> int:
    """粗略估算中文字符对应的 token 数。中文≈1.5 tokens/字，英文≈0.5 tokens/字，+4 格式开销。"""
    if not text:
        return 0
    cn = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other = len(text) - cn
    return int(cn * 1.5 + other * 0.5) + 4


def split_by_tokens(msgs: list[dict], max_tokens: int = 900_000) -> list[list[dict]]:
    """按 token 估算把消息列表分成多批，每批不超过 max_tokens。保留时间顺序。"""
    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_tokens = 0
    for m in msgs:
        tokens = estimate_tokens(m.get("text", ""))
        if current_tokens + tokens > max_tokens and current_batch:
            batches.append(current_batch)
            current_batch, current_tokens = [], 0
        current_batch.append(m)
        current_tokens += tokens
    if current_batch:
        batches.append(current_batch)
    return batches


# ── 消息格式化 ──

def make_simple_messages(raw_msgs: list, max_msgs: int = 500, chat_name: str = "") -> list:
    """把原始消息列表转为 SimpleMsg 对象列表，按时间排序。"""
    msgs = sorted(raw_msgs, key=lambda m: m.get("create_time") or 0)
    msgs = msgs[-max_msgs:]

    result = []
    for m in msgs:
        st = m.get("sender_type", "other")
        sender = m.get("sender", "")
        text = m.get("text", "")
        if not text or not text.strip():
            continue
        result.append(SimpleMsg(
            sender=sender,
            sender_type=st,
            text=text,
            create_time=m.get("create_time"),
            account=m.get("account", ""),
            chat_name=chat_name,
        ))
    return result


# ── 主流程 ──

def _load_env() -> None:
    env_file = Path(__file__).parent.parent / ".env"
    if not env_file.exists():
        return
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def main():
    _load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只统计，不调用 LLM")
    parser.add_argument("--users-only", action="store_true", help="只生成用户 wiki")
    parser.add_argument("--groups-only", action="store_true", help="只生成群聊 wiki")
    parser.add_argument("--min-user-msgs", type=int, default=20, help="用户消息量阈值")
    parser.add_argument("--min-group-msgs", type=int, default=30, help="群聊消息量阈值")
    parser.add_argument("--max-msgs-per-chat", type=int, default=100000, help="每个聊天最多取多少条（安全上限，实际由token估算控制）")
    parser.add_argument("--max-total-msgs", type=int, default=100000, help="单个 wiki 更新最多传多少条消息（安全上限，实际由token估算控制）")
    parser.add_argument("--workers", type=int, default=3, help="并发 worker 数")
    parser.add_argument("--filter-user", dest="filter_user", type=str, default="", help="只处理指定用户（匹配 wxid 或主名）")
    args = parser.parse_args()

    print("=" * 60)
    print("从 data/chats/*.json 批量生成 Wiki（修复版）")
    if args.dry_run:
        print("[DRY RUN] 不调用 LLM")
    print("=" * 60)

    # 1. 加载所有聊天
    chat_files = sorted(CHATS_DIR.glob("*.json"))
    chat_files = [f for f in chat_files if not f.name.startswith("_") and f.name not in ("chatroom_names.json",)]

    all_chats = {}
    for f in chat_files:
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            all_chats[f.stem] = data
        except Exception as e:
            print(f"  ⚠️ 跳过 {f.name}: {e}")

    print(f"\n[加载] 共 {len(all_chats)} 个聊天文件")

    # 2. 加载群聊名称映射
    chatroom_names = load_chatroom_names()
    if chatroom_names:
        print(f"[映射] 加载了 {len(chatroom_names)} 个 @chatroom 名称映射")

    # 3. 分类
    groups = {}
    privates = {}
    unmapped_chatrooms = []

    for stem, data in all_chats.items():
        chat_type = classify_chat(stem, data)
        if chat_type == "group":
            groups[stem] = data
        else:
            privates[stem] = data

    print(f"[分类] 群聊: {len(groups)}, 私聊: {len(privates)}")

    # 检查 @chatroom 映射情况
    for stem, data in groups.items():
        if "@chatroom" in stem:
            chat_name = data.get("chat_name", stem)
            if "@chatroom" in chat_name and chat_name.replace("@chatroom", "").isdigit():
                mapped = chatroom_names.get(stem) or chatroom_names.get(chat_name)
                if not mapped:
                    unmapped_chatrooms.append({
                        "stem": stem,
                        "chat_name": chat_name,
                        "msg_count": len(data.get("messages", [])),
                    })

    if unmapped_chatrooms:
        UNMAPPED_FILE.write_text(
            json.dumps(unmapped_chatrooms, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"⚠️ {len(unmapped_chatrooms)} 个 @chatroom 缺少显示名称")
        print(f"   已保存: {UNMAPPED_FILE}")
        print(f"   请在 {CHATROOM_NAMES_FILE} 中添加映射，例如:")
        print('   {"room_example@chatroom": "某群显示名"}')

    # 4. 建立全局用户索引 + 别名解析
    print("\n[索引] 建立全局用户索引...")
    wxid_index = build_wxid_index(all_chats)
    print(f"  发现 {len(wxid_index)} 个唯一用户")

    # 加载现有别名和 wiki 文件，确定稳定主名
    existing_aliases, name_to_main = load_aliases()
    existing_wikis = {p.stem for p in (WIKI_DIR / "users").glob("*.md")}

    resolved_count = 0
    alias_migrated = 0
    for wxid, info in wxid_index.items():
        resolved = resolve_main_name(wxid, info, name_to_main, existing_wikis)
        if resolved != info["main_name"]:
            alias_migrated += 1
        info["resolved_name"] = resolved
        resolved_count += 1

    # 为同一主名对应多个 wxid 的情况分配唯一文件名
    name_counts = {}
    for wxid, info in wxid_index.items():
        name = info.get("resolved_name", info["main_name"])
        name_counts[name] = name_counts.get(name, 0) + 1

    used_names = set()
    conflict_count = 0
    for wxid, info in wxid_index.items():
        name = info.get("resolved_name", info["main_name"])
        if name_counts[name] > 1:
            suffix = 2
            unique_name = name
            while unique_name in used_names:
                unique_name = f"{name}_{suffix}"
                suffix += 1
            if unique_name != name:
                conflict_count += 1
        else:
            unique_name = name
        used_names.add(unique_name)
        info["unique_name"] = unique_name

    # 自动发现新别名并更新 aliases.json（使用 unique_name 作为键）
    added_aliases = update_aliases_json(wxid_index, name_to_main, dry_run=args.dry_run)
    print(f"  别名解析: {resolved_count} 个用户, {alias_migrated} 个主名迁移, {conflict_count} 个冲突改名, 新增 {added_aliases} 个别名")

    # 过滤活跃用户
    active_users = {k: v for k, v in wxid_index.items() if v["total_msgs"] >= args.min_user_msgs}
    if args.filter_user:
        active_users = {k: v for k, v in active_users.items() if args.filter_user.lower() in k.lower() or args.filter_user.lower() in v.get("main_name", "").lower() or any(args.filter_user.lower() in a.lower() for a in v.get("all_names", {}))}
        print(f"  [FILTER] 只处理匹配 '{args.filter_user}' 的用户: {len(active_users)}")
    print(f"  活跃用户(≥{args.min_user_msgs}条): {len(active_users)}")

    # 过滤活跃群聊
    active_groups = {k: v for k, v in groups.items() if len(v.get("messages", [])) >= args.min_group_msgs}
    print(f"  活跃群聊(≥{args.min_group_msgs}条): {len(active_groups)}")

    if args.dry_run:
        print("\n[DRY RUN] 输出统计信息:")
        print("\n-- 前20个活跃用户 --")
        for wxid, info in sorted(active_users.items(), key=lambda x: -x[1]["total_msgs"])[:20]:
            unique = info.get("unique_name", info.get("resolved_name", info["main_name"]))
            aliases = [n for n in info["all_names"] if n != unique]
            print(f"  {unique:20s} ({info['total_msgs']:5d} 条, {info['chat_count']:2d} 个聊天) 别名: {aliases[:5]}")
        print("\n-- 前20个活跃群聊 --")
        for stem, data in sorted(active_groups.items(), key=lambda x: -len(x[1].get("messages", [])))[:20]:
            cn = resolve_chat_name(stem, data, chatroom_names)
            print(f"  {cn:30s} ({len(data.get('messages', [])):5d} 条)")
        print("\n[DRY RUN] 结束")
        return

    # 5. 初始化 LLM + Engine
    print("\n[LLM] 初始化 QwenClient (deepseek-v4-flash)...")
    try:
        llm = QwenClient(model="deepseek-v4-flash")
    except Exception as e:
        print(f"[LLM] 初始化失败: {e}")
        sys.exit(1)
    engine = MemoryEngine(llm_client=llm)

    # 6. 生成用户 wiki（跨聊天聚合，并发）
    if not args.groups_only:
        print(f"\n{'='*60}")
        print(f"生成用户 Wiki（跨聊天聚合，并发 {args.workers}）")
        print(f"{'='*60}")

        sorted_users = sorted(active_users.items(), key=lambda x: -x[1]["total_msgs"])
        total_users = len(sorted_users)

        def _process_user(args_tuple):
            i, wxid, info = args_tuple
            main_name = info.get("unique_name", info.get("resolved_name", info["main_name"]))

            # 续跑：已存在且内容有效的 wiki 跳过
            wiki_path = WIKI_DIR / "users" / f"{main_name}.md"
            if wiki_path.exists() and wiki_path.stat().st_size > 200:
                return (i, "skip", main_name, info["total_msgs"], info["chat_count"])

            # 收集所有聊天的消息，为每条消息标注 chat_name
            all_msgs = []
            for stem, _ in info["chats"].items():
                chat_data = all_chats.get(stem, {})
                chat_type = classify_chat(stem, chat_data)
                chat_name = resolve_chat_name(stem, chat_data, chatroom_names)
                full_msgs = chat_data.get("messages", [])
                full_msgs = sorted(full_msgs, key=lambda m: m.get("create_time") or 0)

                if chat_type == "private":
                    chat_msgs = [m for m in full_msgs if not is_system_message(m)]
                else:
                    # 群聊：分层上下文策略
                    target_indices = set()
                    for idx, m in enumerate(full_msgs):
                        if m.get("sender_wxid") == wxid:
                            target_indices.add(idx)

                    # 排除容易误匹配的短名字（单字符 或 纯英文≤3）
                    user_names = {
                        name for name in info["all_names"].keys()
                        if len(name) > 1 and not (name.isascii() and len(name) <= 3)
                    }
                    for idx, m in enumerate(full_msgs):
                        text = m.get("text", "")
                        if any(name in text for name in user_names):
                            target_indices.add(idx)

                    k = 10
                    time_window = 5 * 60
                    context_indices = set()
                    for idx in target_indices:
                        for offset in range(-k, k + 1):
                            j = idx + offset
                            if 0 <= j < len(full_msgs):
                                context_indices.add(j)
                        target_ts = full_msgs[idx].get("create_time", 0)
                        for j, m in enumerate(full_msgs):
                            ts = m.get("create_time", 0)
                            if abs(ts - target_ts) <= time_window:
                                context_indices.add(j)

                    chat_msgs = [full_msgs[j] for j in sorted(context_indices) if not is_system_message(full_msgs[j])]

                # 为每条消息标注来源聊天名称（创建副本，避免污染 all_chats 原始数据）
                for m in chat_msgs:
                    m_copy = dict(m)
                    m_copy["_chat_name"] = chat_name
                    all_msgs.append(m_copy)

            # 全局按时间排序（旧 -> 新），按 token 估算分批
            all_msgs = sorted(all_msgs, key=lambda m: m.get("create_time") or 0)
            batches = split_by_tokens(all_msgs, max_tokens=700_000)

            # 分轮次增量更新（从旧到新，每轮复用上一轮生成的 wiki）
            batch_count = len(batches)
            for batch_idx, batch in enumerate(batches):
                # 按聊天名称分批传给 make_simple_messages，保留 chat_name 信息
                simple_msgs = []
                for m in batch:
                    st = m.get("sender_type", "other")
                    sender = m.get("sender", "")
                    text = m.get("text", "")
                    if not text or not text.strip():
                        continue
                    simple_msgs.append(SimpleMsg(
                        sender=sender,
                        sender_type=st,
                        text=text,
                        create_time=m.get("create_time"),
                        account=m.get("account", ""),
                        chat_name=m.get("_chat_name", ""),
                    ))

                if not simple_msgs:
                    continue
                task = {
                    "type": "user",
                    "user_name": main_name,
                    "chat_name": f"{info['chat_count']} 个聊天聚合 (批次 {batch_idx + 1}/{batch_count})" if batch_count > 1 else f"{info['chat_count']} 个聊天聚合",
                    "messages": simple_msgs,
                    "bot_replies": [],
                    "timestamp": time.time(),
                }
                try:
                    engine._do_update_user(task)
                except Exception as e:
                    return (i, "error", main_name, str(e), "")

            return (i, "ok", main_name, info["total_msgs"], info["chat_count"])

        user_tasks = [(i, wxid, info) for i, (wxid, info) in enumerate(sorted_users, 1)]
        ok_count = 0
        err_count = 0
        skip_count = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process_user, t): t for t in user_tasks}
            for future in as_completed(futures):
                result = future.result()
                i, status = result[0], result[1]
                if status == "ok":
                    _, _, main_name, total_msgs, chat_count = result
                    ok_count += 1
                    print(f"  [{i}/{total_users}] ✓ {main_name} ({total_msgs} 条, {chat_count} 个聊天)")
                elif status == "skip":
                    _, _, main_name, total_msgs, chat_count = result
                    skip_count += 1
                    print(f"  [{i}/{total_users}] ⏭ {main_name} (已存在, 跳过)")
                elif status == "error":
                    _, _, main_name, err, _ = result
                    err_count += 1
                    print(f"  [{i}/{total_users}] ✗ {main_name} ({err})")
        print(f"  用户 wiki 完成: {ok_count} 成功, {skip_count} 跳过, {err_count} 失败")

    # 7. 生成群聊 wiki（并发）
    if not args.users_only:
        print(f"\n{'='*60}")
        print(f"生成群聊 Wiki（并发 {args.workers}）")
        print(f"{'='*60}")

        sorted_groups = sorted(active_groups.items(), key=lambda x: -len(x[1].get("messages", [])))
        total_groups = len(sorted_groups)

        def _process_group(args_tuple):
            i, stem, data = args_tuple
            chat_name = resolve_chat_name(stem, data, chatroom_names)

            # 续跑：已存在且内容有效的群聊 wiki 跳过
            safe_name = chat_name.replace("/", "_").replace("\\", "_")
            wiki_path = WIKI_DIR / "groups" / f"{safe_name}.md"
            if wiki_path.exists() and wiki_path.stat().st_size > 200:
                return (i, "skip", chat_name, len(data.get("messages", [])))

            msgs = data.get("messages", [])
            simple_msgs = make_simple_messages(msgs, max_msgs=args.max_total_msgs, chat_name=chat_name)
            if not simple_msgs:
                return (i, "skip", chat_name, 0)
            task = {
                "type": "group",
                "group_name": chat_name,
                "chat_name": chat_name,
                "messages": simple_msgs,
                "bot_replies": [],
                "timestamp": time.time(),
            }
            try:
                engine._do_update_group(task)
                return (i, "ok", chat_name, len(msgs))
            except Exception as e:
                return (i, "error", chat_name, str(e))

        group_tasks = [(i, stem, data) for i, (stem, data) in enumerate(sorted_groups, 1)]
        ok_count = 0
        err_count = 0
        skip_count = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process_group, t): t for t in group_tasks}
            for future in as_completed(futures):
                result = future.result()
                i, status = result[0], result[1]
                if status == "ok":
                    _, _, chat_name, msg_count = result
                    ok_count += 1
                    print(f"  [{i}/{total_groups}] ✓ {chat_name} ({msg_count} 条)")
                elif status == "skip":
                    _, _, chat_name, msg_count = result
                    skip_count += 1
                    print(f"  [{i}/{total_groups}] ⏭ {chat_name} (已存在, 跳过)")
                elif status == "error":
                    _, _, chat_name, err = result
                    err_count += 1
                    print(f"  [{i}/{total_groups}] ✗ {chat_name} ({err})")
        print(f"  群聊 wiki 完成: {ok_count} 成功, {skip_count} 跳过, {err_count} 失败")

    print(f"\n{'='*60}")
    print("完成！")
    if not args.users_only and not args.groups_only:
        print(f"  用户 wiki: {WIKI_DIR / 'users'}")
        print(f"  群聊 wiki: {WIKI_DIR / 'groups'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
