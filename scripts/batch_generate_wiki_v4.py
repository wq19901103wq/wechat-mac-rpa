#!/usr/bin/env python3
"""
批量生成群聊 wiki (v4) - 第一轮：所有群最近 500 条消息生成基础 wiki
- 并发 4 个，外层 600s 强制超时
- 断点续传（跳过已有 wiki 文件）
- 保存到 data/memory/wiki/groups/
"""

import sys
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.session.global_store import GlobalStore
from src.utils.qwen_client import QwenClient
from src.memory.wiki_prompts import BATCH_DEFAULT_GROUP_WIKI, BATCH_UPDATE_GROUP_PROMPT

_logger = logging.getLogger(__name__)

client = QwenClient(model="deepseek-v4-flash")
wiki_dir = Path("data/memory/wiki/groups")
wiki_dir.mkdir(parents=True, exist_ok=True)

store = GlobalStore()

def safe_filename(name: str) -> str:
    """生成安全文件名（保留中文）"""
    return "".join(c if c.isalnum() or c in "_-\u4e00-\u9fff" else "_" for c in name)


def format_conversation(messages):
    lines = []
    for m in messages:
        sender = getattr(m, 'sender', '?')
        text = getattr(m, 'text', '')
        ts_int = getattr(m, 'create_time', None)
        tstr = ""
        if ts_int:
            try:
                tstr = datetime.fromtimestamp(int(ts_int)).strftime("%Y-%m-%d %H:%M")
            except Exception as e:
                _logger.warning("timestamp conversion failed: %s", e)
        lines.append(f"[{tstr}] {sender}: {text}")
    return "\n".join(lines)


def process_chat(name_state):
    name, state = name_state
    safe = safe_filename(name)
    wiki_path = wiki_dir / f"{safe}.md"

    # 断点续传：已有 wiki 且长度 >100 则跳过
    if wiki_path.exists() and wiki_path.stat().st_size > 100:
        return name, "exists"

    msg_count = len(state.messages)
    if msg_count == 0:
        return name, "empty"

    # 第一轮：取最近 500 条
    limit = min(msg_count, 500)
    recent = state.messages[-limit:]
    conversation = format_conversation(recent)

    current_wiki = BATCH_DEFAULT_GROUP_WIKI.format(group_name=name)
    if wiki_path.exists():
        current_wiki = wiki_path.read_text(encoding="utf-8")

    now = time.strftime("%Y-%m-%d %H:%M")
    prompt = BATCH_UPDATE_GROUP_PROMPT.format(
        current_wiki=current_wiki,
        chat_name=name,
        current_time=now,
        conversation=conversation,
    )

    try:
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=10000,
            timeout=500,
        )
        new_wiki = response.strip() if response else ""
        if new_wiki and len(new_wiki) > 50:
            wiki_path.write_text(new_wiki, encoding="utf-8")
            return name, f"ok_{len(new_wiki)}"
        return name, "short"
    except Exception as e:
        return name, f"error_{e}"


# 准备任务
chats = list(store.chats.items())
total = len(chats)
print(f"总共 {total} 个聊天，第一轮：最近 500 条生成基础 wiki（并发 4，timeout 500s）...")

success = 0
failed = 0
skipped = 0

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(process_chat, item): item[0] for item in chats}
    for i, future in enumerate(as_completed(futures), 1):
        name = futures[future]
        try:
            _, status = future.result(timeout=600)
        except Exception as e:
            status = f"timeout_{e}"

        if status.startswith("ok"):
            success += 1
        elif status == "exists":
            skipped += 1
        else:
            failed += 1

        if i % 20 == 0 or i <= 5 or status.startswith("error") or status.startswith("timeout"):
            print(f"  [{i}/{total}] {name[:40]}... {status} | 成功:{success} 跳过:{skipped} 失败:{failed}")

print(f"\n第一轮完成: 成功={success} 跳过={skipped} 失败={failed}")
