#!/usr/bin/env python3
"""
Bot 回复质量 Benchmark — 基于生产环境人工标注的真实 tick

从 tick_log 读取所有有人工标注的 tick，评估 Bot 回复质量：
- 人工标注 badcase 的比例
- 各 badcase 类型的分布
- 逐 case 详情

这个 benchmark 不需要调用 LLM，直接统计人工标注结果。
配合 Judge benchmark 使用，Judge 测"自动判定是否准确"，这个测"Bot 本身回复质量"。

用法:
    python -m pytest src/tests/test_reply_quality_benchmark_v2.py -v
    python src/tests/test_reply_quality_benchmark_v2.py  # CLI 详细报告
"""

import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "cases.db"
DEBUG_DIR = PROJECT_ROOT / "data" / "debug"


@dataclass
class ReplyBenchmarkCase:
    case_name: str
    db_id: int
    session_id: str
    tick_id: int
    chat_name: str
    bot_reply: str
    human_is_badcase: bool
    human_badcase_type: str
    human_notes: str
    tool_calls: list
    judge_is_badcase: bool
    judge_score: float


def load_cases() -> List[ReplyBenchmarkCase]:
    """从 tick_log 读取所有人工标注的 tick。"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM tick_log WHERE human_is_badcase IS NOT NULL ORDER BY id"
    ).fetchall()
    conn.close()

    cases = []
    for r in rows:
        d = dict(r)
        replies = d.get("replies_sent_json", "[]") or "[]"
        try:
            reply_list = json.loads(replies)
            bot_reply = " | ".join(reply_list) if isinstance(reply_list, list) else replies
        except Exception:
            bot_reply = replies

        tc = d.get("tool_calls_json", "[]") or "[]"
        try:
            tool_calls = json.loads(tc)
        except Exception:
            tool_calls = []

        cases.append(ReplyBenchmarkCase(
            case_name=f"real_{d['id']}_{d.get('session_id','')}_tick{d.get('tick_id',0)}",
            db_id=d["id"],
            session_id=d.get("session_id", ""),
            tick_id=d.get("tick_id", 0),
            chat_name=d.get("chat_name", ""),
            bot_reply=bot_reply,
            human_is_badcase=bool(d.get("human_is_badcase", 0)),
            human_badcase_type=d.get("human_badcase_type", "") or "other",
            human_notes=d.get("human_notes", "") or "",
            tool_calls=tool_calls,
            judge_is_badcase=bool(d.get("judge_is_badcase", 0)),
            judge_score=d.get("judge_score", 0) or 0,
        ))
    return cases


BENCHMARK_CASES = load_cases()


# =============================================================================
# Test Runner
# =============================================================================

def _pytest_id(case: ReplyBenchmarkCase) -> str:
    return case.case_name


@pytest.mark.parametrize("case", BENCHMARK_CASES, ids=_pytest_id)
def test_reply_quality(case: ReplyBenchmarkCase):
    """每个 case 验证：人工标注为 badcase 的应该被修复。"""
    detail = (
        f"\n  Chat: {case.chat_name}"
        f"\n  Bot 回复: {case.bot_reply[:100]}"
        f"\n  Badcase 类型: {case.human_badcase_type}"
        f"\n  人工备注: {case.human_notes[:120]}"
        f"\n  Judge 评分: {case.judge_score:.0f}"
        f"\n  Judge 判定: {'badcase' if case.judge_is_badcase else 'OK'}"
    )
    assert not case.human_is_badcase, (
        f"Bot 回复被人工标注为 badcase ({case.case_name}){detail}"
    )


# =============================================================================
# CLI 详细报告
# =============================================================================

def main():
    cases = load_cases()
    if not cases:
        print("❌ 没有找到人工标注的 tick。")
        return

    bad = [c for c in cases if c.human_is_badcase]
    ok = [c for c in cases if not c.human_is_badcase]

    # 类型分布
    type_counts = {}
    for c in bad:
        t = c.human_badcase_type or "other"
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"📊 Bot 回复质量 Benchmark — 基于 {len(cases)} 个人工标注")
    print(f"   Badcase: {len(bad)}/{len(cases)} ({len(bad)/len(cases)*100:.0f}%)")
    print(f"   正常:    {len(ok)}/{len(cases)} ({len(ok)/len(cases)*100:.0f}%)")
    print("\n   Badcase 类型分布:")
    for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"     {t}: {n}")

    judge_match = sum(1 for c in cases if c.human_is_badcase == c.judge_is_badcase)
    print(f"\n   Judge 与人工一致: {judge_match}/{len(cases)} ({judge_match/len(cases)*100:.0f}%)")

    print(f"\n{'='*60}")
    print("逐 Case 详情:")
    for c in cases:
        status = "❌ BAD" if c.human_is_badcase else "✅ OK"
        j_match = "✅" if c.human_is_badcase == c.judge_is_badcase else "❌"
        print(f"  {status} {c.case_name}")
        print(f"    聊天: {c.chat_name} | Judge: {c.judge_score:.0f} {'BAD' if c.judge_is_badcase else 'OK'} {j_match}")
        print(f"    回复: {c.bot_reply[:120]}")
        print(f"    备注: {c.human_notes[:120]}")
        if c.tool_calls:
            tools = ", ".join(t.get("tool_name", "?") for t in c.tool_calls[:3])
            print(f"    工具: {tools}")
        print()


if __name__ == "__main__":
    main()
