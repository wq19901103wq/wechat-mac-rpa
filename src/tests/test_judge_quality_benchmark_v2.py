#!/usr/bin/env python3
"""
Judge LLM 质量 Benchmark — 基于生产环境人工标注的真实 tick

从 tick_log 读取所有有人工标注 (human_is_badcase) 的 tick，
用 JudgeWorker._judge() 重新评分，对比人工 GT 计算准确率。

用法:
    python -m pytest src/tests/test_judge_quality_benchmark_v2.py -v --run-api --n-runs 3
    python -m pytest src/tests/test_judge_quality_benchmark_v2.py -v       # 缓存回归
"""

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "judge_quality_v2"
CACHE_DIR = FIXTURE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = PROJECT_ROOT / "data" / "cases.db"
DEBUG_DIR = PROJECT_ROOT / "data" / "debug"

_logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class JudgeBenchmarkCase:
    case_name: str
    tick_data: dict
    ground_truth_is_badcase: bool
    ground_truth_type: str
    notes: str = ""
    db_id: int = 0
    session_id: str = ""
    tick_id: int = 0
    chat_name: str = ""


@dataclass
class JudgeBenchmarkResult:
    case_name: str
    ground_truth_is_badcase: bool
    ground_truth_type: str
    predicted_is_badcase: bool
    predicted_type: str
    predicted_confidence: float
    passed: bool
    overall_score: float = 0
    dimensions: dict = None
    n_runs: int = 1
    badcase_votes: int = 0
    notes: str = ""
    error: str = ""


# =============================================================================
# Case 加载器
# =============================================================================

def _load_gt_cases() -> List[JudgeBenchmarkCase]:
    """从 tick_log 读取所有人工标注的 tick，构建 benchmark cases。"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM tick_log WHERE human_is_badcase IS NOT NULL ORDER BY id"
    ).fetchall()
    conn.close()

    cases = []
    for r in rows:
        d = dict(r)
        sp = d.get("system_prompt") or ""
        up = d.get("user_prompt") or ""

        # 重建 llm_messages
        llm_messages = []
        if sp:
            llm_messages.append({"role": "system", "content": sp})
        if up:
            llm_messages.append({"role": "user", "content": up})

        # 从 debug JSON 补充更完整的 llm_messages（如果有的话）
        sid = d.get("session_id", "")
        tid = d.get("tick_id", 0)
        # 尝试按 session_id:tick_id 和 tick_id 两种方式查找 debug JSON
        debug_files = sorted(DEBUG_DIR.glob(f"tick_*_{tid}.json"))
        if sid and not debug_files:
            debug_files = sorted(DEBUG_DIR.glob(f"tick_*_{sid}*_{tid}.json"))
        if debug_files:
            try:
                dbg = json.loads(debug_files[-1].read_text(encoding="utf-8"))
                dbg_msgs = dbg.get("reply_llm_messages", []) or []
                if dbg_msgs:
                    llm_messages = dbg_msgs
            except Exception as e:
                _logger.warning("load reply debug messages failed: %s", e)

        # fallback：用 raw_response 补充 assistant message，让 Judge 看到完整的 LLM 交互
        if len(llm_messages) <= 2:
            raw = d.get("raw_response", "")
            if raw:
                try:
                    raw_data = json.loads(raw)
                    if isinstance(raw_data, dict) and raw_data.get("replies"):
                        content = raw_data["replies"][0] if raw_data["replies"] else raw
                    else:
                        content = raw
                except Exception:
                    content = raw
                llm_messages.append({"role": "assistant", "content": str(content)[:2000]})

        # Bot 回复文本
        replies_json = d.get("replies_sent_json", "[]") or "[]"
        try:
            reply_list = json.loads(replies_json)
            bot_reply = " | ".join(reply_list) if isinstance(reply_list, list) else replies_json
        except Exception:
            bot_reply = replies_json

        # tool calls
        tc_json = d.get("tool_calls_json", "[]") or "[]"
        try:
            tool_calls = json.loads(tc_json)
        except Exception:
            tool_calls = []

        # tool results（关键：让 Judge 看到工具返回的具体内容，用于幻觉核查）
        tr_json = d.get("tool_results_json", "[]") or "[]"

        # session_input_messages（从 user_prompt 或 debug JSON 提取）
        session_msgs = []
        if debug_files:
            try:
                dbg = json.loads(debug_files[-1].read_text(encoding="utf-8"))
                session_msgs = dbg.get("session_input_messages", []) or []
            except Exception as e:
                _logger.warning("load session debug messages failed: %s", e)
        if not session_msgs:
            # 从 user_prompt 的 [未读消息] 段粗略提取
            session_msgs = [{"sender": "unknown", "sender_type": "other", "text": up[:200]}]

        tick_data = {
            "tick_id": tid,
            "chat_name": d.get("chat_name", ""),
            "session_input_messages": session_msgs,
            "bot_reply_text": bot_reply,
            "tool_calls": tool_calls,
            "tool_results_json": tr_json,
            "full_user_prompt": up,
            "full_system_prompt": sp,
            "full_llm_messages": llm_messages,
        }

        gt_type = d.get("human_badcase_type", "") or "other"
        gt_is_badcase = bool(d.get("human_is_badcase", 0))

        cases.append(JudgeBenchmarkCase(
            case_name=f"real_{d['id']}_{d.get('session_id','')}_tick{d.get('tick_id',0)}",
            tick_data=tick_data,
            ground_truth_is_badcase=gt_is_badcase,
            ground_truth_type=gt_type,
            notes=(d.get("human_notes") or "")[:200],
            db_id=d["id"],
            session_id=sid,
            tick_id=tid,
            chat_name=d.get("chat_name", ""),
        ))

    return cases


# =============================================================================
# Judge 调用
# =============================================================================

def _run_judge(tick_data: dict) -> dict:
    """调用 JudgeWorker._judge() 一次。"""
    from src.badcase.judge_worker import JudgeWorker
    worker = JudgeWorker(use_fewshot=False)
    return worker._judge(tick_data)


def _run_judge_n_times(tick_data: dict, n: int = 3) -> List[dict]:
    """跑 N 次 Judge，返回所有结果列表。"""
    results = []
    for i in range(n):
        try:
            result = _run_judge(tick_data)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "is_badcase": False, "badcase_type": "error"})
    return results


# =============================================================================
# 指标计算
# =============================================================================

def _compute_metrics(results: List[JudgeBenchmarkResult]) -> dict:
    """计算准确率、精确率、召回率、F1。"""
    tp = sum(1 for r in results if r.ground_truth_is_badcase and r.predicted_is_badcase)
    fp = sum(1 for r in results if not r.ground_truth_is_badcase and r.predicted_is_badcase)
    tn = sum(1 for r in results if not r.ground_truth_is_badcase and not r.predicted_is_badcase)
    fn = sum(1 for r in results if r.ground_truth_is_badcase and not r.predicted_is_badcase)

    total = len(results)
    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {
        "total": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "passed": tp + tn,
        "failed": fp + fn,
    }


# =============================================================================
# Benchmark Cases Collection
# =============================================================================

BENCHMARK_CASES = _load_gt_cases()


# =============================================================================
# Test Runner
# =============================================================================

def _pytest_id(case: JudgeBenchmarkCase) -> str:
    return case.case_name


@pytest.mark.parametrize("case", BENCHMARK_CASES, ids=_pytest_id)
def test_judge_quality(case: JudgeBenchmarkCase, request):
    """每个 case 跑 N 次 Judge，多数投票判定是否通过。"""
    n_runs = request.config.getoption("--n-runs", default=3)

    # 尝试读缓存
    cache_key = hashlib.md5(
        json.dumps(case.tick_data, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    cache_path = CACHE_DIR / f"{case.case_name}_{cache_key}.json"

    if not request.config.getoption("--run-api", default=False):
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            results = cached.get("runs", [])
            if len(results) >= n_runs:
                results = results[:n_runs]
            else:
                # 缓存不够，补跑
                new_results = _run_judge_n_times(case.tick_data, n_runs - len(results))
                results.extend(new_results)
                cache_path.write_text(json.dumps({"runs": results, "case_name": case.case_name},
                                                  ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            pytest.skip("缓存不存在，请用 --run-api 首次生成")
    else:
        results = _run_judge_n_times(case.tick_data, n_runs)
        cache_path.write_text(json.dumps({"runs": results, "case_name": case.case_name},
                                          ensure_ascii=False, indent=2), encoding="utf-8")

    # 多数投票
    bc_votes = sum(1 for r in results if r.get("is_badcase"))
    predicted_is_badcase = bc_votes > n_runs / 2
    predicted_type = "none"
    if bc_votes > 0:
        types = [r.get("badcase_type", "none") for r in results if r.get("is_badcase")]
        predicted_type = max(set(types), key=types.count) if types else "none"

    avg_score = sum(r.get("overall_score", 0) for r in results) / n_runs if n_runs > 0 else 0
    avg_confidence = sum(r.get("confidence", 0) for r in results) / n_runs if n_runs > 0 else 0

    result = JudgeBenchmarkResult(
        case_name=case.case_name,
        ground_truth_is_badcase=case.ground_truth_is_badcase,
        ground_truth_type=case.ground_truth_type,
        predicted_is_badcase=predicted_is_badcase,
        predicted_type=predicted_type,
        predicted_confidence=avg_confidence,
        passed=predicted_is_badcase == case.ground_truth_is_badcase,
        overall_score=avg_score,
        n_runs=n_runs,
        badcase_votes=bc_votes,
        notes=case.notes,
    )

    # 详情输出
    detail = (
        f"\n  GT: {'badcase' if case.ground_truth_is_badcase else 'OK'} ({case.ground_truth_type})"
        f"\n  Pred: {'badcase' if predicted_is_badcase else 'OK'} ({predicted_type})"
        f"\n  Votes: {bc_votes}/{n_runs}  Score: {avg_score:.0f}  Conf: {avg_confidence:.2f}"
        f"\n  Notes: {case.notes[:120]}"
    )

    assert result.passed, (
        f"Judge 判定与人工 GT 不一致 ({case.case_name}){detail}"
    )


# =============================================================================
# CLI 详细报告
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Judge Quality Benchmark v2 (real production data)")
    parser.add_argument("--run-api", action="store_true", help="真实 API 调用")
    parser.add_argument("--n-runs", type=int, default=3, help="每个 case 跑 N 次")
    args = parser.parse_args()

    cases = _load_gt_cases()
    if not cases:
        print("❌ 没有找到人工标注的 tick。请先在 /ticks 页面完成 GT 标注。")
        return

    print(f"📊 Judge Quality Benchmark v2 — 基于 {len(cases)} 个生产人工标注")
    print(f"   每个 case 跑 {args.n_runs} 次，多数投票\n")

    results = []
    for case in cases:
        print(f"[{case.case_name}]", end=" ", flush=True)
        if args.run_api:
            runs = _run_judge_n_times(case.tick_data, args.n_runs)
        else:
            cache_key = hashlib.md5(
                json.dumps(case.tick_data, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
            cache_path = CACHE_DIR / f"{case.case_name}_{cache_key}.json"
            if cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                runs = cached.get("runs", [])[:args.n_runs]
            else:
                print("(无缓存，跳过)")
                continue

        bc_votes = sum(1 for r in runs if r.get("is_badcase"))
        predicted = bc_votes > args.n_runs / 2
        passed = predicted == case.ground_truth_is_badcase
        status = "✅" if passed else "❌"
        print(f"{status} GT={'BAD' if case.ground_truth_is_badcase else 'OK'} "
              f"Pred={'BAD' if predicted else 'OK'} "
              f"Votes={bc_votes}/{args.n_runs}"
              f" | {case.notes[:80]}")

        avg_score = sum(r.get("overall_score", 0) for r in runs) / args.n_runs if runs else 0
        results.append(JudgeBenchmarkResult(
            case_name=case.case_name,
            ground_truth_is_badcase=case.ground_truth_is_badcase,
            ground_truth_type=case.ground_truth_type,
            predicted_is_badcase=predicted,
            predicted_type="",
            predicted_confidence=0,
            passed=passed,
            overall_score=avg_score,
            n_runs=args.n_runs,
            badcase_votes=bc_votes,
            notes=case.notes,
        ))

    metrics = _compute_metrics(results)
    print(f"\n{'='*60}")
    print("📈 总指标:")
    print(f"   Accuracy:  {metrics['accuracy']:.1%} ({metrics['passed']}/{metrics['total']})")
    print(f"   Precision: {metrics['precision']:.1%}")
    print(f"   Recall:    {metrics['recall']:.1%}")
    print(f"   F1:        {metrics['f1']:.3f}")
    print(f"   TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")


if __name__ == "__main__":
    main()
