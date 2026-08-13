"""私有 benchmark 编排、来源留痕和机器候选筛选。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess  # nosec B404 - fixed git commands only
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from src.benchmarks.ocr_quality import build_summary, load_cases, run_benchmark

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = PROJECT_ROOT / "data" / "private_benchmarks"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "cases.db"


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _git_state() -> dict[str, Any]:
    git_path = shutil.which("git")
    if not git_path:
        return {"commit": "", "dirty": None}
    try:
        commit = subprocess.run(  # nosec B603 - executable and arguments are trusted
            [git_path, "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(subprocess.run(  # nosec B603 - executable and arguments are trusted
            [git_path, "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "", "dirty": None}


def run_ocr(refresh: bool = False, model: str = "qwen3.6-flash") -> dict[str, Any]:
    """运行 OCR 私有集，并自动归类失败原因。"""
    cases = load_cases()
    results, missing_cache = run_benchmark(
        use_api=refresh,
        model=model,
        cases=cases,
    )
    summary = build_summary(cases, results, missing_cache)
    failures = []
    reason_counts: Counter[str] = Counter()
    for result in results:
        if result.passed:
            continue
        reasons = []
        if result.error:
            reasons.append("api_error")
        if not result.chat_name_match:
            reasons.append("chat_name")
        if not result.message_count_match:
            reasons.append("message_count")
        if result.sender_accuracy < 1:
            reasons.append("sender")
        if result.text_accuracy < 0.8:
            reasons.append("text")
        reason_counts.update(reasons)
        failures.append({
            "case": result.case_name,
            "cohort": result.cohort,
            "reasons": reasons,
        })

    case_manifest = [{
        "name": case.name,
        "cohort": case.cohort,
        "expected": case.expected,
        "image_sha256": hashlib.sha256(case.screenshot_path.read_bytes()).hexdigest(),
    } for case in cases]
    return {
        "status": summary["status"],
        "model": model,
        "dataset_fingerprint": _fingerprint(case_manifest),
        "summary": summary,
        "failure_triage": {
            "count": len(failures),
            "by_reason": dict(sorted(reason_counts.items())),
            "cases": failures,
        },
    }


def run_judge(refresh: bool = False, n_runs: int = 3) -> dict[str, Any]:
    """运行 Judge 私有集；缓存不完整时不发布局部准确率。"""
    os.environ["RUN_PRODUCTION_BENCHMARKS"] = "1"
    from src.tests.test_judge_quality_benchmark_v2 import (
        _compute_metrics,
        _load_gt_cases,
        run_benchmark,
    )

    cases = _load_gt_cases()
    results = run_benchmark(use_api=refresh, n_runs=n_runs)
    manifest = [{
        "case_name": case.case_name,
        "ground_truth_is_badcase": case.ground_truth_is_badcase,
        "ground_truth_type": case.ground_truth_type,
        "tick_data": case.tick_data,
    } for case in cases]
    complete = bool(cases) and len(results) == len(cases)
    return {
        "status": "available" if complete else "unavailable",
        "model": "deepseek-v4-flash",
        "runs_per_case": n_runs,
        "configured_cases": len(cases),
        "evaluated_cases": len(results),
        "missing_results": len(cases) - len(results),
        "dataset_fingerprint": _fingerprint(manifest),
        "metrics": _compute_metrics(results) if complete else None,
        "note": "缓存不完整时不计算准确率" if not complete else "",
    }


def mine_review_candidates(
    db_path: Path = DEFAULT_DB_PATH,
    limit: int = 50,
) -> dict[str, Any]:
    """从机器信号筛选高价值 tick，不读取或输出消息正文。"""
    if not db_path.exists():
        return {"status": "unavailable", "count": 0, "by_reason": {}, "items": []}
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT id, tick_id, created_at, chat_name,
                   judge_score, judge_is_badcase, judge_badcase_type,
                   human_is_badcase, feedback_decision, self_refine_applied
            FROM tick_log
            WHERE judge_is_badcase = 1
               OR feedback_decision IN ('fail', 'error')
               OR (human_is_badcase IS NOT NULL AND judge_is_badcase IS NOT NULL
                   AND human_is_badcase != judge_is_badcase)
            ORDER BY id DESC
            """
        ).fetchall()
    finally:
        connection.close()

    candidates = []
    reason_counts: Counter[str] = Counter()
    for row in rows:
        reasons = []
        score = 0
        if (
            row["human_is_badcase"] is not None
            and row["judge_is_badcase"] is not None
            and row["human_is_badcase"] != row["judge_is_badcase"]
        ):
            reasons.append("human_judge_disagreement")
            score += 100
        if row["feedback_decision"] == "fail":
            reasons.append("self_refine_failed")
            score += 90
        elif row["feedback_decision"] == "error":
            reasons.append("self_refine_error")
            score += 75
        if row["judge_is_badcase"] == 1:
            reasons.append("judge_badcase")
            score += 60
        if row["judge_is_badcase"] == 1 and row["judge_badcase_type"] in (None, "", "none"):
            reasons.append("judge_output_inconsistent")
            score += 25
        reason_counts.update(reasons)
        chat_ref = hashlib.sha256(str(row["chat_name"] or "").encode()).hexdigest()[:10]
        candidates.append({
            "db_id": row["id"],
            "tick_id": row["tick_id"],
            "created_at": row["created_at"],
            "chat_ref": chat_ref,
            "priority": score,
            "reasons": reasons,
            "judge_score": row["judge_score"],
            "judge_badcase_type": row["judge_badcase_type"] or "none",
            "self_refine_applied": bool(row["self_refine_applied"]),
        })
    candidates.sort(key=lambda item: (-item["priority"], -item["db_id"]))
    selected = candidates[:max(limit, 0)]
    return {
        "status": "available",
        "candidate_pool": len(candidates),
        "count": len(selected),
        "by_reason": dict(sorted(reason_counts.items())),
        "items": selected,
        "privacy": "不包含聊天名或消息正文；chat_ref 是不可逆短哈希",
    }


def run_private_benchmarks(
    refresh: str = "none",
    ocr_model: str = "qwen3.6-flash",
    judge_runs: int = 3,
    candidate_limit: int = 50,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """运行当前可用的私有评测与机器候选筛选。"""
    if refresh not in {"none", "ocr", "judge", "all"}:
        raise ValueError(f"unsupported refresh mode: {refresh}")
    generated_at = datetime.now().astimezone().isoformat()
    report = {
        "schema_version": 1,
        "generated_at": generated_at,
        "mode": "cached" if refresh == "none" else f"refresh:{refresh}",
        "provenance": {
            **_git_state(),
            "runner_fingerprint": _fingerprint(Path(__file__).read_text(encoding="utf-8")),
        },
        "benchmarks": {
            "ocr_quality": run_ocr(refresh in {"ocr", "all"}, ocr_model),
            "judge_quality_v2": run_judge(refresh in {"judge", "all"}, judge_runs),
        },
        "review_candidates": mine_review_candidates(db_path, candidate_limit),
    }
    report["report_fingerprint"] = _fingerprint(report)
    return report
