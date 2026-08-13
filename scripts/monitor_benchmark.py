#!/usr/bin/env python3
"""
Benchmark 监控 — 每 3 小时跑一次，检测指标退化

用法:
    python scripts/monitor_benchmark.py                # 跑一次，对比上次
    python scripts/monitor_benchmark.py --alert-only   # 仅输出告警

数据:
    data/benchmark_history/YYYY-MM-DD.json            # 每日完整快照
    data/benchmark_history/YYYY-MM-DD_HH-MM.json      # 每次监控快照
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
HISTORY_DIR = PROJECT_ROOT / "data" / "benchmark_history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

_logger = logging.getLogger(__name__)

# 退化告警阈值
ALERT_THRESHOLDS = {
    "judge_quality.accuracy": 0.40,
    "judge_quality.recall": 0.25,
    "tool_decision.accuracy": 0.60,
    "tool_decision.recall": 0.80,
    "reply_quality.pass_rate": 0.80,
    "memory_search.recall": 0.75,
}


def _get_git_commit() -> str:
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
                           capture_output=True, timeout=5)
        return r.stdout.decode().strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _run_all():
    """运行所有 benchmark 的轻量指标采集（缓存模式，不调 API）。"""
    from scripts.run_daily_benchmark import (
        _run_tool_decision, _run_reply_quality, _run_memory_search,
        _run_judge_quality,
    )
    return {
        "tool_decision": _run_tool_decision(use_api=False),
        "reply_quality": _run_reply_quality(use_api=False),
        "memory_search": _run_memory_search(),
        "judge_quality": _run_judge_quality(use_api=False),
    }


def _latest_metrics() -> dict | None:
    """获取最近一次的指标快照。"""
    files = sorted(HISTORY_DIR.glob("*.json"))
    for f in reversed(files):
        if f.name == "trend.json":
            continue
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def _load_metrics(filepath: Path) -> dict | None:
    try:
        return json.loads(filepath.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_snapshot(metrics: dict, prefix: str = ""):
    now = datetime.now()
    snapshot = {
        "timestamp": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "git_commit": _get_git_commit(),
        "benchmarks": metrics,
    }
    filename = f"{now.strftime('%Y-%m-%d')}_{now.strftime('%H-%M')}.json"
    filepath = HISTORY_DIR / filename
    filepath.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    # Also update daily metrics in DB
    try:
        from src.badcase.case_db import get_db
        db = get_db()
        db.insert_daily_metrics(
            date=now.strftime("%Y-%m-%d"),
            benchmarks=metrics,
            git_commit=snapshot["git_commit"],
        )
    except Exception as e:
        _logger.warning("save snapshot failed: %s", e)

    return snapshot, filepath


def _compute_change(cur: dict, prev: dict, key: str) -> float | None:
    """计算某个指标的变化。"""
    val = cur.get(key)
    old = prev.get(key)
    if val is None or old is None:
        return None
    return val - old


def _check_alerts(metrics: dict, prev_metrics: dict | None):
    """检查告警阈值 + 对比上次变化。"""
    alerts = []
    warnings = []

    for bench_name, bench_data in metrics.items():
        if "error" in bench_data:
            alerts.append(f"🔴 {bench_name} 运行失败: {bench_data['error']}")
            continue

        for key, val in bench_data.items():
            if not isinstance(val, (int, float)):
                continue
            alert_key = f"{bench_name}.{key}"
            if alert_key in ALERT_THRESHOLDS and val < ALERT_THRESHOLDS[alert_key]:
                alerts.append(f"🔴 {alert_key}={val:.1%} < 阈值 {ALERT_THRESHOLDS[alert_key]:.0%}")

    if prev_metrics and prev_metrics.get("benchmarks"):
        prev_benches = prev_metrics.get("benchmarks", {})
        for bench_name, bench_data in metrics.items():
            if bench_name not in prev_benches:
                continue
            prev_data = prev_benches[bench_name]
            for key in ("accuracy", "recall", "precision", "f1", "pass_rate"):
                delta = _compute_change(bench_data, prev_data, key)
                if delta is not None and delta < -0.08:  # 下降 > 8%
                    warnings.append(f"🟡 {bench_name}.{key} 下降 {abs(delta):.1%}")

    return alerts, warnings


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark 监控")
    parser.add_argument("--alert-only", action="store_true", help="仅输出告警，正常时无输出")
    args = parser.parse_args()

    t0 = time.time()
    prev = _latest_metrics()
    metrics = _run_all()
    snapshot, filepath = _save_snapshot(metrics)
    alerts, warnings = _check_alerts(metrics, prev)

    elapsed = time.time() - t0

    if args.alert_only:
        for a in alerts:
            print(a)
        for w in warnings:
            print(w)
        if not alerts and not warnings:
            pass  # 静默
    else:
        print(f"📊 监控 {datetime.now().strftime('%H:%M')} ({elapsed:.1f}s)  git={snapshot['git_commit'][:8]}")
        for name, data in metrics.items():
            if "error" in data:
                print(f"  {name}: ❌ {data['error']}")
            else:
                acc = data.get("accuracy") or data.get("pass_rate") or data.get("recall")
                if acc is not None:
                    print(f"  {name}: {acc:.1%}  ({data.get('case_count', '?')} cases)")

        if alerts:
            print(f"\n🔴 ALERTS:")
            for a in alerts:
                print(f"  {a}")
        if warnings:
            print(f"\n🟡 WARNINGS:")
            for w in warnings:
                print(f"  {w}")
        if not alerts and not warnings:
            print(f"  ✅ 一切正常")

        print(f"\n💾 {filepath}")


if __name__ == "__main__":
    main()
