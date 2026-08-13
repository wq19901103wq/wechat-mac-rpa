#!/usr/bin/env python3
"""
每日 Benchmark 指标采集 — 定时运行，积累历史趋势数据。

用法:
    python scripts/run_daily_benchmark.py              # 默认：缓存模式，采集指标
    python scripts/run_daily_benchmark.py --run-api    # API 模式：重新调用 LLM
    python scripts/run_daily_benchmark.py --report     # 打印历史趋势报告
    python scripts/run_daily_benchmark.py --html       # 生成 trend_report.html

数据存储:
    data/benchmark_history/YYYY-MM-DD.json   # 每日指标快照
    data/benchmark_history/trend.json        # 聚合趋势数据

Cron 示例（每天凌晨 3:17）:
    17 3 * * * cd ~/wechat-mac-rpa && python scripts/run_daily_benchmark.py
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

HISTORY_DIR = PROJECT_ROOT / "data" / "benchmark_history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
TREND_FILE = HISTORY_DIR / "trend.json"

_logger = logging.getLogger(__name__)


# =============================================================================
# Benchmark runners
# =============================================================================

def _run_tool_decision(use_api: bool = False) -> dict:
    """P0: 工具决策 benchmark"""
    try:
        from src.tests.test_tool_decision_benchmark import run_benchmark, compute_metrics, BENCHMARK_CASES
        results = run_benchmark(use_api=use_api)
        metrics = compute_metrics(results)
        return {
            "case_count": len(BENCHMARK_CASES),
            "precision": round(metrics["precision"], 3),
            "recall": round(metrics["recall"], 3),
            "f1": round(metrics["f1"], 3),
            "accuracy": round(metrics["accuracy"], 3),
            "passed": metrics["passed"],
            "total": metrics["total"],
            "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"], "tn": metrics["tn"],
        }
    except Exception as e:
        return {"error": str(e), "case_count": 0}


def _run_reply_quality(use_api: bool = False) -> dict:
    """P2: 回复质量 benchmark"""
    try:
        from src.tests.test_reply_quality_benchmark import run_benchmark, BENCHMARK_CASES
        results = run_benchmark(use_api=use_api)
        passed = sum(1 for r in results if r.passed)
        total = len(results) if results else 0
        by_category = {}
        for r in results:
            by_category.setdefault(r.category, {"passed": 0, "total": 0})
            by_category[r.category]["total"] += 1
            if r.passed:
                by_category[r.category]["passed"] += 1
        return {
            "case_count": len(BENCHMARK_CASES),
            "passed": passed,
            "total": total,
            "pass_rate": round(passed / total, 3) if total > 0 else 0,
            "by_category": {k: round(v["passed"] / v["total"], 3) if v["total"] > 0 else 0
                          for k, v in by_category.items()},
            "rubric_count": sum(1 for r in results if r.evaluation_mode == "rubric"),
            "keyword_count": sum(1 for r in results if r.evaluation_mode == "keywords"),
        }
    except Exception as e:
        return {"error": str(e), "case_count": 0}


def _run_memory_search() -> dict:
    """P4: 记忆搜索 benchmark（无 API 依赖，始终实时运行）"""
    try:
        from src.tests.test_memory_search_benchmark import run_benchmark, compute_metrics, BENCHMARK_CASES
        results = run_benchmark()  # memory search 不需要 API
        metrics = compute_metrics(results)
        return {
            "case_count": len(BENCHMARK_CASES),
            "precision": round(metrics["precision"], 3),
            "recall": round(metrics["recall"], 3),
            "f1": round(metrics["f1"], 3),
            "accuracy": round(metrics["accuracy"], 3),
            "passed": metrics["passed"],
            "total": metrics["total"],
        }
    except Exception as e:
        return {"error": str(e), "case_count": 0}


def _run_unread_badge(use_api: bool = False) -> dict:
    """P5: 未读角标 benchmark"""
    try:
        from src.tests.test_chat_list_unread_benchmark import (
            _load_fixture_cases, _read_ground_truth, _read_cached_api_result,
            FIXTURE_DIR,
        )
        # 如果 use_api，需要调用 SmartPerceptionPipeline
        cases = _load_fixture_cases()
        if not cases:
            return {"error": "no fixture cases", "case_count": 0}

        tp = fp = tn = fn = 0
        skipped = 0
        for case_dir in cases:
            gt = _read_ground_truth(case_dir)
            if use_api:
                # API 模式需要实际调用 qwen 多模态 API
                from src.perception.smart_pipeline import _QwenAPIClient
                client = _QwenAPIClient()
                screenshot = case_dir / "screenshot.png"
                if not screenshot.exists():
                    skipped += 1
                    continue
                raw = client.recognize(str(screenshot))
                chat_list = raw.get("chat_list", [])
                # 找 target_nickname 对应的项
                target = gt.get("target_nickname", "")
                predicted_has_unread = False
                for item in chat_list:
                    if item.get("nickname", "") == target:
                        unread = item.get("unread_count", "")
                        predicted_has_unread = bool(unread and unread.isdigit() and int(unread) > 0)
                        break
            else:
                cached = _read_cached_api_result(case_dir)
                if cached is None:
                    skipped += 1
                    continue
                # 从缓存的 API 结果推断
                predicted_has_unread = bool(cached.get("unread_count", ""))

            gt_has = gt.get("has_unread", False)
            if gt_has and predicted_has_unread:
                tp += 1
            elif gt_has and not predicted_has_unread:
                fn += 1
            elif not gt_has and predicted_has_unread:
                fp += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "case_count": len(cases),
            "skipped": skipped,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
    except Exception as e:
        return {"error": str(e), "case_count": 0}


def _run_judge_quality(use_api: bool = False) -> dict:
    """Meta: Judge 质量 benchmark v2。"""
    try:
        os.environ["RUN_PRODUCTION_BENCHMARKS"] = "1"
        from src.tests.test_judge_quality_benchmark_v2 import (
            BENCHMARK_CASES,
            _compute_metrics,
            run_benchmark,
        )
        case_count = len(BENCHMARK_CASES)
        if case_count == 0:
            return {
                "error": "Judge Quality v2 不可运行：没有私有固定 GT 或数据库人工标签",
                "case_count": 0,
                "status": "unavailable",
            }
        results = run_benchmark(use_api=use_api)
        if len(results) != case_count:
            return {
                "error": f"Judge Quality v2 缓存不完整：{len(results)}/{case_count} cases 可用，请使用 --run-api 生成缓存",
                "case_count": case_count,
                "status": "unavailable",
            }
        metrics = _compute_metrics(results)
        passed = sum(1 for r in results if r.passed)
        total = len([r for r in results if not r.error])
        return {
            "case_count": case_count,
            "precision": round(metrics["precision"], 3),
            "recall": round(metrics["recall"], 3),
            "f1": round(metrics["f1"], 3),
            "accuracy": round(metrics["accuracy"], 3),
            "passed": passed,
            "total": total,
            "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"], "tn": metrics["tn"],
        }
    except Exception as e:
        return {"error": str(e), "case_count": 0}


def _run_ocr_quality(use_api: bool = False) -> dict:
    """P1: 私有真实 OCR 质量 benchmark。"""
    try:
        from src.benchmarks.ocr_quality import build_summary, load_cases, run_benchmark

        cases = load_cases()
        if not cases:
            return {
                "error": "私有 OCR benchmark 不可运行：data/private_benchmarks/ocr/fixtures 中没有 case",
                "case_count": 0,
                "status": "unavailable",
            }
        results, missing_cache = run_benchmark(use_api=use_api)
        summary = build_summary(cases, results, missing_cache)
        if summary["status"] != "available":
            return {
                "error": f"私有 OCR benchmark 缓存不完整：{len(results)}/{len(cases)} cases 可用",
                "case_count": len(cases),
                "status": "unavailable",
            }
        representative = summary["cohorts"]["representative"]
        regression = summary["cohorts"]["regression"]
        fields = summary["field_metrics_all"]
        return {
            "case_count": len(cases),
            "representative_cases": representative["total"],
            "representative_pass_rate": round(representative["pass_rate"], 3),
            "regression_cases": regression["total"],
            "regression_pass_rate": round(regression["pass_rate"], 3),
            "chat_name_accuracy": round(fields["chat_name_accuracy"], 3),
            "message_count_accuracy": round(fields["message_count_accuracy"], 3),
            "sender_accuracy": round(fields["sender_accuracy"], 3),
            "text_accuracy": round(fields["text_accuracy"], 3),
            "private_benchmark": True,
        }
    except Exception as e:
        return {"error": str(e), "case_count": 0}


# =============================================================================
# History management
# =============================================================================

def _load_history(days: int = 30) -> list[dict]:
    """加载最近 N 天的历史数据。"""
    records = []
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    for f in sorted(HISTORY_DIR.glob("*.json")):
        if f.name == "trend.json":
            continue
        date_str = f.stem  # YYYY-MM-DD
        if date_str < cutoff:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["date"] = date_str
            records.append(data)
        except Exception as e:
            _logger.warning("load history file failed: %s", e)
    return records


def _save_snapshot(metrics: dict):
    """保存当日指标快照。"""
    today = datetime.now().strftime("%Y-%m-%d")
    snapshot = {
        "date": today,
        "timestamp": datetime.now().isoformat(),
        "benchmarks": metrics,
        "git_commit": _get_git_commit(),
    }
    path = HISTORY_DIR / f"{today}.json"
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 已保存: {path}")

    # 同步到数据库
    try:
        from src.badcase.case_db import get_db
        db = get_db()
        db.insert_daily_metrics(
            date=today,
            benchmarks=metrics,
            git_commit=snapshot.get("git_commit", ""),
        )
        print("💾 已同步到 SQLite: data/cases.db")
    except Exception as e:
        print(f"  DB 同步跳过: {e}")

    # 更新趋势文件
    _update_trend(snapshot)


def _get_git_commit() -> str:
    """获取当前 git commit hash。"""
    try:
        import subprocess
        r = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, timeout=5
        )
        return r.stdout.decode().strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _update_trend(snapshot: dict):
    """更新趋势文件（只保留每个 benchmark 的关键指标时间序列）。"""
    if TREND_FILE.exists():
        trend = json.loads(TREND_FILE.read_text(encoding="utf-8"))
    else:
        trend = {"series": {}, "updated": ""}

    date = snapshot["date"]
    for name, bench in snapshot.get("benchmarks", {}).items():
        if name not in trend["series"]:
            trend["series"][name] = {"dates": [], "metrics": {}}
        series = trend["series"][name]

        # 去重同一天
        if date in series["dates"]:
            continue

        series["dates"].append(date)
        # 只保留最近 90 天的数据点
        if len(series["dates"]) > 90:
            series["dates"] = series["dates"][-90:]

        for key, value in bench.items():
            if isinstance(value, (int, float)):
                if key not in series["metrics"]:
                    series["metrics"][key] = []
                series["metrics"][key].append(value)
                if len(series["metrics"][key]) > 90:
                    series["metrics"][key] = series["metrics"][key][-90:]

    trend["updated"] = datetime.now().isoformat()
    TREND_FILE.write_text(json.dumps(trend, ensure_ascii=False, indent=2), encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================

def run_all(use_api: bool = False) -> dict:
    """运行所有 benchmark，返回指标字典。"""
    benchmarks = {}

    print("=" * 60)
    print(f"📊 每日 Benchmark 指标采集 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   模式: {'API 调取' if use_api else '缓存回归'}")
    print("=" * 60)

    t0 = time.time()

    print("\n[1/6] P0 工具决策...")
    benchmarks["tool_decision"] = _run_tool_decision(use_api)
    _print_single("工具决策", benchmarks["tool_decision"])

    print("\n[2/6] P1 OCR 质量...")
    benchmarks["ocr_quality"] = _run_ocr_quality(use_api)
    _print_single("OCR 质量", benchmarks["ocr_quality"])

    print("\n[3/6] P2 回复质量...")
    benchmarks["reply_quality"] = _run_reply_quality(use_api)
    _print_single("回复质量", benchmarks["reply_quality"])

    print("\n[4/6] P4 记忆搜索...")
    benchmarks["memory_search"] = _run_memory_search()
    _print_single("记忆搜索", benchmarks["memory_search"])

    print("\n[5/6] P5 未读角标...")
    benchmarks["unread_badge"] = _run_unread_badge(use_api)
    _print_single("未读角标", benchmarks["unread_badge"])

    print("\n[6/6] Judge 质量...")
    benchmarks["judge_quality"] = _run_judge_quality(use_api)
    _print_single("Judge 质量", benchmarks["judge_quality"])

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"✅ 全部完成 — 耗时 {elapsed:.1f}s")
    print(f"{'=' * 60}")

    return benchmarks


def _print_single(name: str, bench: dict):
    """打印单个 benchmark 的核心指标。"""
    if "error" in bench:
        print(f"  ❌ {bench['error']}")
        return

    parts = []
    if "accuracy" in bench:
        parts.append(f"Acc={bench['accuracy']:.1%}")
    if "pass_rate" in bench:
        parts.append(f"Pass={bench['pass_rate']:.1%}")
    if "precision" in bench:
        parts.append(f"Pre={bench['precision']:.1%}")
    if "recall" in bench:
        parts.append(f"Rec={bench['recall']:.1%}")
    if "f1" in bench:
        parts.append(f"F1={bench['f1']:.3f}")
    if "sender_accuracy" in bench:
        parts.append(f"Sender={bench['sender_accuracy']:.1%}")
    if "text_accuracy" in bench:
        parts.append(f"Text={bench['text_accuracy']:.1%}")
    if "representative_pass_rate" in bench:
        parts.append(f"Representative={bench['representative_pass_rate']:.1%}")
    if "regression_pass_rate" in bench:
        parts.append(f"Regression={bench['regression_pass_rate']:.1%}")

    status = "✅" if bench.get("error") is None else "⚠️"
    print(f"  {status} {bench.get('case_count', '?')} cases | {'  '.join(parts)}")


# =============================================================================
# HTML 趋势报告
# =============================================================================

def generate_html_report():
    """生成 trend_report.html — 可视化趋势。"""
    records = _load_history(days=90)
    if not records:
        print("暂无历史数据，先跑一次 benchmark 吧")
        return

    # 收集所有 benchmark 名称和指标
    bench_names = set()
    metric_names = set()
    for r in records:
        for bname, bdata in r.get("benchmarks", {}).items():
            if "error" in bdata:
                continue
            bench_names.add(bname)
            for key, val in bdata.items():
                if isinstance(val, (int, float)) and key not in ("tp", "fp", "fn", "tn", "passed", "total", "skipped", "case_count"):
                    metric_names.add(key)

    bench_names = sorted(bench_names)

    # 构建 Chart.js 数据
    dates_json = json.dumps([r["date"] for r in records])

    datasets = []
    colors = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff", "#79c0ff"]
    color_idx = 0

    for bname in bench_names:
        for mname in sorted(metric_names):
            values = []
            for r in records:
                b = r.get("benchmarks", {}).get(bname, {})
                if "error" in b or mname not in b:
                    values.append(None)
                else:
                    values.append(b[mname])
            if any(v is not None for v in values):
                datasets.append({
                    "label": f"{bname}/{mname}",
                    "data": values,
                    "borderColor": colors[color_idx % len(colors)],
                    "backgroundColor": "transparent",
                    "tension": 0.2,
                    "spanGaps": True,
                })
                color_idx += 1

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Benchmark Trend Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#c9d1d9;--muted:#8b949e;--green:#3fb950;--red:#f85149;--yellow:#d29922;--blue:#58a6ff}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:var(--bg);color:var(--text);padding:24px;max-width:1200px;margin:0 auto}}
h1{{text-align:center;margin-bottom:8px}}
.subtitle{{text-align:center;color:var(--muted);margin-bottom:24px;font-size:14px}}
.chart-container{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:20px;margin-bottom:20px}}
canvas{{max-height:320px}}
.summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:24px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px}}
.card h3{{font-size:12px;color:var(--muted);margin-bottom:8px;text-transform:uppercase}}
.card .value{{font-size:28px;font-weight:700}}
.card .value.green{{color:var(--green)}}
.card .value.yellow{{color:var(--yellow)}}
.card .value.red{{color:var(--red)}}
.card .trend{{font-size:12px;margin-top:4px}}
.card .date{{font-size:11px;color:var(--muted);margin-top:2px}}
.footer{{text-align:center;color:var(--muted);font-size:12px;padding-top:24px;border-top:1px solid var(--border)}}
</style>
</head>
<body>
<h1>📊 Benchmark 指标趋势</h1>
<div class="subtitle">
  {len(records)} 天数据 · {records[0]["date"]} ~ {records[-1]["date"]} · 下次调度: 每天 03:17
</div>

<div class="summary" id="summary"></div>

<div class="chart-container">
  <canvas id="trendChart"></canvas>
</div>

<div class="footer">
  数据来源: data/benchmark_history/ · <code>python scripts/run_daily_benchmark.py --html</code> 生成
</div>

<script>
const dates = {dates_json};
const datasets = {json.dumps(datasets, ensure_ascii=False)};

// 趋势图
const ctx = document.getElementById('trendChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{ labels: dates, datasets: datasets }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ color: '#8b949e', boxWidth: 12, padding: 10, font: {{size: 11}} }} }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', maxTicksLimit: 15 }} }},
      y: {{ min: 0, max: 1, ticks: {{ color: '#8b949e', callback: v => (v*100).toFixed(0) + '%' }} }}
    }}
  }}
}});

// 最新指标摘要
const latest = {json.dumps(records[-1] if records else {{}}, ensure_ascii=False)};
const summaryDiv = document.getElementById('summary');
const benches = latest.benchmarks || {{}};
for (const [name, data] of Object.entries(benches)) {{
  if (data.error) continue;
  const card = document.createElement('div');
  card.className = 'card';
  const parts = [];
  if (data.accuracy !== undefined) parts.push('Acc: ' + (data.accuracy*100).toFixed(0) + '%');
  if (data.precision !== undefined) parts.push('Pre: ' + (data.precision*100).toFixed(0) + '%');
  if (data.recall !== undefined) parts.push('Rec: ' + (data.recall*100).toFixed(0) + '%');
  card.innerHTML = '<h3>' + name + '</h3><div class="value green">' + parts.join(' ') + '</div><div class="date">' + (data.case_count || '?') + ' cases</div>';
  summaryDiv.appendChild(card);
}}
</script>
</body>
</html>"""

    report_path = PROJECT_ROOT / "trend_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"\n📈 趋势报告已生成: {report_path}")


def print_history_report(days: int = 14):
    """打印命令行历史趋势报告。"""
    records = _load_history(days=days)
    if not records:
        print("暂无历史数据")
        return

    print(f"\n📊 Benchmark 历史趋势（最近 {len(records)} 天）\n")

    # 收集所有指标名
    metric_keys = {}
    for r in records:
        for bname, bdata in r.get("benchmarks", {}).items():
            if "error" in bdata:
                continue
            for key, val in bdata.items():
                if isinstance(val, (int, float)) and key not in ("tp", "fp", "fn", "tn", "passed", "total", "skipped", "case_count"):
                    metric_keys.setdefault(bname, set()).add(key)

    for bname in sorted(metric_keys.keys()):
        print(f"  {bname}:")
        for mname in sorted(metric_keys[bname]):
            values = []
            for r in records:
                b = r.get("benchmarks", {}).get(bname, {})
                if "error" not in b and mname in b:
                    values.append(b[mname])
            if not values:
                continue
            latest = values[-1]
            avg = sum(values) / len(values)
            first = values[0]
            delta = latest - first
            arrow = "↑" if delta > 0.01 else "↓" if delta < -0.01 else "→"
            print(f"    {mname}: 当前={latest:.1%}  均值={avg:.1%}  变化={arrow}{abs(delta):.1%}")


# =============================================================================
# Main
# =============================================================================

def _git_changed_since_last_run() -> bool:
    """检查 git commit 是否与上次保存的快照不同。"""
    last_snapshot = _latest_snapshot()
    if not last_snapshot:
        return True  # 首次运行
    last_commit = last_snapshot.get("git_commit", "")
    current_commit = _get_git_commit()
    if not last_commit or not current_commit:
        return True
    return last_commit != current_commit


def _latest_snapshot() -> dict | None:
    """读取最近一次的快照。"""
    files = sorted(HISTORY_DIR.glob("*.json"))
    if not files:
        return None
    # 跳过 trend.json
    data_files = [f for f in files if f.name != "trend.json"]
    if not data_files:
        return None
    try:
        return json.loads(data_files[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


def _print_compare(metrics: dict):
    """与上次结果对比，打印变化。"""
    prev = _latest_snapshot()
    if not prev or prev.get("date") == datetime.now().strftime("%Y-%m-%d"):
        print("\n(首次运行或今日已采集，无历史对比)")
        return

    prev_benches = prev.get("benchmarks", {})
    print(f"\n📊 对比上次 ({prev.get('date', '?')}, {prev.get('git_commit', '?')[:8]}):")
    changed = False
    for name in sorted(metrics.keys()):
        cur = metrics[name]
        old = prev_benches.get(name, {})
        if "error" in cur or "error" in old:
            continue
        deltas = []
        for key in ("accuracy", "pass_rate", "precision", "recall", "f1"):
            if key in cur and key in old:
                d = cur[key] - old[key]
                if abs(d) >= 0.005:
                    arrow = "↑" if d > 0 else "↓"
                    deltas.append(f"{key}={arrow}{abs(d):.1%}")
                    changed = True
        if deltas:
            print(f"  {name}: {'  '.join(deltas)}")
    if not changed:
        print("  (指标无显著变化)")


def main():
    parser = argparse.ArgumentParser(description="每日 Benchmark 指标采集 — 代码变更时自动重跑 API")
    parser.add_argument("--run-api", action="store_true", help="强制调用真实 LLM API（默认自动检测 git 变化）")
    parser.add_argument("--report", type=int, nargs="?", const=14, help="打印历史趋势（默认 14 天）")
    parser.add_argument("--html", action="store_true", help="生成 trend_report.html")
    parser.add_argument("--force", action="store_true", help="强制重新采集，即使 git 未变化")
    args = parser.parse_args()

    if args.report:
        print_history_report(days=args.report)
        return

    if args.html:
        generate_html_report()
        return

    # 自动检测：git 有变化 → 用 --run-api 重新生成缓存
    use_api = args.run_api
    if not use_api and not args.force:
        if _git_changed_since_last_run():
            print("🔍 检测到代码变更，自动启用 --run-api 重新生成缓存\n")
            use_api = True
        else:
            print("📦 代码未变化，使用缓存回归\n")

    metrics = run_all(use_api=use_api)
    _save_snapshot(metrics)
    _print_compare(metrics)
    generate_html_report()


if __name__ == "__main__":
    main()
