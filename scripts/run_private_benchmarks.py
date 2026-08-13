#!/usr/bin/env python3
"""运行私有 benchmark，并生成带来源信息的 JSON/HTML 报告。"""

import argparse
import html
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.private_runner import (  # noqa: E402
    PRIVATE_ROOT,
    run_private_benchmarks,
)


def _value(value):
    return "—" if value is None else str(value)


def _rate(value):
    return "—" if value is None else f"{value:.1%}"


def render_html(report: dict) -> str:
    ocr = report["benchmarks"]["ocr_quality"]
    judge = report["benchmarks"]["judge_quality_v2"]
    candidates = report["review_candidates"]
    representative = ocr["summary"]["cohorts"]["representative"]
    regression = ocr["summary"]["cohorts"]["regression"]
    judge_metrics = judge["metrics"] or {}
    candidate_rows = "".join(
        "<tr>"
        f"<td>{item['db_id']}</td>"
        f"<td>{html.escape(str(item['created_at']))}</td>"
        f"<td>{item['priority']}</td>"
        f"<td>{html.escape(', '.join(item['reasons']))}</td>"
        f"<td>{html.escape(item['chat_ref'])}</td>"
        "</tr>"
        for item in candidates["items"]
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Private Benchmark Report</title>
<style>
body{{font-family:-apple-system,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#202124}}
.note{{padding:12px 16px;background:#eef6ff;border-radius:8px;margin:16px 0}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:20px 0}}
.card{{border:1px solid #ddd;border-radius:10px;padding:16px}}.value{{font-size:27px;font-weight:700;margin-top:8px}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}
code{{background:#f3f3f3;padding:2px 5px;border-radius:4px}}
</style></head><body>
<h1>私有 Benchmark 自动报告</h1>
<p>生成时间：{html.escape(report['generated_at'])} · 模式：{html.escape(report['mode'])}</p>
<div class="note">真实数据、GT、缓存和本报告均位于 Git 忽略目录。常规集与挑战集分开显示；缓存不完整时不计算准确率。</div>
<h2>评测结果</h2>
<div class="cards">
<div class="card">OCR 代表性场景<div class="value">{_rate(representative['pass_rate'])}</div><small>{representative['passed']}/{representative['total']} · {ocr['status']}</small></div>
<div class="card">OCR 回归挑战<div class="value">{_rate(regression['pass_rate'])}</div><small>{regression['passed']}/{regression['total']}</small></div>
<div class="card">Judge Quality v2<div class="value">{_rate(judge_metrics.get('accuracy'))}</div><small>{judge['evaluated_cases']}/{judge['configured_cases']} · {judge['status']}</small></div>
<div class="card">机器候选池<div class="value">{candidates['candidate_pool']}</div><small>本报告展示 {candidates['count']} 条</small></div>
</div>
<h2>来源留痕</h2>
<p>Git：<code>{html.escape(report['provenance']['commit'][:12])}</code> · Dirty：{_value(report['provenance']['dirty'])} · Report：<code>{report['report_fingerprint']}</code></p>
<p>OCR 数据集：<code>{ocr['dataset_fingerprint']}</code> · Judge 数据集：<code>{judge['dataset_fingerprint']}</code></p>
<h2>OCR 自动归因</h2>
<p>{html.escape(json.dumps(ocr['failure_triage']['by_reason'], ensure_ascii=False))}</p>
<h2>机器筛选的 Review 候选</h2>
<p>{html.escape(candidates['privacy'])}</p>
<table><thead><tr><th>DB ID</th><th>时间</th><th>优先级</th><th>原因</th><th>Chat Ref</th></tr></thead><tbody>{candidate_rows}</tbody></table>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="私有 benchmark 自动运行器")
    parser.add_argument(
        "--refresh",
        choices=("none", "ocr", "judge", "all"),
        default="none",
        help="显式刷新指定 API 缓存；默认只读现有缓存",
    )
    parser.add_argument("--ocr-model", default="qwen3.6-flash")
    parser.add_argument("--judge-runs", type=int, default=3)
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PRIVATE_ROOT / "reports" / "latest.json",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=PRIVATE_ROOT / "reports" / "latest.html",
    )
    args = parser.parse_args()

    report = run_private_benchmarks(
        refresh=args.refresh,
        ocr_model=args.ocr_model,
        judge_runs=args.judge_runs,
        candidate_limit=args.candidate_limit,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.html_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.html_output.write_text(render_html(report), encoding="utf-8")
    print(f"JSON: {args.json_output}")
    print(f"HTML: {args.html_output}")
    print(f"OCR: {report['benchmarks']['ocr_quality']['status']}")
    judge = report["benchmarks"]["judge_quality_v2"]
    print(f"Judge: {judge['status']} ({judge['evaluated_cases']}/{judge['configured_cases']})")
    print(f"Review candidates: {report['review_candidates']['candidate_pool']}")


if __name__ == "__main__":
    main()
