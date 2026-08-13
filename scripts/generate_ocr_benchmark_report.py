#!/usr/bin/env python3
"""生成私有真实 OCR benchmark 报告。"""

import argparse
import html
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.ocr_quality import build_summary, load_cases, run_benchmark


def _rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def generate_html(
    output_path: Path = PROJECT_ROOT / "data" / "reports" / "ocr_benchmark_report.html",
    use_api: bool = False,
) -> dict:
    cases = load_cases()
    results, missing_cache = run_benchmark(use_api=use_api, cases=cases)
    summary = build_summary(cases, results, missing_cache)
    representative = summary["cohorts"]["representative"]
    regression = summary["cohorts"]["regression"]
    fields = summary["field_metrics_all"]

    rows = []
    for result in sorted(results, key=lambda item: (item.cohort, item.case_name)):
        rows.append(
            "<tr>"
            f"<td>{html.escape(result.case_name)}</td>"
            f"<td>{html.escape(result.cohort)}</td>"
            f"<td>{'PASS' if result.passed else 'FAIL'}</td>"
            f"<td>{'✓' if result.chat_name_match else '✗'}</td>"
            f"<td>{'✓' if result.message_count_match else '✗'}</td>"
            f"<td>{result.sender_accuracy:.0%}</td>"
            f"<td>{result.text_accuracy:.0%}</td>"
            "</tr>"
        )

    unavailable = ""
    if summary["status"] != "available":
        unavailable = (
            '<div class="warning">当前结果不可完整报告：'
            f'{len(results)}/{len(cases)} cases 有缓存。'
            '使用 <code>--run-api</code> 生成缺失缓存。</div>'
        )

    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Private OCR Benchmark</title>
<style>
body{{font-family:-apple-system,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#202124}}
.note,.warning{{padding:12px 16px;border-radius:8px;margin:16px 0}}.note{{background:#eef6ff}}.warning{{background:#fff4e5}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:20px 0}}
.card{{border:1px solid #ddd;border-radius:10px;padding:16px}}.value{{font-size:28px;font-weight:700;margin-top:8px}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}
code{{background:#f3f3f3;padding:2px 5px;border-radius:4px}}
</style></head><body>
<h1>私有真实 OCR Benchmark</h1>
<p>生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · 状态：{summary['status']}</p>
{unavailable}
<div class="note"><strong>指标解释：</strong>代表性场景和回归挑战场景分开统计。严格整 case 通过要求聊天名正确、消息数正确、sender 全部正确、text 正确率至少 80%。任何通过率都不是“文字 OCR 准确率”。真实截图、GT、缓存和本报告均位于 Git 忽略目录。</div>
<div class="cards">
<div class="card">代表性场景严格通过率<div class="value">{_rate(representative['pass_rate'])}</div><small>{representative['passed']}/{representative['total']}</small></div>
<div class="card">回归挑战恢复率<div class="value">{_rate(regression['pass_rate'])}</div><small>{regression['passed']}/{regression['total']}</small></div>
<div class="card">Chat Name<div class="value">{_rate(fields['chat_name_accuracy'])}</div></div>
<div class="card">Message Count<div class="value">{_rate(fields['message_count_accuracy'])}</div></div>
<div class="card">Sender 平均<div class="value">{_rate(fields['sender_accuracy'])}</div></div>
<div class="card">Text 平均<div class="value">{_rate(fields['text_accuracy'])}</div></div>
</div>
<h2>逐 case 结果</h2>
<table><thead><tr><th>Case</th><th>分层</th><th>严格结果</th><th>名称</th><th>数量</th><th>Sender</th><th>Text</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    print(f"报告已生成: {output_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="私有真实 OCR benchmark 报告")
    parser.add_argument("--run-api", action="store_true", help="调用真实 API 并刷新缓存")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "reports" / "ocr_benchmark_report.html",
    )
    args = parser.parse_args()
    generate_html(args.output, use_api=args.run_api)


if __name__ == "__main__":
    main()
