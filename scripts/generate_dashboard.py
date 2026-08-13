#!/usr/bin/env python3
"""
生成 Benchmark Dashboard HTML — 包含指标趋势图、每日变化、出错的 case 明细。

用法:
    python scripts/generate_dashboard.py              # 从已有 history 生成
    python scripts/generate_dashboard.py --open       # 生成并打开浏览器
"""

import json
import sys
from datetime import datetime
from pathlib import Path
import logging


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

HISTORY_DIR = PROJECT_ROOT / "data" / "benchmark_history"
REPORT_PATH = PROJECT_ROOT / "benchmark_dashboard.html"

_logger = logging.getLogger(__name__)


def _load_history(days: int = 90) -> list[dict]:
    records = []
    for f in sorted(HISTORY_DIR.glob("*.json")):
        if f.name == "trend.json":
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["date"] = f.stem
            records.append(data)
        except Exception as e:
            _logger.warning("load history file failed: %s", e)
    if days and len(records) > days:
        records = records[-days:]
    return records


def _run_failing_cases() -> dict:
    """采集各个 benchmark 的具体分歧 case，包含上下文。"""
    failures: dict[str, list[dict]] = {}

    # P0: Tool Decision
    try:
        from src.tests.test_tool_decision_benchmark import run_benchmark, BENCHMARK_CASES
        case_map = {c.case_name: c for c in BENCHMARK_CASES}
        results = run_benchmark(use_api=False)
        for r in results:
            if not r.passed and not getattr(r, 'error', ''):
                bc = case_map.get(r.case_name)
                failures.setdefault("tool_decision", []).append({
                    "case_name": r.case_name,
                    "should_call": r.should_call,
                    "actually_called": r.actually_called,
                    "called_tools": r.called_tools,
                    "category": r.category,
                    "user_message": bc.user_message if bc else "",
                    "notes": bc.notes if bc else "",
                    "evaluation_mode": r.evaluation_mode,
                    "rubric_scores": _simplify_rubric(r.rubric_scores) if r.rubric_scores else None,
                })
    except Exception as e:
        failures["tool_decision"] = [{"error": str(e)}]

    # P2: Reply Quality
    try:
        from src.tests.test_reply_quality_benchmark import run_benchmark, BENCHMARK_CASES
        case_map = {c.case_name: c for c in BENCHMARK_CASES}
        results = run_benchmark(use_api=False)
        for r in results:
            if not r.passed:
                bc = case_map.get(r.case_name)
                # 提取对话上下文
                context_lines = []
                if bc:
                    for m in bc.unreplied:
                        context_lines.append(f"[未读] {m.sender}: {m.text}")
                failures.setdefault("reply_quality", []).append({
                    "case_name": r.case_name,
                    "category": r.category,
                    "missing_keywords": r.missing_keywords,
                    "found_forbidden": r.found_forbidden,
                    "reply_count": r.reply_count,
                    "replies": r.replies[:3] if len(r.replies) > 3 else r.replies,
                    "context": context_lines,
                    "notes": bc.notes if bc else "",
                    "evaluation_mode": r.evaluation_mode,
                    "rubric_scores": _simplify_rubric(r.rubric_scores) if r.rubric_scores else None,
                })
    except Exception as e:
        failures["reply_quality"] = [{"error": str(e)}]

    # P4: Memory Search
    try:
        from src.tests.test_memory_search_benchmark import run_benchmark, BENCHMARK_CASES
        case_map = {c.case_name: c for c in BENCHMARK_CASES}
        results = run_benchmark()
        for r in results:
            if not r.passed:
                bc = case_map.get(r.case_name)
                failures.setdefault("memory_search", []).append({
                    "case_name": r.case_name,
                    "query": r.query if hasattr(r, 'query') else (bc.query if bc else ""),
                    "found_expected": r.found_expected,
                    "missed_expected": r.missed_expected,
                    "found_unexpected": r.found_unexpected,
                    "missing_fragments": r.missing_fragments,
                    "expected_docs": bc.expected_docs if bc else [],
                    "notes": bc.notes if bc else "",
                    "precision": r.precision,
                    "recall": r.recall,
                })
    except Exception as e:
        failures["memory_search"] = [{"error": str(e)}]

    # Meta: Judge Quality — 多维度评分（多轮平均）
    try:
        from src.tests.test_judge_quality_benchmark import run_benchmark, BENCHMARK_CASES
        import statistics
        import json as _json
        case_map = {c.case_name: c for c in BENCHMARK_CASES}
        results = run_benchmark(use_api=False, n_runs=3)
        for r in results:
            bc = case_map.get(r.case_name)
            context_lines = []
            bot_reply = ""
            if bc and bc.tick_data:
                for m in bc.tick_data.get("session_input_messages", []):
                    role = "Bot" if m.get("sender_type") == "self" else m.get("sender", "User")
                    context_lines.append(f"{role}: {m.get('text', '')}")
                bot_reply = bc.tick_data.get("bot_reply_text", "")

            # 读取多轮缓存，平均维度评分
            cache_path = Path(__file__).parent.parent / "src" / "tests" / "fixtures" / "judge_quality" / "cache" / f"{r.case_name}.json"
            avg_dims = {}
            dim_var = {}
            avg_score = 0
            score_std = 0
            n_runs = 1
            badcase_votes = 0
            if cache_path.exists():
                try:
                    cached = _json.loads(cache_path.read_text(encoding="utf-8"))
                    runs = cached.get("runs", [cached] if "runs" not in cached else [])
                    n_runs = len(runs)
                    badcase_votes = sum(1 for run in runs if run.get("is_badcase"))
                    scores = [float(run.get("overall_score", 0)) for run in runs]
                    avg_score = round(statistics.mean(scores), 1) if scores else 0
                    score_std = round(statistics.stdev(scores), 1) if len(scores) >= 2 else 0
                    # 平均各维度
                    dim_names = ["幻觉控制", "记忆召回", "幽默感", "逼格语气", "个性一致性", "简洁度", "上下文理解"]
                    for name in dim_names:
                        vals = [float(run.get("dimensions", {}).get(name, {}).get("score", 0)) for run in runs]
                        comment = runs[0].get("dimensions", {}).get(name, {}).get("comment", "")
                        avg_dims[name] = {"score": round(statistics.mean(vals), 1) if vals else 0, "comment": comment}
                        if len(vals) >= 2 and len(set(vals)) > 1:
                            dim_var[name] = round(statistics.stdev(vals), 2)
                except Exception as e:
                    _logger.warning("compute dimension stats failed: %s", e)

            entry = {
                "case_name": r.case_name,
                "ground_truth_is_badcase": r.ground_truth_is_badcase,
                "ground_truth_type": r.ground_truth_type,
                "predicted_is_badcase": r.predicted_is_badcase,
                "predicted_type": r.predicted_type,
                "predicted_confidence": r.predicted_confidence,
                "overall_score": avg_score,
                "overall_score_std": score_std,
                "dimensions": avg_dims,
                "dimension_variance": dim_var,
                "n_runs": n_runs,
                "badcase_votes": badcase_votes,
                "context": context_lines[-6:] if len(context_lines) > 6 else context_lines,
                "bot_reply": bot_reply,
                "notes": bc.notes if bc else "",
                "passed": r.passed,
            }
            failures.setdefault("judge_quality", []).append(entry)
    except Exception as e:
        failures["judge_quality"] = [{"error": str(e)}]

    return failures


def _collect_stability() -> list[dict]:
    """采集回复稳定性 benchmark 数据。"""
    cases = []
    try:
        from src.tests.test_reply_stability_benchmark import run_benchmark
        results = run_benchmark(use_api=False)
        cache_dir = Path(__file__).parent.parent / "src" / "tests" / "fixtures" / "reply_stability" / "cache"
        for r in results:
            if r.error:
                continue
            cache_path = cache_dir / f"{r.case_name}.json"
            replies = []
            reply_scores = []
            full_user_prompt = ""
            full_system_prompt = ""
            if cache_path.exists():
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    for run in cached.get("runs", []):
                        reply = run.get("reply", "")
                        judge = run.get("judge", {})
                        replies.append(reply)
                        reply_scores.append({
                            "reply": reply[:500],
                            "overall_score": judge.get("overall_score", 0),
                            "is_badcase": judge.get("is_badcase", False),
                            "dimensions": judge.get("dimensions", {}),
                            "tool_log": run.get("tool_log", []),
                        })
                        if not full_user_prompt:
                            full_user_prompt = run.get("user_prompt", "")
                        if not full_system_prompt:
                            full_system_prompt = run.get("system_prompt", "")
                except Exception as e:
                    _logger.warning("load run prompts failed: %s", e)
            cases.append({
                "case_name": r.case_name,
                "source": r.source,
                "notes": r.notes,
                "context": r.context_msgs,
                "full_user_prompt": full_user_prompt,
                "full_system_prompt": full_system_prompt,
                "replies": replies,
                "reply_scores": reply_scores,
                "n_generations": r.n_generations,
                "avg_overall_score": r.avg_overall_score,
                "overall_score_std": r.overall_score_std,
                "avg_dimensions": {k: v for k, v in r.avg_dimensions.items()} if r.avg_dimensions else {},
                "cross_similarity": r.cross_similarity,
            })
    except Exception as e:
        _logger.warning("load cases failed: %s", e)
    return cases


def _simplify_rubric(scores: dict) -> dict:
    """精简 rubric 结果，避免 JSON 过大。"""
    if not scores:
        return None
    dims = []
    for d in scores.get("dimensions", []):
        dims.append({"name": d.get("name", ""), "score": d.get("score", "FAIL"), "reason": d.get("reason", "")[:200]})
    return {"overall": scores.get("overall", "FAIL"), "dimensions": dims, "explanation": scores.get("explanation", "")[:300]}


def generate_dashboard():
    """生成完整的 Dashboard HTML。"""
    records = _load_history(days=90)
    failures = _run_failing_cases()
    stability_cases = _collect_stability()

    if not records:
        print("暂无历史数据，先运行 python scripts/run_daily_benchmark.py")
        return

    # 聚合所有指标
    all_metrics: dict[str, list[dict]] = {}
    for r in records:
        for bname, bdata in r.get("benchmarks", {}).items():
            if "error" in bdata:
                continue
            all_metrics.setdefault(bname, []).append({"date": r["date"], **bdata})

    # 计算最新值和变化
    summary_cards = []
    for bname in sorted(all_metrics.keys()):
        series = all_metrics[bname]
        latest = series[-1]
        if len(series) >= 2:
            prev = series[-2]
        else:
            prev = None

        for key in ("accuracy", "pass_rate", "precision", "recall", "f1", "sender_accuracy"):
            if key not in latest:
                continue
            cur = latest[key]
            label = key.replace("_", " ").title()
            delta = ""
            if prev and key in prev:
                d = cur - prev[key]
                if abs(d) >= 0.001:
                    arrow = "↑" if d > 0 else "↓"
                    delta = f"{arrow}{abs(d):.1%}"
            color = "green" if cur >= 0.85 else "yellow" if cur >= 0.7 else "red"
            summary_cards.append({
                "bench": bname,
                "metric": label,
                "value": f"{cur:.1%}",
                "delta": delta,
                "color": color,
                "cases": latest.get("case_count", "?"),
            })

    # 构建 Chart.js 数据
    dates = [r["date"] for r in records]
    dates_json = json.dumps(dates)

    datasets = []
    colors = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff", "#79c0ff", "#ffa657", "#a5d6ff"]
    ci = 0
    for bname in sorted(all_metrics.keys()):
        series = all_metrics[bname]
        for mname in ["accuracy", "f1", "precision", "recall", "pass_rate", "sender_accuracy"]:
            values = [s.get(mname) for s in series]
            if any(v is not None for v in values):
                datasets.append({
                    "label": f"{bname}/{mname}",
                    "data": values,
                    "borderColor": colors[ci % len(colors)],
                    "backgroundColor": "transparent",
                    "tension": 0.3,
                    "spanGaps": True,
                    "pointRadius": 3,
                })
                ci += 1

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    first_date = records[0]["date"]
    last_date = records[-1]["date"]

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Benchmark Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#c9d1d9;--muted:#8b949e;--green:#3fb950;--red:#f85149;--yellow:#d29922;--blue:#58a6ff;--purple:#bc8cff}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--text);padding:20px 24px;max-width:1400px;margin:0 auto}}
h1{{text-align:center;margin-bottom:4px;font-size:24px}}
.subtitle{{text-align:center;color:var(--muted);margin-bottom:20px;font-size:13px}}
.tabs{{display:flex;gap:8px;margin-bottom:20px;flex-wrap:wrap}}
.tab{{background:var(--card);border:1px solid var(--border);color:var(--text);padding:8px 16px;border-radius:8px;cursor:pointer;font-size:13px;transition:border-color .15s}}
.tab:hover{{border-color:var(--blue)}}
.tab.active{{border-color:var(--blue);color:var(--blue);font-weight:600}}
.section{{display:none}}
.section.active{{display:block}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;margin-bottom:24px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px 16px}}
.card .bench{{font-size:10px;color:var(--muted);text-transform:uppercase;margin-bottom:4px}}
.card .metric{{font-size:11px;color:var(--muted)}}
.card .value{{font-size:28px;font-weight:700;margin:4px 0}}
.card .value.green{{color:var(--green)}}
.card .value.yellow{{color:var(--yellow)}}
.card .value.red{{color:var(--red)}}
.card .delta{{font-size:12px}}
.card .delta.up{{color:var(--green)}}
.card .delta.down{{color:var(--red)}}
.card .cases{{font-size:11px;color:var(--muted);margin-top:2px}}
.chart-box{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:20px;margin-bottom:20px}}
.chart-box h3{{font-size:13px;color:var(--muted);margin-bottom:12px}}
canvas{{max-height:350px}}
.failures{{margin-bottom:20px}}
.failures h2{{font-size:15px;margin-bottom:10px;padding-bottom:6px;border-bottom:2px solid var(--border)}}
.fail-case{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px 16px;margin-bottom:8px;display:flex;gap:12px;align-items:flex-start}}
.fail-case .badge{{flex-shrink:0;width:8px;height:8px;border-radius:50%;margin-top:6px}}
.fail-case .badge.red{{background:var(--red)}}
.fail-case .badge.yellow{{background:var(--yellow)}}
.fail-case .body{{flex:1;min-width:0}}
.fail-case .name{{font-weight:600;font-size:13px;margin-bottom:4px}}
.fail-case .detail{{font-size:12px;color:var(--muted);line-height:1.5}}
.fail-case .detail .tag{{display:inline-block;padding:1px 6px;border-radius:4px;font-size:11px;margin-right:4px}}
.fail-case .detail .tag.green{{background:rgba(63,185,80,.15);color:var(--green)}}
.fail-case .detail .tag.red{{background:rgba(248,81,73,.15);color:var(--red)}}
.fail-case .detail .tag.blue{{background:rgba(88,166,255,.15);color:var(--blue)}}
.fail-case .detail .tag.purple{{background:rgba(188,140,255,.15);color:var(--purple)}}
.fail-case pre{{font-size:11px;color:var(--text);background:rgba(0,0,0,.2);padding:6px 8px;border-radius:4px;margin-top:6px;overflow-x:auto;white-space:pre-wrap;max-height:120px;overflow-y:auto}}
.footer{{text-align:center;color:var(--muted);font-size:11px;padding-top:16px;border-top:1px solid var(--border);margin-top:24px}}
</style>
</head>
<body>

<h1>📊 Benchmark Dashboard</h1>
<div class="subtitle">
  {first_date} ~ {last_date} · {len(records)} 天数据 · 下次调度: 每天 03:17 · 生成于 {now}
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('overview')">📈 概览</div>
  <div class="tab" onclick="switchTab('trends')">📉 趋势图</div>
  <div class="tab" onclick="switchTab('failures')">⚡ 人机分歧 ({sum(len(v) for v in failures.values())})</div>
  <div class="tab" onclick="switchTab('stability')">🔁 回复稳定性 ({len(stability_cases)})</div>
</div>

<!-- ========== TAB 1: 概览 ========== -->
<div class="section active" id="section-overview">
  <div class="metrics">
    {_render_summary_cards(summary_cards)}
  </div>
</div>

<!-- ========== TAB 2: 趋势图 ========== -->
<div class="section" id="section-trends">
  <div class="chart-box">
    <h3>全部指标趋势 (0% = 0, 100% = 1.0)</h3>
    <canvas id="trendChart"></canvas>
  </div>
</div>

<!-- ========== TAB 3: 分歧 Case ========== -->
<div class="section" id="section-failures">
  {_render_failures(failures)}
</div>

<!-- ========== TAB 4: 回复稳定性 ========== -->
<div class="section" id="section-stability">
  {_render_stability(stability_cases)}
</div>

<div class="footer">
  数据来源: data/benchmark_history/ · <code>python scripts/run_daily_benchmark.py</code> 采集 · <code>python scripts/generate_dashboard.py</code> 生成
</div>

<script>
function switchTab(name) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelector('.tab[onclick*="' + name + '"]').classList.add('active');
  document.getElementById('section-' + name).classList.add('active');
}}

const dates = {dates_json};
const datasets = {json.dumps(datasets, ensure_ascii=False)};

const ctx = document.getElementById('trendChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{ labels: dates, datasets: datasets }},
  options: {{
    responsive: true,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ color: '#8b949e', boxWidth: 10, padding: 8, font: {{size: 10}}, usePointStyle: true }} }},
      tooltip: {{ callbacks: {{ label: ctx => ctx.dataset.label + ': ' + (ctx.parsed.y * 100).toFixed(1) + '%' }} }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', maxTicksLimit: 20, font: {{size: 10}} }} }},
      y: {{ min: 0, max: 1, ticks: {{ color: '#8b949e', callback: v => (v*100).toFixed(0) + '%', font: {{size: 10}} }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"✅ Dashboard 已生成: {REPORT_PATH}")


def _render_summary_cards(cards: list) -> str:
    """渲染指标卡片 HTML。"""
    lines = []
    for c in cards:
        delta_cls = "up" if "↑" in c["delta"] else "down" if "↓" in c["delta"] else ""
        lines.append(f"""<div class="card">
  <div class="bench">{c["bench"]}</div>
  <div class="metric">{c["metric"]}</div>
  <div class="value {c["color"]}">{c["value"]}</div>
  <div class="delta {delta_cls}">{c["delta"]}</div>
  <div class="cases">{c["cases"]} cases</div>
</div>""")
    return "\n    ".join(lines)


def _render_failures(failures: dict) -> str:
    """渲染分歧 case HTML，包含完整上下文。"""
    if not failures or all(len(v) == 0 for v in failures.values()):
        return '<p style="color:var(--muted);text-align:center;padding:40px">✅ 所有 case 通过，没有分歧项</p>'

    parts = []
    for bname, cases in sorted(failures.items()):
        if not cases:
            continue
        bname_labels = {
            "tool_decision": "P0 Tool Decision",
            "reply_quality": "P2 Reply Quality",
            "memory_search": "P4 Memory Search",
            "judge_quality": "Judge Quality",
        }
        label = bname_labels.get(bname, bname)
        parts.append(f"""<div class="failures">
  <h2>{label} — {len(cases)} 分歧</h2>""")

        for c in cases:
            if "error" in c:
                parts.append(f'  <div class="fail-case"><div class="badge red"></div><div class="body"><div class="detail" style="color:var(--red)">采集分歧: {c["error"]}</div></div></div>')
                continue

            detail_parts = []
            context_lines = c.get("context", [])
            user_msg = c.get("user_message", "")
            notes = c.get("notes", "")
            bot_reply = c.get("bot_reply", "")

            # 通用：显示备注/预期行为
            if notes:
                detail_parts.append(f'<div style="font-size:11px;color:var(--yellow);margin-bottom:4px">📌 预期: {_e(notes[:200])}</div>')

            # 显示用户消息
            if user_msg:
                detail_parts.append(f'<div style="font-size:12px;margin-bottom:4px">💬 <b>用户消息:</b> {_e(user_msg[:300])}</div>')

            # 显示 Bot 回复
            if bot_reply:
                detail_parts.append(f'<div style="font-size:12px;margin-bottom:4px">🤖 <b>Bot 回复:</b> {_e(bot_reply[:300])}</div>')

            # 显示对话上下文
            if context_lines:
                ctx_html = "<br>".join(_e(line[:200]) for line in context_lines[-8:])
                detail_parts.append(f'<pre style="font-size:11px;color:var(--muted);background:rgba(0,0,0,.2);padding:8px;border-radius:4px;margin:6px 0;max-height:200px;overflow-y:auto;white-space:pre-wrap;line-height:1.4">{ctx_html}</pre>')

            # 显示 replies
            replies = c.get("replies", [])
            if replies:
                replies_str = " | ".join(str(r)[:200] for r in replies)
                detail_parts.append(f'<div style="font-size:12px;margin-bottom:4px">📤 <b>生成回复:</b> {_e(replies_str[:500])}</div>')

            # Benchmark 特有的指标标签
            if bname == "tool_decision":
                should = "应调用" if c["should_call"] else "不应调用"
                actual = "调用了" if c["actually_called"] else "未调用"
                detail_parts.append(f'<span class="tag blue">{should}</span>')
                detail_parts.append(f'<span class="tag {"green" if c["should_call"] == c["actually_called"] else "red"}">实际: {actual}</span>')
                if c["called_tools"]:
                    detail_parts.append(f'<span class="tag purple">已调用: {", ".join(c["called_tools"])}</span>')
                if c.get("rubric_scores"):
                    rs = c["rubric_scores"]
                    detail_parts.append(f'<span class="tag red">Rubric: {rs.get("overall", "?")}</span>')
                    for d in rs.get("dimensions", []):
                        detail_parts.append(f'<span class="tag {"green" if d["score"]=="PASS" else "red"}">{d["name"]}: {d["score"]}</span>')
            elif bname == "reply_quality":
                if c.get("missing_keywords"):
                    detail_parts.append(f'<span class="tag red">缺关键词: {", ".join(c["missing_keywords"])}</span>')
                if c.get("found_forbidden"):
                    detail_parts.append(f'<span class="tag red">含禁用词: {", ".join(c["found_forbidden"])}</span>')
                if c.get("rubric_scores"):
                    rs = c["rubric_scores"]
                    for d in rs.get("dimensions", []):
                        score_cls = "green" if d["score"] == "PASS" else "red"
                        detail_parts.append(f'<span class="tag {score_cls}">{d["name"]}: {d["score"]}</span>')
                        if d["score"] != "PASS":
                            detail_parts.append(f'<div style="font-size:11px;color:var(--muted);margin:2px 0">  ↳ {_e(d.get("reason", "")[:200])}</div>')
            elif bname == "memory_search":
                detail_parts.append(f'<span class="tag blue">查询: {_e(c.get("query", ""))}</span>')
                if c.get("expected_docs"):
                    detail_parts.append(f'<span class="tag blue">期望文档: {", ".join(c["expected_docs"])}</span>')
                if c.get("missed_expected"):
                    detail_parts.append(f'<span class="tag red">未召回: {", ".join(c["missed_expected"])}</span>')
                if c.get("found_unexpected"):
                    detail_parts.append(f'<span class="tag red">误召回: {", ".join(c["found_unexpected"])}</span>')
                if c.get("missing_fragments"):
                    detail_parts.append(f'<span class="tag red">缺片段: {", ".join(c["missing_fragments"])}</span>')
                detail_parts.append(f'<span class="tag blue">P={c["precision"]:.0%} R={c["recall"]:.0%}</span>')
            elif bname == "judge_quality":
                gt_is = "badcase" if c["ground_truth_is_badcase"] else "normal"
                pred_is = "badcase" if c["predicted_is_badcase"] else "normal"
                n_runs = c.get("n_runs", 1)
                votes = c.get("badcase_votes", 0)
                detail_parts.append(f'<span class="tag blue">GT: {gt_is}/{c["ground_truth_type"]}</span>')
                detail_parts.append(f'<span class="tag {"green" if c["ground_truth_is_badcase"] == c["predicted_is_badcase"] else "red"}">'
                    f'Pred: {pred_is}/{c["predicted_type"]} (conf={c["predicted_confidence"]:.0%})</span>')
                if n_runs > 1:
                    detail_parts.append(f'<span class="tag purple">{n_runs}轮投票: {votes}/{n_runs} badcase</span>')
                # 维度评分条形图
                dims = c.get("dimensions", {})
                if dims:
                    detail_parts.append('<div style="margin:6px 0;font-size:11px">')
                    score_std = c.get("overall_score_std", 0)
                    std_str = f" ±{score_std:.1f}" if score_std > 0.5 else ""
                    detail_parts.append(f'<span style="color:var(--muted)">总分: {c.get("overall_score", "?")}{std_str}/35 </span>')
                    # show variance per dimension
                    dim_var = c.get("dimension_variance", {})
                    for name, dd in dims.items():
                        s = round(dd.get("score", 0))
                        bar = "▮" * s + "▯" * (5 - s)
                        color = "var(--green)" if s >= 4 else "var(--yellow)" if s >= 2 else "var(--red)"
                        cmt = _e(dd.get("comment", "")[:120])
                        var = dim_var.get(name, 0)
                        var_str = f" ±{var:.1f}" if var > 0.2 else ""
                        detail_parts.append(f'<div style="margin:2px 0"><span style="color:{color}">{bar}</span> <b>{name}</b> {dd.get("score", "?")}/5{var_str} <span style="color:var(--muted)">— {cmt}</span></div>')
                    detail_parts.append('</div>')

            parts.append(f"""<div class="fail-case">
    <div class="badge red"></div>
    <div class="body">
      <div class="name">{c["case_name"]} ({c.get("category", "?")})</div>
      <div class="detail">{' '.join(detail_parts)}</div>
    </div>
  </div>""")

        parts.append("</div>")

    return "\n".join(parts)


def _render_stability(cases: list) -> str:
    """渲染回复稳定性 benchmark HTML。"""
    if not cases:
        return '<p style="color:var(--muted);text-align:center;padding:40px">暂无数据，先跑: python src/tests/test_reply_stability_benchmark.py --run-api</p>'

    parts = ['<div class="failures"><h2>🤖 Bot 回复稳定性 — 每个 case × 3 次生成</h2>']
    for c in cases:
        score = c["avg_overall_score"]
        std = c["overall_score_std"]
        sim = c["cross_similarity"]
        stable = "🟢" if std < 3 else "🟡" if std < 6 else "🔴"
        parts.append(f"""<div class="fail-case">
    <div class="badge {"green" if std < 3 else "yellow" if std < 6 else "red"}"></div>
    <div class="body">
      <div class="name">{c["case_name"]} {stable} 总分 {score:.0f}±{std:.0f}/35 | 回复相似度 {sim:.0%}</div>""")

        if c.get("notes"):
            parts.append(f'<div style="font-size:11px;color:var(--yellow);margin:4px 0">📌 {_e(c["notes"][:200])}</div>')

        # 对话上下文
        ctx = c.get("context", [])
        if ctx:
            ctx_html = "<br>".join(_e(line[:200]) for line in ctx[-8:])
            parts.append(f'<pre style="font-size:11px;color:var(--muted);background:rgba(0,0,0,.2);padding:8px;border-radius:4px;margin:6px 0;max-height:150px;overflow-y:auto;white-space:pre-wrap;line-height:1.4">{ctx_html}</pre>')

        # 3次回复，每次带评分 + 工具调用 + 完整prompt
        reply_scores = c.get("reply_scores", [])
        if reply_scores:
            # 完整 prompt（可折叠）
            full_prompt = c.get("full_user_prompt", "")
            sys_prompt = c.get("full_system_prompt", "")
            if full_prompt or sys_prompt:
                pid = f"prompt-{c['case_name']}"
                parts.append(f'<details style="margin:6px 0;font-size:11px"><summary style="cursor:pointer;color:var(--blue)">📋 完整生产 Prompt（system + user）</summary>')
                if sys_prompt:
                    parts.append(f'<pre style="font-size:10px;color:var(--muted);background:rgba(0,0,0,.2);padding:8px;border-radius:4px;max-height:200px;overflow:auto;white-space:pre-wrap;margin:4px 0">[system]\n{_e(sys_prompt[:2000])}</pre>')
                if full_prompt:
                    parts.append(f'<pre style="font-size:10px;color:var(--muted);background:rgba(0,0,0,.2);padding:8px;border-radius:4px;max-height:200px;overflow:auto;white-space:pre-wrap;margin:4px 0">[user]\n{_e(full_prompt[:3000])}</pre>')
                parts.append('</details>')

            parts.append('<div style="font-size:12px;margin:6px 0"><b>3次生成 + 每次评分:</b></div>')
            for i, rs in enumerate(reply_scores):
                s = rs.get("overall_score", 0)
                bc = "⚠️badcase" if rs.get("is_badcase") else "✓"
                dim_summary = " | ".join(
                    f"{name}:{dd.get('score','?')}"
                    for name, dd in list(rs.get("dimensions", {}).items())[:4]
                )
                color = "var(--green)" if s >= 28 else "var(--yellow)" if s >= 21 else "var(--red)"
                parts.append(f'<div style="font-size:11px;margin:3px 0;padding:4px 8px;background:rgba(0,0,0,.12);border-radius:4px">'
                    f'<span style="color:{color};font-weight:600">[{i+1}] {s}/35 {bc}</span> '
                    f'<span style="color:var(--muted)">{dim_summary}</span><br>'
                    f'<span style="color:var(--text)">{_e(rs.get("reply", "")[:400])}</span>')
                # 工具调用日志
                tool_log = rs.get("tool_log", [])
                if tool_log:
                    parts.append('<div style="margin:4px 0 0 12px;font-size:10px">🛠️ 工具调用:')
                    for t in tool_log:
                        parts.append(f'<div style="margin:2px 0;padding:2px 6px;background:rgba(88,166,255,.08);border-radius:3px">'
                            f'<span style="color:var(--blue)">{t["name"]}({_e(t["args"][:100])})</span>'
                            f'<pre style="color:var(--muted);margin:2px 0 0 0;white-space:pre-wrap;max-height:80px;overflow:auto">{_e(t["result"][:300])}</pre></div>')
                    parts.append('</div>')
                parts.append('</div>')

        # 平均维度评分
        dims = c.get("avg_dimensions", {})
        if dims:
            parts.append('<div style="margin:8px 0;font-size:11px"><b>3次平均:</b>')
            for name, dd in dims.items():
                s = round(dd.get("score", 0))
                bar = "▮" * s + "▯" * (5 - s)
                color = "var(--green)" if s >= 4 else "var(--yellow)" if s >= 2 else "var(--red)"
                std_val = dd.get("std", 0)
                std_str = f" ±{std_val:.1f}" if std_val > 0.3 else ""
                parts.append(f'<div style="margin:2px 0"><span style="color:{color}">{bar}</span> <b>{name}</b> {dd.get("score", "?")}/5{std_str}</div>')
            parts.append('</div>')

        parts.append('</div></div>')

    parts.append('</div>')
    return "\n".join(parts)


def _e(s: str) -> str:
    """HTML 转义"""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="生成 Benchmark Dashboard")
    parser.add_argument("--open", action="store_true", help="生成后在浏览器中打开")
    args = parser.parse_args()

    print("🔍 采集分歧 case...")
    generate_dashboard()

    if args.open:
        import subprocess
        subprocess.run(["open", str(REPORT_PATH)])


if __name__ == "__main__":
    main()
