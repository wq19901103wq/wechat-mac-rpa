#!/usr/bin/env python3
"""Merge only the 120-second retries into the preserved 120-case HTML audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.evaluate_current_c0_historical_ab import _render


ROOT = Path(__file__).resolve().parent.parent
INITIAL_HTML = ROOT / "data/reports/private/current_c0_historical_ab_stage1_real_names_v1/index.initial_120.html"
MANIFEST = ROOT / "data/reports/private/adversarial_hidden_holdout_v1/manifest.json"
RETRY_SOURCES = (
    ROOT / "data/reports/private/current_c0_historical_ab_stage1_real_names_v1/results.partial_31.json",
    ROOT / "data/reports/private/current_c0_historical_ab_stage1_timeout_retry_early_v1/results.json",
    ROOT / "data/reports/private/current_c0_historical_ab_stage1_timeout_retry_v1/results.json",
)
RETRY_POSITIONS = (
    1, 3, 6, 8, 13, 15, 17, 19, 20, 21, 23, 24, 25, 32, 35, 36, 37, 39,
    40, 42, 43, 47, 49, 51, 55, 58, 62, 65, 66, 67, 69, 72, 74, 76, 77,
    78, 79, 81, 83, 85, 87, 91, 97, 112, 116, 120,
)
ORIGINAL_SUMMARY = """<div class="summary"><div><b>120</b><span>固定样本</span></div><div><b>40</b><span>明确应回复</span></div><div><b>80</b><span>可回可不回</span></div><div><b>46</b><span>生成失败</span></div><div><b>36</b><span>C0 沉默</span></div><div><b>38</b><span>C0 实际回复</span></div><div><b>7/28</b><span>非空回复 Judge</span></div></div>"""
FACT_CHECK_NOTICE = """<section class="fact-notice"><b>事实核查在本轮实际做了什么</b><p>当前生产配置关闭 Self-Refine，因此 Qwen 事实核查没有运行：调用 0、拦截 0、改写 0。本轮无法据此计算改进或误伤；Qwen 与 Codex 的取舍需要另做同题受控对比。</p></section>"""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_card(row: dict[str, Any], position: int) -> str:
    dummy_summary = {
        "total": 1,
        "participation": {"must_reply": 0, "optional_reply": 0},
        "outcomes": {"generation_failed": 0, "c0_silent": 0, "c0_replied": 0},
        "judge_correct": 0,
        "judge_scored": 0,
        "fact_check": {
            "attempted": 0,
            "issues_found": 0,
            "rewrite_applied": 0,
            "disabled": 0,
            "not_reached": 0,
            "errors": 0,
        },
        "coverage": {
            "group_chat": 0,
            "private_chat": 0,
            "complex_large_group": 0,
            "correct_common_prosperity_with_luxixi": 0,
            "youzi_group": 0,
        },
    }
    rendered = _render({"cases": [row], "summary": dummy_summary})
    match = re.search(r'(<article class="case".*?</article>)', rendered, flags=re.DOTALL)
    if not match:
        raise RuntimeError(f"could not render retry card at position {position}")
    return match.group(1).replace("#1 ·", f"#{position} ·", 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    initial = INITIAL_HTML.read_text(encoding="utf-8")
    if ORIGINAL_SUMMARY not in initial:
        raise RuntimeError("preserved HTML no longer has the verified original 120-case summary")
    cards = re.findall(r'(<article class="case".*?</article>)', initial, flags=re.DOTALL)
    if len(cards) != 120:
        raise RuntimeError(f"expected 120 preserved cards, found {len(cards)}")

    manifest_cases = _load_json(MANIFEST)["cases"]
    expected = {position: manifest_cases[position - 1]["internal_id"] for position in RETRY_POSITIONS}
    rows: dict[str, dict[str, Any]] = {}
    source_for_case: dict[str, str] = {}
    for source in RETRY_SOURCES:
        for row in _load_json(source).get("cases") or []:
            case_id = row["case_id"]
            if case_id not in expected.values():
                continue
            if float(row.get("generation", {}).get("timeout_budget_seconds") or 0) != 120.0:
                continue
            if case_id in rows:
                raise RuntimeError(f"duplicate 120-second retry: {case_id}")
            rows[case_id] = row
            source_for_case[case_id] = str(source.relative_to(ROOT))

    missing = {position: case_id for position, case_id in expected.items() if case_id not in rows}
    if missing:
        raise RuntimeError(f"missing retry rows: {missing}")
    if len(rows) != 46:
        raise RuntimeError(f"expected 46 unique retry rows, found {len(rows)}")

    for position, case_id in expected.items():
        if 'data-outcome="生成失败"' not in cards[position - 1]:
            raise RuntimeError(f"preserved card {position} was not an original generation failure")
        cards[position - 1] = _render_card(rows[case_id], position)

    retry_rows = list(rows.values())
    retry_scorable = sum(bool(row["validity"]["formally_scorable"]) for row in retry_rows)
    retry_judged = [row for row in retry_rows if row.get("judge")]
    retry_correct = sum(bool(row["judge"]["correct"]) for row in retry_judged)
    retry_failed = sum(bool(row["generation"]["failed"]) for row in retry_rows)
    retry_silent = sum(not row["generation"]["failed"] and not row["c0_reply"] for row in retry_rows)
    retry_replied = sum(not row["generation"]["failed"] and bool(row["c0_reply"]) for row in retry_rows)

    summary_html = (
        '<div class="summary"><div><b>120</b><span>固定样本</span></div>'
        '<div><b>40</b><span>明确应回复</span></div><div><b>80</b><span>可回可不回</span></div>'
        f'<div><b>{retry_failed}</b><span>生成失败</span></div>'
        f'<div><b>{36 + retry_silent}</b><span>C0 沉默</span></div>'
        f'<div><b>{38 + retry_replied}</b><span>C0 实际回复</span></div>'
        f'<div><b>{74 + retry_scorable}/120</b><span>可计分</span></div>'
        f'<div><b>{7 + retry_correct}/{28 + len(retry_judged)}</b><span>辅助 Judge</span></div></div>'
    )
    merged = initial.replace(ORIGINAL_SUMMARY, summary_html, 1)
    merged = merged.replace(
        '<div class="summary">',
        FACT_CHECK_NOTICE + '<div class="summary">',
        1,
    )
    merged = merged.replace(
        ".summary{display:grid;",
        ".fact-notice{background:#211d16;border:1px solid #d29922;border-radius:12px;padding:12px;margin:12px 0}.fact-notice p{margin:6px 0 0;color:var(--muted)}.summary{display:grid;",
        1,
    )
    merged = re.sub(
        r'<div id="deck">.*?</div></main><script>',
        '<div id="deck">' + "".join(cards) + '</div></main><script>',
        merged,
        count=1,
        flags=re.DOTALL,
    )
    merged = merged.replace(
        "历史完整聊天生成前对齐；生成后停止，不发送。页面显示历史导出的真实昵称；Judge 收到的是重新匿名后的成员名。",
        "历史完整聊天生成前对齐；生成后停止，不发送。46 条原超时题仅延长等待到 120 秒，其余题未重跑；页面显示历史导出的真实昵称，Judge 仅作辅助诊断。",
        1,
    )
    merged = merged.replace(
        '<nav class="controls" aria-label="题目导航"><button id="prev" type="button">上一题</button><span class="progress" id="progress">准备中</span><button id="next" type="button">下一题</button></nav>',
        '<nav class="controls" aria-label="题目导航"><select id="filter" aria-label="筛选题目"><option value="all">全部题目</option><option value="must_reply">明确应回复</option><option value="optional_reply">可回可不回</option><option value="生成失败">生成失败</option><option value="C0 沉默">C0 沉默</option><option value="C0 实际回复">C0 实际回复</option></select><button id="prev" type="button">上一题</button><span class="progress" id="progress">准备中</span><button id="next" type="button">下一题</button></nav>',
        1,
    )
    merged = merged.replace(
        ".controls button{appearance:none;border:1px solid var(--accent);background:var(--card);color:var(--text);border-radius:8px;padding:9px 13px;font-size:14px}",
        ".controls button,.controls select{appearance:none;border:1px solid var(--accent);background:var(--card);color:var(--text);border-radius:8px;padding:9px 13px;font-size:14px}",
        1,
    )
    merged = re.sub(
        r"<script>\(\(\)=>\{const cards=.*?</script>",
        """<script>(()=>{const cards=[...document.querySelectorAll('.case')];const progress=document.getElementById('progress');const prev=document.getElementById('prev');const next=document.getElementById('next');const filter=document.getElementById('filter');let visible=cards;let index=0;function applyFilter(){const value=filter.value;visible=cards.filter(card=>value==='all'||card.dataset.participation===value||card.dataset.outcome===value);cards.forEach(card=>card.classList.remove('active'));index=0;show(0)}function show(value){if(!visible.length){progress.textContent='0 / 0';prev.disabled=next.disabled=true;return}index=Math.max(0,Math.min(value,visible.length-1));cards.forEach(card=>card.classList.remove('active'));visible[index].classList.add('active');progress.textContent=`${index+1} / ${visible.length}`;prev.disabled=index===0;next.disabled=index===visible.length-1;window.scrollTo({top:document.querySelector('.controls').offsetTop-6,behavior:'smooth'})}prev.addEventListener('click',()=>show(index-1));next.addEventListener('click',()=>show(index+1));filter.addEventListener('change',applyFilter);applyFilter()})();</script>""",
        merged,
        count=1,
        flags=re.DOTALL,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(merged, encoding="utf-8")
    args.output.chmod(0o600)
    audit = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "preserved_initial_html": str(INITIAL_HTML.relative_to(ROOT)),
        "preserved_initial_sha256": _sha256(INITIAL_HTML),
        "retry_positions": list(RETRY_POSITIONS),
        "retry_count": len(retry_rows),
        "retry_sources": {case_id: source_for_case[case_id] for case_id in sorted(source_for_case)},
        "summary": {
            "total": 120,
            "formally_scorable": 74 + retry_scorable,
            "excluded": 46 - retry_scorable,
            "participation": {"must_reply": 40, "optional_reply": 80},
            "outcomes": {
                "generation_failed": retry_failed,
                "c0_silent": 36 + retry_silent,
                "c0_replied": 38 + retry_replied,
            },
            "judge_scored": 28 + len(retry_judged),
            "judge_correct": 7 + retry_correct,
            "coverage": {
                "group_chat": 80,
                "private_chat": 40,
                "complex_large_group": 50,
                "correct_common_prosperity_with_luxixi": 0,
                "youzi_group": 0,
            },
        },
        "interpretation": "Judge is auxiliary only and cannot alone declare C0 quality.",
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.audit_output.chmod(0o600)
    print(json.dumps(audit["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
