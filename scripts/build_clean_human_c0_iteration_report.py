#!/usr/bin/env python3
"""Build the fixed mobile report for clean-human C0 iterations."""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "data/reports/private/clean_human_c0_reserved_baseline_v1/results.json"
DEFAULT_CANDIDATE = ROOT / "data/reports/private/clean_human_c0_reserved_candidate_v2/results.json"
DEFAULT_OUTPUT = ROOT / "data/reports/private/clean_human_c0_iteration_v1/index.html"
DEFAULT_RT = ROOT / "data/reports/private/persona_few_shot_rt_v1/results.json"
DEFAULT_FIXED_TEST = ROOT / "data/reports/private/adversarial_stage2_holdout_v1/manifest.json"
DEFAULT_FIXED_RESULTS = ROOT / "data/reports/private/fixed_c0_holdout_v1/results.json"
DEFAULT_FORMAL_AB = ROOT / "data/reports/private/fixed_c0_formal_ab_v2/results.json"
DEFAULT_RETRIEVAL_BASELINE = ROOT / "data/reports/private/clean_human_c0_retrieval_dev_baseline_v1/results.json"
DEFAULT_RETRIEVAL_CANDIDATE = ROOT / "data/reports/private/clean_human_c0_retrieval_dev_candidate_v1/results.json"
DEFAULT_WORKFLOW = ROOT / "docs/03-guides/JUDGE_C0_CANONICAL_WORKFLOW.json"

VERIFIED_HARD_FAILURES: dict[str, str] = {}

FACT_RISKS = {
    "019a495ddbbeb640": "候选声称自己正在来、马上到；当前材料无法核实现场行程，但也没有与完整上下文或 Wiki 明确冲突，不能判硬伤。",
    "95d8e392e7e8dbf7": "候选提到掉发；这是 Judge 可能不知道的本人信息，当前没有与完整上下文或 Wiki 明确冲突，只能待核实。",
    "ea176c751118345e": "候选说‘到时候见’；语气比先前的‘看情况’更确定，但当前没有足以确认其与实际安排或 Wiki 冲突的证据，只能标风险。",
}

EVALUATION_LIMITATIONS = {
    "a87eb70675abbcad": "关键图片未识别，无法核实滤芯灯是否闪烁；该题不能用于判定事实硬伤。",
}

MEDIA_TAIL_MARKERS = ("<img", "cdnmidimgurl", "[图片]", "[表情]", "[视频]", "<type>3</type>")

CASE_NOTES = {
    "568120fe719d0297": "有效信号：候选接住了更早群聊里已经出现的 RSU 退税信息，简短且有依据。",
    "9b10a4df3fd01f41": "有效信号：候选直接承接搬家安排，没有把对话扩写成解释。",
    "d5b9e0f47400d6d7": "有效信号：候选用短促反向自嘲回应吹捧，更接近真人聊天动作。",
    "121128533897bde9": "仍然偏 AI：像加工出的完整金句，没有真人原回复那种关系特定的拆台。",
    "7a7b1da3a32d0657": "评测噪声：两版 C0 回复完全相同，但 Judge 给出了不同判断。",
    "257a36cf3ecdf3a0": "稳定性提醒：两版 C0 回复完全相同；Judge 结论相同，但理由措辞仍有变化。",
    "86bbe58461d39d75": "目标事实风险已收紧：从肯定看房规则，改为‘不确定能不能带，回头问下开发商’。现有材料未证明旧回复与 Wiki 明确冲突，因此不追溯标为硬伤。",
    "6b88200a2fff7472": "仍然偏 AI：两大段完整科普，解释腔明显。",
    "44e4d1eb14adf1b1": "仍然偏 AI：使用‘吸猫一时爽，一直吸一直爽’老梗，老梗不等于真人幽默。",
    "9015700657757824": "仍有无依据扩写：增加了上下文未明确的‘回来’设定。",
    "36825c72ce69ccc3": "没有跟随真人原回复‘不想当了’的真实话题动作，转成了泛化玩梗。",
}

ITERATION_LEDGER = (
    {
        "round": "旧链第 1 轮",
        "change": "同时加入三类守护规则：不重复解释显然信息、避免无依据现实行动或承诺、群聊默认短回复。",
        "source": "来自用户指出的显然信息复述、老梗复用和未知事实武断判断，再由 Judge 提名案例、人工查看完整上下文归类。",
        "result": "20 条已暴露训练案例：候选胜 8、基线胜 4、持平 8。当时记录了身份、机制等事实风险；未按当前‘上下文/Wiki 明确冲突’口径重新核实，不能统称硬伤。",
        "decision": "不采用；且后来确认这些上下文含 AI 延续内容，整轮只保留为历史探索。",
    },
    {
        "round": "旧链第 2 轮",
        "change": "细化为显然信息去重、可信私人事实、未知系统机制、现实承诺边界和群聊克制五类规则。",
        "source": "把第 1 轮胜负理由按旧梗、重复解释、事实冲突、话题锚定和语气展开聚类，只保留跨案例复现的问题。",
        "result": "20 条训练案例：候选胜 9、基线胜 3、持平 8；仍把已知属狗答错。",
        "decision": "曾作为旧链临时比较基线；因 AI 上下文污染，现已失去正式采用依据。",
    },
    {
        "round": "旧链第 3 轮",
        "change": "只增加双关语义识别规则。",
        "source": "人工重看“你属什么”案例后，发现上一轮把真人的“属什么 / 不属于你”双关误归为私人事实冲突。",
        "result": "候选胜 3、基线胜 7、持平 10；目标双关未修复，并新增机制幻觉和泛化解释。",
        "decision": "不采用；同时沉淀出‘抽象规则写进提示词不等于机制生效’。",
    },
    {
        "round": "旧链第 4 轮",
        "change": "只增加上下文证据门控，限制无依据机制、因果、动机和风险判断。",
        "source": "从 Judge 提名案例中人工聚类出机制幻觉、用途或动机猜测、无人追问仍主动解释三类重复错误。",
        "result": "候选胜 5、基线胜 5、持平 10；没有净收益，私人事实和平台行为断言仍会出错。",
        "decision": "不采用。",
    },
    {
        "round": "旧链第 5 轮",
        "change": "只增加发送前终稿核对：事实冲突、未知平台行为、旧话题回捞和解释腔。",
        "source": "人工复盘前几轮稳定失败，把明确错因聚为事实冲突、无依据判断、旧话题回捞、解释腔。",
        "result": "候选胜 4、基线胜 4、持平 12；新增漏身份纠正、异常语序和无依据亲历。",
        "decision": "不采用；证明通用末端检查不足以约束事实与身份。",
    },
    {
        "round": "旧链第 6 轮",
        "change": "只增加生成前的可信事实与身份锚定。",
        "source": "第 5 轮仍反复出现属相冲突、漏身份纠正和无依据具体断言，因此优先验证事实与身份绑定。",
        "result": "候选胜 2、基线胜 10、持平 8；当时标注的事实风险由 1 个增至 3 个，但这些旧标注尚未按当前硬伤口径复核。",
        "decision": "不采用；之后用户指出 AI 上下文循环继承 AI 味，旧链整体停止。",
    },
    {
        "round": "真人案例第 1 轮",
        "change": "唯一变量是把生成 few-shot 从 11 条幽默样本扩大到 973 条已核实真人样本。",
        "source": "只用真实聊天上下文：冻结 Judge 提名能识别出的 C0，人工对照同上下文真人原回复，再追踪 few-shot ID、工具调用和历史证据。",
        "result": "共同 27 题中 Judge 识别率从 25/27 降至 21/27，真人感有局部净收益；但新增‘预看房人数和许可规则’事实风险。",
        "decision": "不采用；定位出‘可迁移风格，不能把历史个例升级成当前规则’。现有材料未证明该回复与 Wiki 明确冲突，不再称为硬伤。",
    },
    {
        "round": "真人案例第 2 轮",
        "change": "唯一变量是加入证据强度门控：历史单次经历不能推出当前普遍许可、固定条件、流程或结果保证。",
        "source": "沿第 1 轮预看房事实风险追踪完整 trace，确认不是直接复制 few-shot，而是把另一群的一次经历升级成通用规则。",
        "result": "30 条中 27 条可评分；共同可比 24 条的 Judge 识别率从 22/24 降至 21/24。目标预看房风险已收紧；人工复核未确认任何与完整上下文或 Wiki 明确冲突的硬伤。",
        "decision": "暂不采用；目前只有 1/24 的微小改善，样本仅 30 条且群聊覆盖不足，证据不够。当前 C0 基线保持不变，但不再用错误的硬伤判断否决候选。",
    },
    {
        "round": "真人案例第 3 轮（最新）",
        "change": "唯一变量是提高真人案例召回相关性：仅同一聊天，bigram 重合至少 2、Dice 至少 0.06，最多 2 条并允许 0 条。",
        "source": "人工抽查默认召回发现固定塞入弱相关案例，例如‘试用期→打孔/小广告’、‘csig→小米交易’，因此验证宁缺毋滥是否比固定取满更好。",
        "result": "18 条可评分，Judge 识别率 15/18→15/18；17 条共同有 Judge 结果中改善 1、退步 2、不变 14，逐题净退步 1。已核实硬伤 0。候选 RT p50/p95 为 12.860/52.903 秒；few-shot 字符 p50/p95 为 0/539。",
        "decision": "不采用；提示词明显缩短且 RT 可接受，但没有真人感净收益，保留固定 240 题对应的现有 C0 基线。",
    },
)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def reply_text(value: object) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "")


def full_context(case: dict) -> str:
    blind = case.get("blind_case") or {}
    original = blind.get("context_original") or []
    if original:
        return "\n".join(str(item) for item in original)
    return reply_text(case.get("source_context"))


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(correct: int, total: int) -> str:
    return f"{correct / total:.1%}" if total else "—"


def fmt_ms(value: object) -> str:
    try:
        return f"{float(value):,.0f} ms"
    except (TypeError, ValueError):
        return "—"


def fixed_test_coverage(path: Path = DEFAULT_FIXED_TEST) -> dict:
    payload = load(path)
    rows = payload.get("cases") or []
    eligible = []
    excluded = []
    for row in rows:
        tail = "\n".join(str(value) for value in (row.get("context") or [])[-3:]).lower()
        (excluded if any(marker.lower() in tail for marker in MEDIA_TAIL_MARKERS) else eligible).append(row)
    groups = [row for row in eligible if row.get("chat_type") == "group"]
    return {
        "reserved_total": len(rows),
        "scoreable_total": len(eligible),
        "media_tail_excluded": len(excluded),
        "group_chat": len(groups),
        "private_chat": len(eligible) - len(groups),
        "complex_large_group": sum(bool(row.get("complex_large_group")) for row in groups),
        "distinct_chats": len({str(row.get("chat_id")) for row in eligible}),
        "correct_common_prosperity_with_luxixi": sum(
            row.get("specified_group") == "correct_common_prosperity_with_luxixi" for row in eligible
        ),
        "youzi_group": sum(row.get("specified_group") == "youzi_group_2" for row in eligible),
    }


def _rt_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(percentile * len(values)) - 1)]


def rt_summary(path: Path = DEFAULT_RT) -> dict:
    if not path.exists():
        return {"completed": 0, "expected": 38, "complete": False}
    payload = load(path)
    entries = payload.get("entries") or {}
    by_case: dict[str, dict[str, dict]] = {}
    for row in entries.values():
        if row.get("status") == "ok":
            by_case.setdefault(str(row.get("case_id")), {})[str(row.get("condition"))] = row
    pairs = [pair for pair in by_case.values() if {"small_11", "large_973"} <= pair.keys()]
    deltas = [float(pair["large_973"]["elapsed_ms"]) - float(pair["small_11"]["elapsed_ms"]) for pair in pairs]
    relationship_deltas = {}
    for relationship in ("group", "private"):
        values = [
            float(pair["large_973"]["elapsed_ms"]) - float(pair["small_11"]["elapsed_ms"])
            for pair in pairs
            if pair["large_973"].get("relationship") == relationship
        ]
        relationship_deltas[relationship] = {
            "n": len(values),
            "p50_ms": round(statistics.median(values), 1) if values else None,
            "p95_ms": round(_rt_percentile(values, 0.95), 1) if values else None,
        }
    local = payload.get("local_retrieval") or {}
    executable_cases = len({str(row.get("case_id")) for row in entries.values()})
    expected_calls = executable_cases * 2
    return {
        "completed": len(entries),
        "expected": expected_calls,
        "complete": bool(executable_cases) and len(entries) >= expected_calls,
        "target_cases": 20,
        "executable_cases": executable_cases,
        "skipped_cases": 20 - executable_cases,
        "failures": sum(row.get("status") != "ok" for row in entries.values()),
        "paired_ok": len(pairs),
        "paired_delta_p50_ms": round(statistics.median(deltas), 1) if deltas else None,
        "paired_delta_p95_ms": round(_rt_percentile(deltas, 0.95), 1) if deltas else None,
        "paired_delta_by_relationship": relationship_deltas,
        "conditions": payload.get("summary") or {},
        "local_retrieval": local,
    }


def render_iteration_ledger() -> str:
    rows = []
    for item in ITERATION_LEDGER:
        rows.append(
            "<article class=\"ledger-row\">"
            f"<h3>{esc(item['round'])}</h3>"
            f"<p><b>改了什么：</b>{esc(item['change'])}</p>"
            f"<p><b>思路从哪来：</b>{esc(item['source'])}</p>"
            f"<p><b>结果：</b>{esc(item['result'])}</p>"
            f"<p><b>是否采用：</b>{esc(item['decision'])}</p>"
            "</article>"
        )
    return "".join(rows)


def fixed_context(case: dict) -> str:
    rows = []
    for item in case.get("context") or []:
        if isinstance(item, dict):
            speaker = item.get("speaker") or item.get("role") or "unknown"
            when = item.get("time") or ""
            content = item.get("content") or ""
            rows.append(f"{speaker}（{when}）：{content}" if when else f"{speaker}：{content}")
        else:
            rows.append(str(item))
    return "\n".join(rows)


def render_fixed_case(case: dict) -> str:
    status = str(case.get("status") or "not_run")
    judge = case.get("judge") or {}
    chat_type = "group" if case.get("chat_type") == "group" else "private"
    tags = ["fixed", status, chat_type]
    if case.get("complex_large_group"):
        tags.append("complex")
    specified = str(case.get("specified_group") or "")
    if specified == "correct_common_prosperity_with_luxixi":
        tags.append("prosperity")
    if specified.startswith("youzi_group"):
        tags.append("youzi")
    if judge:
        tags.append("identified" if judge.get("correct") else "fooled")
    status_label = {
        "scorable": "可成对评测",
        "excluded_unscorable": "静默 / 无法成对",
        "excluded_media_dependency": "媒体依赖",
        "generation_error": "生成失败",
        "not_run": "待处理",
    }.get(status, status)
    if judge:
        judge_html = (
            f'<p><b>冻结 Judge：</b>{"判对（识别出 C0）" if judge.get("correct") else "判错（把 C0 当真人）"}'
            f' · 置信度 {esc(judge.get("confidence", "—"))}</p>'
            f'<p class="reason">{esc(judge.get("reason", ""))}</p>'
        )
    elif status == "scorable":
        judge_html = '<p class="muted">冻结 Judge：待评测</p>'
    else:
        judge_html = f'<p class="muted">状态：{esc(status_label)}</p>'
    error = case.get("generation_error") or case.get("judge_error") or ""
    error_html = f'<div class="hard-note"><b>执行错误：</b>{esc(error)}</div>' if error else ""
    return f"""
    <article class="case {' '.join(tags)}" data-tags="{' '.join(tags)}">
      <div class="case-head"><b>{esc(case.get('case_id', ''))}</b><span>{esc(chat_type)} · {esc(status_label)}</span></div>
      {error_html}
      <details><summary>完整真人上下文（展开）</summary><pre>{esc(fixed_context(case))}</pre></details>
      <section class="human"><h3>真人原回复</h3><pre>{esc(reply_text(case.get('human_reply')))}</pre></section>
      <section><h3>当前采用的 C0</h3><pre>{esc(reply_text(case.get('c0_reply')) or f'[{status_label}]')}</pre>{judge_html}</section>
    </article>"""


def render_case(base: dict | None, cand: dict | None) -> str:
    case = cand or base or {}
    case_id = str(case.get("case_id", ""))
    relationship = str(case.get("relationship", "unknown"))
    base_status = str((base or {}).get("status", "missing"))
    cand_status = str((cand or {}).get("status", "missing"))
    base_reply = reply_text((base or {}).get("c0_reply"))
    cand_reply = reply_text((cand or {}).get("c0_reply"))
    changed = base_reply != cand_reply
    base_judge = (base or {}).get("judge") or {}
    cand_judge = (cand or {}).get("judge") or {}
    base_correct = base_judge.get("correct")
    cand_correct = cand_judge.get("correct")
    noise = (
        base_status == "scorable"
        and cand_status == "scorable"
        and not changed
    )
    hard = case_id in VERIFIED_HARD_FAILURES
    risk = case_id in FACT_RISKS
    limitation = case_id in EVALUATION_LIMITATIONS
    fooled = cand_status == "scorable" and cand_correct is False
    tags = [relationship, "changed" if changed else "same"]
    if fooled:
        tags.append("fooled")
    if hard:
        tags.append("hard")
    if risk:
        tags.append("risk")
    if limitation:
        tags.append("limitation")
    if noise:
        tags.append("noise")
    if base_status != "scorable" or cand_status != "scorable":
        tags.append("unpaired")

    note_parts = []
    if case_id in CASE_NOTES:
        note_parts.append(f'<div class="note">{esc(CASE_NOTES[case_id])}</div>')
    if hard:
        note_parts.append(f'<div class="hard-note"><b>已核实硬伤：</b>{esc(VERIFIED_HARD_FAILURES[case_id])}</div>')
    if risk:
        note_parts.append(f'<div class="risk-note"><b>待核实风险：</b>{esc(FACT_RISKS[case_id])}</div>')
    if limitation:
        note_parts.append(f'<div class="limit-note"><b>评测限制：</b>{esc(EVALUATION_LIMITATIONS[case_id])}</div>')

    def judge_line(label: str, status: str, judge: dict) -> str:
        if status != "scorable" or not judge:
            return f'<p class="muted">{esc(label)}：{esc(status)}</p>'
        correct = judge.get("correct")
        outcome = "识别出 C0" if correct else "未分清"
        return (
            f'<p><b>{esc(label)}：</b>{outcome} · 置信度 {esc(judge.get("confidence", "—"))}</p>'
            f'<p class="reason">{esc(judge.get("reason", ""))}</p>'
        )

    return f"""
    <article class="case {' '.join(tags)}" data-tags="{' '.join(tags)}">
      <div class="case-head"><b>{esc(case_id)}</b><span>{esc(relationship)}</span></div>
      {''.join(note_parts)}
      <details><summary>完整真人上下文（展开）</summary><pre>{esc(full_context(case))}</pre></details>
      <section class="human"><h3>真人原回复</h3><pre>{esc(reply_text(case.get('human_reply')))}</pre></section>
      <div class="grid">
        <section><h3>当前基线 C0</h3><pre>{esc(base_reply or f'[{base_status}]')}</pre>{judge_line('冻结 Judge', base_status, base_judge)}</section>
        <section><h3>候选 C0</h3><pre>{esc(cand_reply or f'[{cand_status}]')}</pre>{judge_line('冻结 Judge', cand_status, cand_judge)}</section>
      </div>
      <details><summary>本题 few-shot 来源</summary>
        <p><b>基线：</b>{esc(', '.join((base or {}).get('few_shot_ids') or []))}</p>
        <p><b>候选：</b>{esc(', '.join((cand or {}).get('few_shot_ids') or []))}</p>
      </details>
    </article>"""


def render_retrieval_case(base: dict, cand: dict) -> str:
    base_judge = base.get("judge") or {}
    cand_judge = cand.get("judge") or {}
    return f"""
    <article class="case changed">
      <div class="case-head"><b>{esc(cand.get('case_id', ''))}</b><span>{esc(cand.get('relationship', ''))}</span></div>
      <details open><summary>完整真人上下文</summary><pre>{esc(full_context(cand))}</pre></details>
      <section class="human"><h3>真人原回复</h3><pre>{esc(reply_text(cand.get('human_reply')))}</pre></section>
      <div class="grid">
       <section><h3>基线 C0</h3><pre>{esc(reply_text(base.get('c0_reply')))}</pre><p><b>Judge：</b>{'识别出 C0' if base_judge.get('correct') else '未分清'}</p><p class="reason">{esc(base_judge.get('reason', ''))}</p></section>
       <section><h3>候选 C0</h3><pre>{esc(reply_text(cand.get('c0_reply')))}</pre><p><b>Judge：</b>{'识别出 C0' if cand_judge.get('correct') else '未分清'}</p><p class="reason">{esc(cand_judge.get('reason', ''))}</p></section>
      </div>
      <p><b>候选 few-shot：</b>{esc(', '.join(cand.get('few_shot_ids') or []) or '0 条')}</p>
    </article>"""


def build(baseline: dict, candidate: dict) -> str:
    workflow = load(DEFAULT_WORKFLOW)
    fixed_target = workflow["fixed_test"]["target"]
    page_url = workflow["reporting"]["url"]
    holdout = fixed_test_coverage()
    fixed_result = load(DEFAULT_FIXED_RESULTS) if DEFAULT_FIXED_RESULTS.exists() else {"summary": {}, "cases": []}
    fixed_summary = fixed_result.get("summary") or {}
    fixed_cases = fixed_result.get("cases") or []
    fixed_counts = fixed_summary.get("status_counts") or {}
    fixed_segments = fixed_summary.get("segments") or {}
    fixed_cards = "".join(render_fixed_case(case) for case in fixed_cases)
    fixed_judged = int(fixed_summary.get("judged") or 0)
    fixed_correct = int(fixed_summary.get("judge_correct") or 0)
    fixed_scorable = int(fixed_counts.get("scorable") or 0)
    fixed_unscorable = int(fixed_counts.get("excluded_unscorable") or 0)
    fixed_media = int(fixed_counts.get("excluded_media_dependency") or 0)
    fixed_errors = int(fixed_counts.get("generation_error") or 0)
    fixed_pending = int(fixed_counts.get("not_run") or 0)
    fixed_done = len(fixed_cases) - fixed_pending
    fixed_complete = fixed_summary.get("status") == "complete"
    retrieval_base = load(DEFAULT_RETRIEVAL_BASELINE)
    retrieval_cand = load(DEFAULT_RETRIEVAL_CANDIDATE)
    retrieval_base_by_id = {case["case_id"]: case for case in retrieval_base["cases"]}
    retrieval_cand_by_id = {case["case_id"]: case for case in retrieval_cand["cases"]}
    retrieval_paired = [
        case_id for case_id in retrieval_base_by_id.keys() & retrieval_cand_by_id.keys()
        if retrieval_base_by_id[case_id].get("status") == "scorable"
        and retrieval_cand_by_id[case_id].get("status") == "scorable"
    ]
    retrieval_common_judged = [
        case_id for case_id in retrieval_paired
        if retrieval_base_by_id[case_id].get("judge") and retrieval_cand_by_id[case_id].get("judge")
    ]
    retrieval_improved = [case_id for case_id in retrieval_common_judged if retrieval_base_by_id[case_id]["judge"]["correct"] and not retrieval_cand_by_id[case_id]["judge"]["correct"]]
    retrieval_regressed = [case_id for case_id in retrieval_common_judged if not retrieval_base_by_id[case_id]["judge"]["correct"] and retrieval_cand_by_id[case_id]["judge"]["correct"]]
    retrieval_flips = retrieval_improved + retrieval_regressed
    retrieval_cards = "".join(render_retrieval_case(retrieval_base_by_id[i], retrieval_cand_by_id[i]) for i in retrieval_flips)
    retrieval_base_summary = retrieval_base.get("summary") or {}
    retrieval_cand_summary = retrieval_cand.get("summary") or {}
    segment_rows = []
    for key in ("group", "private", "complex_large_group", "common_prosperity", "youzi"):
        row = fixed_segments.get(key) or {}
        segment_rows.append(
            "<tr>"
            f"<td>{esc(row.get('name', key))}</td>"
            f"<td>{esc(row.get('reserved', 0))}</td>"
            f"<td>{esc(row.get('scorable', 0))}</td>"
            f"<td>{esc(row.get('judged', 0))}</td>"
            f"<td>{esc(row.get('judge_correct', 0))}/{esc(row.get('judged', 0))} "
            f"({pct(int(row.get('judge_correct') or 0), int(row.get('judged') or 0))})</td>"
            "</tr>"
        )
    fixed_segment_table = "".join(segment_rows)
    rt = rt_summary()
    small_rt = (rt.get("conditions") or {}).get("small_11") or {}
    large_rt = (rt.get("conditions") or {}).get("large_973") or {}
    small_local = (rt.get("local_retrieval") or {}).get("small_11") or {}
    large_local = (rt.get("local_retrieval") or {}).get("large_973") or {}
    rt_detail = ""
    if rt["complete"]:
        split = rt.get("paired_delta_by_relationship") or {}
        rt_detail = (
            f"<p><b>结果：</b>旧 11 条库端到端 p50/p95 为 "
            f"{fmt_ms((small_rt.get('end_to_end') or {}).get('p50_ms'))}/"
            f"{fmt_ms((small_rt.get('end_to_end') or {}).get('p95_ms'))}；新 973 条库为 "
            f"{fmt_ms((large_rt.get('end_to_end') or {}).get('p50_ms'))}/"
            f"{fmt_ms((large_rt.get('end_to_end') or {}).get('p95_ms'))}。"
            f"成功配对的逐题差值 p50/p95 为 {fmt_ms(rt.get('paired_delta_p50_ms'))}/"
            f"{fmt_ms(rt.get('paired_delta_p95_ms'))}。</p>"
            f"<p><b>分场景配对差：</b>群聊 n={esc((split.get('group') or {}).get('n','—'))}，"
            f"p50 {fmt_ms((split.get('group') or {}).get('p50_ms'))}；私聊 n={esc((split.get('private') or {}).get('n','—'))}，"
            f"p50 {fmt_ms((split.get('private') or {}).get('p50_ms'))}。</p>"
            f"<p><b>开销拆分：</b>本地检索 p50 从 "
            f"{fmt_ms((small_local.get('timing') or {}).get('p50_ms'))} 到 "
            f"{fmt_ms((large_local.get('timing') or {}).get('p50_ms'))}；实际选入中位数从 "
            f"{esc(small_rt.get('selected_count_p50','—'))} 条 / {esc(small_rt.get('rendered_chars_p50','—'))} 字符，"
            f"变为 {esc(large_rt.get('selected_count_p50','—'))} 条 / {esc(large_rt.get('rendered_chars_p50','—'))} 字符。</p>"
        )
    base_by_id = {case["case_id"]: case for case in baseline["cases"]}
    cand_by_id = {case["case_id"]: case for case in candidate["cases"]}
    ids = list(dict.fromkeys([*base_by_id, *cand_by_id]))
    paired = [
        case_id for case_id in ids
        if base_by_id.get(case_id, {}).get("status") == "scorable"
        and cand_by_id.get(case_id, {}).get("status") == "scorable"
    ]
    base_correct = sum(bool(base_by_id[i]["judge"]["correct"]) for i in paired)
    cand_correct = sum(bool(cand_by_id[i]["judge"]["correct"]) for i in paired)
    changed = [i for i in paired if reply_text(base_by_id[i].get("c0_reply")) != reply_text(cand_by_id[i].get("c0_reply"))]
    base_changed_correct = sum(bool(base_by_id[i]["judge"]["correct"]) for i in changed)
    cand_changed_correct = sum(bool(cand_by_id[i]["judge"]["correct"]) for i in changed)
    identical_reruns = [
        i for i in paired
        if reply_text(base_by_id[i].get("c0_reply")) == reply_text(cand_by_id[i].get("c0_reply"))
    ]
    outcome_flips = [
        i for i in identical_reruns
        if base_by_id[i]["judge"]["correct"] != cand_by_id[i]["judge"]["correct"]
    ]
    total_cases = len(ids)
    group_cases = sum((cand_by_id.get(i) or base_by_id.get(i) or {}).get("relationship") == "group" for i in ids)
    paired_group = [i for i in paired if (cand_by_id.get(i) or base_by_id.get(i) or {}).get("relationship") == "group"]
    base_group_correct = sum(bool(base_by_id[i]["judge"]["correct"]) for i in paired_group)
    cand_group_correct = sum(bool(cand_by_id[i]["judge"]["correct"]) for i in paired_group)
    cards = "".join(render_case(base_by_id.get(i), cand_by_id.get(i)) for i in ids)
    ledger = render_iteration_ledger()
    summary = {
        "fixed_240_evaluation": fixed_summary,
        "retrieval_relevance_experiment": {
            "development_cases": 20,
            "scorable": 18,
            "baseline_judge_identified": 15,
            "candidate_judge_identified": 15,
            "common_judged": len(retrieval_common_judged),
            "improved": len(retrieval_improved),
            "regressed": len(retrieval_regressed),
            "unchanged": len(retrieval_common_judged) - len(retrieval_flips),
            "verified_hard_failures": 0,
            "candidate_generation_rt_seconds": retrieval_cand_summary.get("generation_rt_seconds"),
            "candidate_few_shot_rendered_chars": retrieval_cand_summary.get("few_shot_rendered_chars"),
            "adopted": False,
        },
        "evaluation_status": {
            "evaluated_comparable": len(paired),
            "fixed_test_pending": fixed_pending,
            "media_dependent_unscored": holdout["media_tail_excluded"],
            "fixed_test_reserved_total": holdout["reserved_total"],
        },
        "paired_scorable": len(paired),
        "baseline_correct": base_correct,
        "candidate_correct": cand_correct,
        "changed_reply_paired": len(changed),
        "baseline_changed_correct": base_changed_correct,
        "candidate_changed_correct": cand_changed_correct,
        "identical_output_judge_reruns": identical_reruns,
        "identical_output_outcome_flips": outcome_flips,
        "verified_hard_failures": list(VERIFIED_HARD_FAILURES),
        "fact_risks": FACT_RISKS,
        "evaluation_limitations": EVALUATION_LIMITATIONS,
        "group_cases": group_cases,
        "total_cases": total_cases,
        "paired_group_cases": len(paired_group),
        "fixed_test": holdout,
        "rt_benchmark": rt,
        "adopted": False,
    }
    out_dir = DEFAULT_OUTPUT.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>C0 统一流程与历史过渡基线</title>
<style>
:root{{color-scheme:dark;--bg:#0b1020;--card:#151c31;--line:#2a3554;--text:#e7ecf5;--muted:#9fb0d0;--good:#4cc38a;--bad:#ff7b72;--warn:#e3b341}}
*{{box-sizing:border-box}}body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--text);margin:0;padding:12px}}main{{max-width:1040px;margin:auto}}
h1{{font-size:24px;margin:8px 0}}h2{{font-size:18px}}h3{{font-size:14px;margin:0 0 8px}}p{{line-height:1.55}}.muted,.reason{{color:var(--muted)}}
.summary,.case,.decision,.ledger{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin:12px 0}}.decision{{border-color:var(--warn)}}
.ledger-row{{background:#0d1427;border-radius:10px;padding:11px;margin:10px 0}}.ledger-row p{{margin:6px 0}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}.metric{{background:#0d1427;border-radius:10px;padding:10px}}.metric b{{display:block;font-size:21px}}
.toolbar{{position:sticky;top:0;z-index:2;background:rgba(11,16,32,.95);padding:9px 0;display:flex;gap:7px;overflow:auto}}button{{border:1px solid var(--line);background:#18223b;color:var(--text);border-radius:999px;padding:7px 11px;white-space:nowrap}}button.active{{border-color:#68a7ff;background:#214277}}
.case-head{{display:flex;justify-content:space-between;color:var(--muted);margin-bottom:9px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}section{{background:#0d1427;border-radius:10px;padding:11px;margin:10px 0}}section.human{{border-left:3px solid var(--good)}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}}details{{margin:10px 0}}summary{{cursor:pointer;color:#b8c8e8}}.note,.hard-note,.risk-note,.limit-note{{padding:9px;border-radius:9px;margin:8px 0;background:#172d28}}.hard-note{{background:#3a2025;color:#ffd4d1}}.risk-note{{background:#3b321c;color:#ffe5a0}}.limit-note{{background:#202d46;color:#c9dcff}}.case.hard{{border-color:var(--bad)}}.case.risk{{border-color:var(--warn)}}.case.noise{{box-shadow:inset 3px 0 var(--warn)}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:8px 5px;border-bottom:1px solid var(--line);text-align:left}}.development-only,.history{{display:none}}
@media(max-width:720px){{.metrics{{grid-template-columns:1fr 1fr}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>C0 正式固定测试结果</h1>
<p class="muted">当前只迭代 C0；Codex Judge v5 冻结；Bot 关闭。每轮只更新本页：{esc(page_url)}</p>
<div class="summary"><h2>先看懂这页在做什么</h2>
 <p><b>开发材料：</b>可以看真人答案，用来找“不像真人”的原因、选择真人示例和修改 C0；不能用这里的分数证明正式提升。</p>
 <p><b>固定测试集：</b>相当于最后考试，答案不能用于调优。最终目标是 {fixed_target['scorable_total']} 条全部有效可比较，其中群聊 {fixed_target['group']}、私聊 {fixed_target['private']}；复杂大群至少 {fixed_target['complex_large_group_minimum']}，共同富裕群 {fixed_target['common_prosperity']}，柚子群 {fixed_target['youzi']}。</p>
 <p><b>强制不静默：</b>开发复验和正式测试时每题都要求 C0 回复，避免靠“不回复”绕开比较；这不改变未来生产环境是否应该参与对话的判断。</p>
 <p><b>怎样采用新版：</b>在相同题目、相同冻结 Judge 和相同条件下，Judge 识别出 C0 的数量只要严格减少，哪怕少 1 条也算改善；同时新增与完整上下文或适用 Wiki 明确矛盾的回复必须为 0。人工 A/B 和响应时间继续展示，但不是硬门槛。</p>
 <p><b>当前唯一改进方向：</b>最多找 3 条真人示例，按人物关系、聊天处境和需要怎样接话来匹配，不按几个相同词来匹配，并排除当前案例自身。</p>
</div>
<div class="summary"><h2>正式测试未完成：当前 91 / 240 条有效</h2>
 <p class="risk-note"><b>这不是最终 240 条有效测试。</b>它只是旧清单的处理结果：240 个场景都处理过，但只有 {fixed_judged} 个同时具备真人回复、C0 回复和 Judge 判断，可以计算识别率。最终固定集仍须最后一次补齐到 240 条有效案例，之后永久封存。</p>
 <div class="metrics">
  <div class="metric"><span>有效进度</span><b>{fixed_judged}/240</b><small>必须补满 240 才是正式完整结果</small></div>
  <div class="metric"><span>尚缺有效案例</span><b>{max(0, 240-fixed_judged)}</b></div>
  <div class="metric"><span>旧预留场景已处理</span><b>{fixed_done}/240</b><small>跑过不等于有效</small></div>
  <div class="metric"><span>无效或执行失败</span><b>{fixed_unscorable+fixed_media+fixed_errors}</b></div>
 </div>
 <p><b>其他状态：</b>待处理 {fixed_pending}；生成失败 {fixed_errors}。识别率只按已经同时具备真人回复与 C0 回复、且完成冻结 Judge 判断的案例计算，静默不自动算错。</p>
 <p><b>历史基线识别率：</b>{fixed_correct}/{fixed_judged}（{pct(fixed_correct,fixed_judged)}）。{'旧清单处理已结束，但最终固定测试集尚未补齐。' if fixed_complete else '旧清单仍有未完成执行；不能写成正式结论。'}</p>
</div>
<div class="summary"><h2>场景分布与结果</h2>
 <table><thead><tr><th>场景</th><th>预留</th><th>可成对</th><th>已判</th><th>Judge 识别</th></tr></thead><tbody>{fixed_segment_table}</tbody></table>
</div>
<div class="toolbar" id="fixed-toolbar">
 <button class="active" data-scope="fixed" data-filter="all">全部 240 个预留场景</button><button data-scope="fixed" data-filter="identified">Judge 判对（识别出 C0）</button><button data-scope="fixed" data-filter="fooled">Judge 判错（把 C0 当真人）</button><button data-scope="fixed" data-filter="group">群聊</button><button data-scope="fixed" data-filter="private">私聊</button><button data-scope="fixed" data-filter="complex">复杂大群</button><button data-scope="fixed" data-filter="prosperity">共同富裕</button><button data-scope="fixed" data-filter="youzi">柚子群</button><button data-scope="fixed" data-filter="excluded_unscorable">静默/不可成对</button><button data-scope="fixed" data-filter="excluded_media_dependency">媒体依赖</button><button data-scope="fixed" data-filter="generation_error">生成失败</button>
</div>
<p id="fixed-count" class="muted"></p><div id="fixed-cases">{fixed_cards}</div>
<div class="development-only"><div class="decision"><h2>真人案例第 3 轮：召回更准，但不采用</h2>
 <p><b>唯一改动：</b>同一 973 条真人库，只召回相同聊天中 bigram 重合至少 2、Dice 至少 0.06 的案例；允许 0 条，最多 2 条。固定 240 题完全封存，未参与规则归纳、阈值选择或本轮调优。</p>
 <div class="metrics">
  <div class="metric"><span>Judge 识别率</span><b>15/18 → 15/18</b></div>
  <div class="metric"><span>共同 Judge 结果</span><b>{len(retrieval_common_judged)} 条</b><small>改善 {len(retrieval_improved)} / 退步 {len(retrieval_regressed)}</small></div>
  <div class="metric"><span>候选生成 RT</span><b>12.860 秒</b><small>p95 52.903 秒</small></div>
  <div class="metric"><span>已核实硬伤</span><b>0 条</b><small>仅按上下文/Wiki 矛盾</small></div>
 </div>
 <p><b>结论：</b>候选提示词显著缩短，12 条不加载 few-shot、6 条只加载 1 条；few-shot 字符 p50/p95 为 0/539。RT 相比旧同池参考 p50 12.404 秒、p95 60.334 秒没有明显变慢，但真人感总分持平，逐题为改善 1、退步 2、不变 {len(retrieval_common_judged)-len(retrieval_flips)}，净退步 1，因此不采用。</p>
</div>
<div class="summary"><h2>本轮发生判断翻转的 3 条案例</h2><p class="muted">下面展示完整上下文、真人回复、两版 C0 和冻结 Judge 结果；三条均未发现与完整上下文或 Wiki 的可靠矛盾，不标硬伤。</p></div>
{retrieval_cards}
</div>
<div class="history">
<h1>历史 24 条实验记录</h1>
<p class="muted">以下小样本仅保留迭代账单与案例追溯，不再作为当前 C0 效果结论。</p>
<div class="summary"><h2>当前真实进度</h2>
 <div class="metrics">
  <div class="metric"><span>已评测 / 共同可比</span><b>{len(paired)} 条</b></div>
  <div class="metric"><span>固定测试集已完成</span><b>{fixed_done}/240 条</b></div>
  <div class="metric"><span>固定集有效对照</span><b>{fixed_judged} 条</b></div>
  <div class="metric"><span>固定测试集总预留</span><b>{holdout['reserved_total']} 条</b></div>
 </div>
 <p><b>口径：</b>页面下方 {len(paired)} 条是旧的小样本基线/候选实验，只用于追溯。固定 240 条已经全部处理，其中 {fixed_judged} 条形成有效真人/C0 对照；结论只能按这 {fixed_judged} 条讲，不能称为 240 条都完成了真人辨别。</p>
</div>
<div class="metrics">
 <div class="metric"><span>共同可评分</span><b>{len(paired)} 题</b></div>
 <div class="metric"><span>基线 Judge 识别率</span><b>{base_correct}/{len(paired)}</b><small>{pct(base_correct,len(paired))}</small></div>
 <div class="metric"><span>候选 Judge 识别率</span><b>{cand_correct}/{len(paired)}</b><small>{pct(cand_correct,len(paired))}</small></div>
 <div class="metric"><span>候选净变化</span><b>{cand_correct-base_correct:+d} 题</b><small>越接近 50% 越难区分</small></div>
</div>
<div class="decision"><h2>结论：有局部收益，但不采用候选版</h2>
 <p>候选把共同样本上的 Judge 识别率从 {base_correct}/{len(paired)}（{pct(base_correct,len(paired))}）降到 {cand_correct}/{len(paired)}（{pct(cand_correct,len(paired))}），只改善 1 题。只看实际发生回复变化的 {len(changed)} 题，则从 {base_changed_correct}/{len(changed)} 降到 {cand_changed_correct}/{len(changed)}。</p>
 <p>目标案例已从肯定看房规则改成“不能确定，向开发商确认”，说明门控局部生效。按纠正后的口径，只有与完整上下文或 Wiki 明确矛盾才算硬伤；本轮已核实硬伤为 {len(VERIFIED_HARD_FAILURES)}。另有 {len(FACT_RISKS)} 条只能标记为待核实风险、{len(EVALUATION_LIMITATIONS)} 条受缺失图片限制，不能因 Judge 不知道私人事实就判错。</p>
 <p>候选仍暂不采用：改善只有 1/{len(paired)}，且 {len(identical_reruns)} 题两版回复完全相同、其中 {len(outcome_flips)} 题 Judge 结论翻转；再加上样本和群聊覆盖不足，目前不能确认收益可迁移。</p>
</div>
<div class="summary"><h2>这一方向怎么发现、下一轮怎么用</h2>
 <p>第 1 轮从 973 条真人 few-shot 中发现：风格迁移有收益，但一次历史经历会被错误升级成当前通用规则。本轮因此只增加证据强度门控，禁止用历史个例推出当前许可、固定条件、流程或结果保证。</p>
 <p>本轮没有确认新的事实硬伤。候选不采用的原因是独立证据太弱，而不是把未知私人事实误判为硬伤；后续方向仍须回到训练集里的已核实真人案例发现。</p>
</div>
<div class="summary"><h2>群聊覆盖与缺口</h2>
 <p>本轮共 {total_cases} 条，其中群聊 {group_cases} 条（{pct(group_cases,total_cases)}）、私聊 {total_cases-group_cases} 条（{pct(total_cases-group_cases,total_cases)}）。共同可比样本中群聊只有 {len(paired_group)} 条，基线与候选均被 Judge 识别 {base_group_correct}/{len(paired_group)} 和 {cand_group_correct}/{len(paired_group)}。</p>
 <p><b>结论：</b>群聊占比确实不足，尤其共同可比群聊只有 {len(paired_group)} 条，不能证明复杂群聊有任何提升。扩大固定测试集仍并行进行，但不阻塞 C0 训练集迭代。</p>
 <p>唯一固定测试集 {holdout['reserved_total']} 条已全部处理；其中 {fixed_judged} 条形成有效真人/C0 对照，媒体依赖 {fixed_media} 条、静默或不可成对 {fixed_unscorable} 条。有效对照的实际群聊、私聊和复杂大群成绩见页面顶部固定集表格。</p>
 <p>重点群可计分覆盖：共同富裕群（含鲁西西）{holdout['correct_common_prosperity_with_luxixi']} 条、柚子群 {holdout['youzi_group']} 条。固定测试集答案不参与 C0 规则归纳或 few-shot 选择；当前候选尚未因扩集而自动采用。</p>
</div>
<div class="summary"><h2>973 条 few-shot 与响应时间</h2>
 <p>973 条是真人检索库，不会整批塞进一次提示词。受控对照固定最多选 8 条、最多 2500 字符；旧 11 条库和新 973 条库使用同一批 20 个真人场景（群聊 10、私聊 10），逐题 AB/BA 交替。</p>
 <p><b>当前进度：</b>{rt['completed']}/{rt['expected']} 次可执行生成完成；原计划 20 个场景中 {rt.get('executable_cases',0)} 个具备完整生成输入，{rt.get('skipped_cases',0)} 个未进入调用。失败 {rt.get('failures',0)} 次，完整成功配对 {rt.get('paired_ok',0)} 题。{'对照尚未跑满，页面不提前把速度差写成结论。' if not rt['complete'] else '可执行对照已完成；以下数字按成功调用统计。'}</p>
 {rt_detail}
 <p><b>召回质量抽查：</b>当前 973 库没有语义向量索引，实际主要按群聊/私聊硬过滤后，以字符与双字词重合、长度接近和同群加分排序。人工检查 20 个查询的前排命中后，既有“我到了→到达/房号”这类有效案例，也有“试用期→打孔/小广告”“csig→小米交易”这类弱相关案例。固定取满 8 条会把噪声和额外提示词一起带入，因此当前候选不采用；下一步先提升按互动动作的召回质量并设置最低相关阈值，宁缺毋滥。</p>
 <p><b>测量范围：</b>同时记录本地检索、实际选入条数与字符、端到端 p50/p95；最终以逐题配对 RT 差为主，避免模型自然波动被误算成检索开销。只有 Judge 更难分清、没有与上下文/Wiki 的可靠矛盾，且 RT 回退可接受，候选才可采用。</p>
</div>
<div class="summary"><h2>结果页安排</h2>
 <p>不再维护第二套 C0 页面。Judge 页面继续承载 Judge 自身验收；本页复用相同的手机布局、完整上下文和筛选能力，作为唯一 C0 固定入口，每轮只更新这里，不生成新地址。</p>
</div>
<div class="ledger"><h2>C0 迭代账单</h2>
 <p class="muted">只统计真正修改过 C0 能力的实验；数据准备、页面修改、失败重试不计入。旧链使用 AI 延续上下文，已按后续纠正降级，不能与真人案例轮次混算。</p>
 {ledger}
</div>
<div class="toolbar" id="toolbar">
 <button class="active" data-filter="all">全部</button><button data-filter="fooled">候选骗过 Judge</button><button data-filter="hard">已核实硬伤</button><button data-filter="risk">待核实风险</button><button data-filter="limitation">评测限制</button><button data-filter="noise">评测噪声</button><button data-filter="changed">回复有变化</button><button data-filter="group">群聊</button><button data-filter="private">私聊</button>
</div>
<p id="count" class="muted"></p>{cards}
</div></main><script>
function bindToolbar(toolbarId, selector, countId){{
 const toolbar=document.querySelector(toolbarId),cases=[...document.querySelectorAll(selector)],count=document.querySelector(countId);
 function show(tag){{let n=0;cases.forEach(c=>{{const ok=tag==='all'||c.dataset.tags.split(' ').includes(tag);c.hidden=!ok;if(ok)n++}});count.textContent=`显示 ${{n}} / ${{cases.length}} 条`;}}
 toolbar.addEventListener('click',e=>{{if(e.target.tagName!=='BUTTON')return;toolbar.querySelectorAll('button').forEach(b=>b.classList.remove('active'));e.target.classList.add('active');show(e.target.dataset.filter);}});show('all');
}}
bindToolbar('#fixed-toolbar','#fixed-cases .case','#fixed-count');bindToolbar('#toolbar','.history > .case','#count');
</script></body></html>"""
    development_start = html.index('<div class="development-only">')
    script_start = html.index('</main><script>')
    public_script = html[script_start:].replace(
        "bindToolbar('#fixed-toolbar','#fixed-cases .case','#fixed-count');bindToolbar('#toolbar','.history > .case','#count');",
        "bindToolbar('#fixed-toolbar','#fixed-cases .case','#fixed-count');",
    )
    return html[:development_start] + public_script


def render_formal_case(case: dict) -> str:
    baseline = case.get("baseline") or {}
    candidate = case.get("candidate") or {}
    baseline_judge = baseline.get("judge") or {}
    candidate_judge = candidate.get("judge") or {}
    baseline_correct = bool(baseline_judge.get("correct"))
    candidate_correct = bool(candidate_judge.get("correct"))
    if baseline_correct and not candidate_correct:
        change_tag, change_label = "improved", "改善：候选更难被 Judge 识别"
    elif not baseline_correct and candidate_correct:
        change_tag, change_label = "regressed", "退步：候选被 Judge 识别"
    else:
        change_tag, change_label = "unchanged", "不变"
    chat_tag = "group" if case.get("chat_type") == "group" else "private"
    tags = [chat_tag, change_tag]
    if not baseline_correct:
        tags.append("baseline_wrong")
    if not candidate_correct:
        tags.append("candidate_wrong")
    if case.get("complex_large_group"):
        tags.append("complex")
    specified_group = str(case.get("specified_group") or case.get("group") or "")
    if specified_group == "correct_common_prosperity_with_luxixi":
        tags.append("prosperity")
    if specified_group.startswith("youzi_group"):
        tags.append("youzi")

    def judge_block(judge: dict) -> str:
        correct = bool(judge.get("correct"))
        return (
            f'<p><b>Judge 结果：</b>{"判对：识别出这是 C0" if correct else "判错：把 C0 当成真人"}'
            f' · 置信度 {esc(judge.get("confidence", "—"))}</p>'
            f'<p class="reason">{esc(judge.get("reason", ""))}</p>'
        )

    return f"""
    <article class="case {' '.join(tags)}" data-tags="{' '.join(tags)}">
      <div class="case-head"><b>{esc(case.get('case_id', ''))}</b><span>{'群聊' if chat_tag == 'group' else '私聊'} · {change_label}</span></div>
      <details><summary>完整真人上下文（展开）</summary><pre>{esc(fixed_context(case))}</pre></details>
      <section class="human"><h3>真人原回复</h3><pre>{esc(reply_text(case.get('human_reply')))}</pre></section>
      <div class="grid">
        <section><h3>基线 C0</h3><pre>{esc(reply_text(baseline.get('c0_reply')))}</pre>{judge_block(baseline_judge)}</section>
        <section><h3>候选 C0</h3><pre>{esc(reply_text(candidate.get('c0_reply')))}</pre>{judge_block(candidate_judge)}</section>
      </div>
    </article>"""


def build_formal_report(payload: dict, output_dir: Path | None = None) -> str:
    summary = payload["summary"]
    cases = payload["cases"]
    segments = summary["segments"]
    workflow = load(DEFAULT_WORKFLOW)
    page_url = workflow["reporting"]["url"]
    baseline_correct = int(summary["baseline_judge_correct"])
    candidate_correct = int(summary["candidate_judge_correct"])
    paired = int(summary["paired"])
    delta = candidate_correct - baseline_correct
    cards = "".join(render_formal_case(case) for case in cases)
    segment_rows = "".join(
        "<tr>"
        f"<td>{esc(segments[key]['name'])}</td>"
        f"<td>{segments[key]['paired']}</td>"
        f"<td>{segments[key]['baseline_judge_correct']}/{segments[key]['paired']} ({segments[key]['baseline_judge_discrimination_accuracy']:.2%})</td>"
        f"<td>{segments[key]['candidate_judge_correct']}/{segments[key]['paired']} ({segments[key]['candidate_judge_discrimination_accuracy']:.2%})</td>"
        "</tr>"
        for key in ("group", "private")
        if key in segments
    )
    public_summary = {
        "status": summary["status"],
        "paired": paired,
        "baseline_judge_correct": baseline_correct,
        "candidate_judge_correct": candidate_correct,
        "baseline_judge_discrimination_accuracy": summary["baseline_judge_discrimination_accuracy"],
        "candidate_judge_discrimination_accuracy": summary["candidate_judge_discrimination_accuracy"],
        "improved": summary["improved"],
        "regressed": summary["regressed"],
        "unchanged": summary["unchanged"],
        "net_improvement": summary["net_improvement"],
        "adopted": summary.get("adopted"),
        "decision": summary.get("decision", ""),
        "segments": segments,
        "response_time_seconds": summary.get("response_time_seconds", {}),
        "paired_exact_sign_test_p": summary.get("paired_exact_sign_test_p"),
        "attributable_new_hard_failures": summary.get("attributable_new_hard_failures"),
        "hard_failure_review": summary.get("hard_failure_review", []),
        "hard_failure_status": summary.get("hard_failure_status", ""),
    }
    out_dir = output_dir or DEFAULT_OUTPUT.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(public_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>C0 固定 240 题正式结果</title>
<style>
:root{{color-scheme:dark;--bg:#0b1020;--card:#151c31;--line:#2a3554;--text:#e7ecf5;--muted:#9fb0d0;--good:#4cc38a;--bad:#ff7b72;--warn:#e3b341}}
*{{box-sizing:border-box}}body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--text);margin:0;padding:12px}}main{{max-width:1040px;margin:auto}}
h1{{font-size:24px;margin:8px 0}}h2{{font-size:18px}}h3{{font-size:14px;margin:0 0 8px}}p{{line-height:1.55}}.muted,.reason{{color:var(--muted)}}
.summary,.case,.decision{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin:12px 0}}.decision{{border-color:{'var(--good)' if summary.get('adopted') else 'var(--bad)'}}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}.metric{{background:#0d1427;border-radius:10px;padding:10px}}.metric b{{display:block;font-size:21px}}small{{display:block;color:var(--muted);margin-top:4px}}
.toolbar{{position:sticky;top:0;z-index:2;background:rgba(11,16,32,.96);padding:9px 0;display:flex;gap:7px;overflow:auto}}button{{border:1px solid var(--line);background:#18223b;color:var(--text);border-radius:999px;padding:7px 11px;white-space:nowrap}}button.active{{border-color:#68a7ff;background:#214277}}
.case-head{{display:flex;justify-content:space-between;gap:8px;color:var(--muted);margin-bottom:9px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}section{{background:#0d1427;border-radius:10px;padding:11px;margin:10px 0}}section.human{{border-left:3px solid var(--good)}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}}details{{margin:10px 0}}summary{{cursor:pointer;color:#b8c8e8}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:8px 5px;border-bottom:1px solid var(--line);text-align:left}}
@media(max-width:720px){{.metrics{{grid-template-columns:1fr 1fr}}.grid{{grid-template-columns:1fr}}.case-head{{display:block}}}}
</style></head><body><main>
<h1>C0 固定 240 题正式 A/B</h1>
<p class="muted">固定地址：{esc(page_url)} · 冻结 Judge · Bot 关闭</p>
<div class="summary"><h2>最重要指标：Judge 识别率</h2>
 <p>Judge 识别率表示 Judge 有多少次成功认出回复来自 C0；<b>数字越低，C0 越像真人。</b></p>
 <div class="metrics">
  <div class="metric"><span>完整有效题目</span><b>{paired}/240</b></div>
  <div class="metric"><span>基线识别率</span><b>{baseline_correct}/{paired}</b><small>{summary['baseline_judge_discrimination_accuracy']:.2%}</small></div>
  <div class="metric"><span>候选识别率</span><b>{candidate_correct}/{paired}</b><small>{summary['candidate_judge_discrimination_accuracy']:.2%}</small></div>
  <div class="metric"><span>候选变化</span><b>{delta:+d}/240</b><small>{summary['candidate_judge_discrimination_accuracy']-summary['baseline_judge_discrimination_accuracy']:+.2%}；正数是退步</small></div>
 </div>
</div>
<div class="decision"><h2>结论：{esc(summary.get('decision', '待决定'))}</h2>
 <p>逐题比较：候选改善 {summary['improved']} 题、退步 {summary['regressed']} 题、不变 {summary['unchanged']} 题，净改善 {summary['net_improvement']:+d} 题。候选与基线的 Judge 识别数差为 {delta:+d}（负数是改善）；双侧精确符号检验 p={summary.get('paired_exact_sign_test_p', '—')}。</p>
 <p><b>硬伤状态：</b>{esc(summary.get('hard_failure_status', '未单独确认'))}</p>
 <p>本次只使用 3 个结果有效条件：生成调用成功、回复非空、固定测试题真人答案没有进入模型输入。时间差异、工具调用以及回复是否与真人相似只记录，不会让题目失效。</p>
</div>
<div class="summary"><h2>不同聊天场景</h2><table><thead><tr><th>场景</th><th>题数</th><th>基线识别率</th><th>候选识别率</th></tr></thead><tbody>{segment_rows}</tbody></table></div>
<div class="summary"><h2>端到端回复时间</h2><p>基线：平均 {summary.get('response_time_seconds', {}).get('baseline', {}).get('mean', '—')} 秒，中位数 {summary.get('response_time_seconds', {}).get('baseline', {}).get('median', '—')} 秒，p95 {summary.get('response_time_seconds', {}).get('baseline', {}).get('p95', '—')} 秒。<br>候选：平均 {summary.get('response_time_seconds', {}).get('candidate', {}).get('mean', '—')} 秒，中位数 {summary.get('response_time_seconds', {}).get('candidate', {}).get('median', '—')} 秒，p95 {summary.get('response_time_seconds', {}).get('candidate', {}).get('p95', '—')} 秒。</p></div>
<div class="toolbar" id="toolbar">
 <button class="active" data-filter="all">全部 240</button><button data-filter="baseline_wrong">基线 Judge 判错</button><button data-filter="candidate_wrong">候选 Judge 判错</button><button data-filter="improved">候选改善</button><button data-filter="regressed">候选退步</button><button data-filter="group">群聊</button><button data-filter="private">私聊</button>
</div>
<p id="count" class="muted"></p><div id="cases">{cards}</div>
</main><script>
const toolbar=document.querySelector('#toolbar'),cases=[...document.querySelectorAll('#cases .case')],count=document.querySelector('#count');
function show(tag){{let n=0;cases.forEach(c=>{{const ok=tag==='all'||c.dataset.tags.split(' ').includes(tag);c.hidden=!ok;if(ok)n++}});count.textContent=`显示 ${{n}} / ${{cases.length}} 条`;}}
toolbar.addEventListener('click',e=>{{if(e.target.tagName!=='BUTTON')return;toolbar.querySelectorAll('button').forEach(b=>b.classList.remove('active'));e.target.classList.add('active');show(e.target.dataset.filter);}});show('all');
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = build_formal_report(load(DEFAULT_FORMAL_AB))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
