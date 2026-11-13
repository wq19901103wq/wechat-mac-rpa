#!/usr/bin/env python3
"""Probe a deterministic participation gate without constructing WeChat Bot."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from replay_human_participation_cases import (
    PROJECT_ROOT,
    REPLAY_ENV,
    _build_jsonl_case,
    _build_window_case,
    _json,
    _load_env,
    _sha256_text,
    _write_private,
)
from replay_participation_holdout_ab import _build_reply_window_case
from replay_participation_recent_thread_ab import _build_cases
from replay_participation_missed_reply_probe import _few_shot_leakage


PROTECTED_C0_SHA256 = "625b47803fff55a2e18914c5e734fcb2f01ae7c16e5180fdebdc817badd2656b"
GATE_SYSTEM_PROMPT = """你只判断群聊里的本人此刻是否会自然发一条消息，不写回复内容。

目标是预测历史真人的参与习惯，不是让助理尽量回答。按完整上下文判断：
- 有明确指向本人的提问、追问、点名或请求，选择 reply。
- 本人刚主动发起或明确参与当前话题，未读消息直接回答、补充、调侃或推进同一话题，真人自然会继续接话时，也选择 reply。
- 他人互聊、纯图片表情、单独链接、群通知或广播、已经换题，选择 silence。
- 只有硬接梗、复用本人旧包袱或猜测事实才能回复时，选择 silence。
- 不因为“可以说点什么”就选择 reply；判断这个具体的人在这个具体时刻是否真会参与。

只输出 JSON：{"action":"reply 或 silence","target":"被回应的未读消息序号；沉默时为空","reason":"一句简短理由"}。"""

RECENT_CHAIN_RULE = """

额外边界：
- 如果本人刚参与，未读仍沿着本人刚才的观点、类比、共同笑点或共同安排继续同一条语义链，即使没有点名、问句或直接引用，也可选择 reply。
- 如果未读已经变成群友之间连续互答、换了话题，或本人已经完成确认，则选择 silence；不能仅凭关键词相近硬接。"""

FINAL_GAP_RULE = """

最后补充三类真人参与边界：
- 逐条检查未读。较早未读若仍直接延续本人尚未结束的话题，后面夹入一两条无关旁支消息不能自动抹掉这条联系；但群友已经围绕新话题连续互答时仍应沉默。
- 未读若提供了能与本人刚提内容自然形成反转或共同笑点的新信息，可选择 reply；必须只依赖当前聊天里的明确信息，不能为了造梗猜测事实或硬凑关键词。
- 共同安排中，即使本人已经确认，未读若新增了群友为共同活动主动承担的具体行动或便利，真人会自然给一句情绪回应时可选择 reply；普通确认、各自报备、组织通知和群友间安排仍选择 silence。"""


def _message_text(message: Any) -> str:
    text = str(getattr(message, "text", "") or "").strip()
    if text:
        return text
    description = str(getattr(message, "image_description", "") or "").strip()
    if description:
        return description
    message_type = str(getattr(message, "message_type", "") or "媒体消息")
    return f"[{message_type}]"


def _few_shot_context(case: dict[str, Any]) -> tuple[str, list[str]]:
    from src.reply.few_shot import PersonaFewShotRetriever

    configured = os.environ.get("PERSONA_FEW_SHOT_PATH", "")
    path = PROJECT_ROOT / configured
    retriever = PersonaFewShotRetriever(path)
    if not retriever.is_approved():
        raise ValueError("active persona few-shot set is not approved")
    unread_count = len(case["unreplied"])
    prior = list(case["all_messages"][:-unread_count]) if unread_count else list(case["all_messages"])
    context_texts = [str(message.text) for message in prior if getattr(message, "text", "")][-3:]
    route_text = _message_text(case["unreplied"][-1])
    rows = retriever.retrieve(
        query="\n".join(context_texts + [route_text]),
        chat_name=str(case["unreplied"][-1].chat_name or ""),
        is_group=bool(case["is_group"]),
        limit=3,
    )
    content, ids = retriever.render(rows, max_chars=2500)
    if not content:
        return "", []
    prefix = (
        "以下是该真人过去选择参与时的真实例子，只用于学习参与习惯；"
        "不是当前对话事实，也不得复制回复内容。\n"
    )
    return prefix + content, ids


def _gate_prompt(
    case: dict[str, Any],
    use_few_shots: bool = False,
    decision_few_shots: str = "",
    decision_few_shot_ids: list[str] | None = None,
) -> tuple[str, list[str]]:
    from src.models.base import SenderType

    unread_count = len(case["unreplied"])
    all_messages = list(case["all_messages"])
    history = all_messages[:-unread_count] if unread_count else all_messages
    lines = [
        f"聊天类型：{'群聊' if case['is_group'] else '私聊'}",
        "以下历史只用于理解当前对话；本人已经说过的内容不能当作新回复素材。",
        "<history>",
    ]
    for index, message in enumerate(history[-80:], 1):
        sender = "本人" if message.sender_type == SenderType.SELF else str(message.sender or "对方")
        lines.append(f"H{index} {sender}: {_message_text(message)}")
    lines.extend(["</history>", "<unread>"])
    for index, message in enumerate(case["unreplied"], 1):
        sender = "本人" if message.sender_type == SenderType.SELF else str(message.sender or "对方")
        lines.append(f"U{index} {sender}: {_message_text(message)}")
    lines.extend(["</unread>", "现在只判断 action，不生成回复。"])
    few_shot_content, few_shot_ids = _few_shot_context(case) if use_few_shots else ("", [])
    if few_shot_content:
        lines.extend(["", few_shot_content])
    if decision_few_shots:
        lines.extend(["", decision_few_shots])
        few_shot_ids = list(decision_few_shot_ids or [])
    return "\n".join(lines), few_shot_ids


def _parse_gate(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"status": "invalid_json", "action": None}
    if not isinstance(value, dict) or value.get("action") not in {"reply", "silence"}:
        return {"status": "invalid_schema", "action": None}
    return {
        "status": "completed",
        "action": value["action"],
        "target": str(value.get("target") or ""),
        "reason": str(value.get("reason") or ""),
    }


def _run_gate(
    case: dict[str, Any],
    use_few_shots: bool = False,
    decision_few_shots: str = "",
    decision_few_shot_ids: list[str] | None = None,
    recent_chain_rule: bool = False,
    final_gap_rule: bool = False,
) -> dict[str, Any]:
    from src.utils.qwen_client import QwenClient

    client = QwenClient()
    prompt, few_shot_ids = _gate_prompt(
        case,
        use_few_shots=use_few_shots,
        decision_few_shots=decision_few_shots,
        decision_few_shot_ids=decision_few_shot_ids,
    )
    started = time.monotonic()
    raw = client.chat(
        messages=[
            {
                "role": "system",
                "content": GATE_SYSTEM_PROMPT
                + (RECENT_CHAIN_RULE if recent_chain_rule else "")
                + (FINAL_GAP_RULE if final_gap_rule else ""),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=300,
        timeout=30,
        response_format={"type": "json_object"},
        max_retries=0,
    )
    result = _parse_gate(raw if isinstance(raw, str) else "")
    result.update(
        {
            "raw": raw if isinstance(raw, str) else str(raw),
            "duration_seconds": round(time.monotonic() - started, 3),
            "model": client.model,
            "temperature": 0,
            "few_shot_ids": few_shot_ids,
        }
    )
    return result


def _decision_cases(manifest: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs = []
    for spec in manifest.get("cases", []):
        kind = spec.get("source_kind")
        if kind == "weflow_reply_window":
            case = _build_reply_window_case(spec)
        elif kind == "weflow_window":
            case = _build_window_case(spec)
        elif kind == "jsonl":
            case = _build_jsonl_case(spec)
        else:
            raise ValueError(f"unsupported decision example source kind: {kind}")
        if case["expected_action"] != spec.get("expected_action"):
            raise ValueError(f"decision mismatch: {spec['case_id']}")
        if case["evaluation_time"].year >= 2026 and case["evaluation_time"].month >= 4:
            raise ValueError(f"decision example is not before cutoff: {spec['case_id']}")
        pairs.append((spec, case))
    expected = int(manifest["expected_case_count"])
    actions = [case["expected_action"] for _, case in pairs]
    if len(pairs) != expected or len({case["case_id"] for _, case in pairs}) != expected:
        raise ValueError("decision example count or uniqueness mismatch")
    if actions.count("reply") != actions.count("silence"):
        raise ValueError("decision examples must be balanced")
    pair_ids = [spec.get("pair_id") for spec, _ in pairs]
    if any(pair_ids):
        if not all(pair_ids):
            raise ValueError("contrastive examples must all have pair_id")
        for pair_id in set(pair_ids):
            pair_actions = [
                case["expected_action"]
                for spec, case in pairs
                if spec["pair_id"] == pair_id
            ]
            if sorted(pair_actions) != ["reply", "silence"]:
                raise ValueError(f"contrastive pair must contain reply and silence: {pair_id}")
    return pairs


def _render_decision_examples(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[str, list[str]]:
    lines = [
        "以下是测试集之外的真人参与决策示例。只学习参与结构，不按话题词匹配；",
        "示例没有提供真人回复正文，也不能把示例中的事实带入当前聊天。",
    ]
    ids = []
    for index, (spec, case) in enumerate(pairs, 1):
        ids.append(case["case_id"])
        unread_count = len(case["unreplied"])
        history = list(case["all_messages"][:-unread_count]) if unread_count else list(case["all_messages"])
        pair = f" pair=\"{spec['pair_id']}\"" if spec.get("pair_id") else ""
        lines.append(f"<decision_example index=\"{index}\"{pair}>")
        if spec.get("contrast_focus"):
            lines.append(f"对照重点: {spec['contrast_focus']}")
        for message in history[-8:]:
            sender = "本人" if str(message.sender_type) == "SenderType.SELF" else "群友"
            lines.append(f"历史 {sender}: {_message_text(message)}")
        for message in case["unreplied"][:12]:
            sender = "本人" if str(message.sender_type) == "SenderType.SELF" else "群友"
            lines.append(f"未读 {sender}: {_message_text(message)}")
        lines.append(f"action: {case['expected_action']}")
        lines.append(f"结构理由: {spec['decision_basis']}")
        lines.append("</decision_example>")
    return "\n".join(lines), ids


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _input_rows(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for spec, case in pairs:
        for message in case["all_messages"]:
            rows.append(
                {
                    "case_id": case["case_id"],
                    "source": spec["source"],
                    "local_id": getattr(message, "local_id", None),
                    "timestamp": str(getattr(message, "timestamp", "") or ""),
                    "text": _normalized_text(_message_text(message)),
                }
            )
    return rows


def _decision_leakage(
    decision_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    benchmark_manifest: dict[str, Any],
    benchmark_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    benchmark_pairs = list(zip(benchmark_manifest["cases"], benchmark_cases, strict=True))
    decision_rows = _input_rows(decision_pairs)
    benchmark_rows = _input_rows(benchmark_pairs)
    decision_keys = {
        (row["source"], row["local_id"])
        for row in decision_rows
        if row["local_id"] is not None
    }
    benchmark_keys = {
        (row["source"], row["local_id"])
        for row in benchmark_rows
        if row["local_id"] is not None
    }
    shared_message_keys = sorted(decision_keys & benchmark_keys)
    decision_texts = {
        row["text"]
        for row in decision_rows
        if len(row["text"]) >= 8 and not row["text"].startswith("[")
    }
    benchmark_texts = {
        row["text"]
        for row in benchmark_rows
        if len(row["text"]) >= 8 and not row["text"].startswith("[")
    }
    shared_texts = sorted(decision_texts & benchmark_texts)
    benchmark_human_replies = {
        _normalized_text(case["human_evidence"].get("human_reply_original", {}).get("text"))
        for case in benchmark_cases
        if case["expected_action"] == "reply"
    }
    leaked_answers = sorted((benchmark_human_replies - {""}) & decision_texts)
    if shared_message_keys or shared_texts or leaked_answers:
        raise ValueError("decision few-shot overlaps benchmark input or answer")
    return {
        "shared_source_local_ids": shared_message_keys,
        "shared_exact_texts_min_8_chars": shared_texts,
        "benchmark_human_answers_in_examples": leaked_answers,
    }


def _summarize(report: dict[str, Any]) -> dict[str, Any]:
    reply_hits = 0
    reply_total = 0
    silence_hits = 0
    silence_total = 0
    format_failures = 0
    durations = []
    reply_case_hits = {}
    for case in report["cases"]:
        hits = 0
        for run in case["runs"]:
            durations.append(float(run["duration_seconds"]))
            format_failures += run.get("status") != "completed"
            if case["expected_action"] == "reply":
                reply_total += 1
                hit = run.get("action") == "reply"
                reply_hits += hit
                hits += hit
            else:
                silence_total += 1
                silence_hits += run.get("action") == "silence"
        if case["expected_action"] == "reply":
            reply_case_hits[case["case_id"]] = hits
    stable_reply_cases = sum(hits >= 4 for hits in reply_case_hits.values())
    average_duration = sum(durations) / len(durations)
    gates = {
        "silence_30_of_30": silence_hits == 30 and silence_total == 30,
        "reply_at_least_20_of_45": reply_hits >= 20 and reply_total == 45,
        "stable_reply_categories_at_least_5_of_9": stable_reply_cases >= 5,
        "zero_format_failures": format_failures == 0,
        "average_latency_at_most_1_5_seconds": average_duration <= 1.5,
    }
    return {
        "reply_hits": reply_hits,
        "reply_total": reply_total,
        "silence_hits": silence_hits,
        "silence_total": silence_total,
        "reply_case_hits": reply_case_hits,
        "stable_reply_cases": stable_reply_cases,
        "format_failures": format_failures,
        "average_duration_seconds": round(average_duration, 3),
        "acceptance_gates": gates,
        "accepted_for_next_offline_stage": all(gates.values()),
        "content_safety_note": "Gate emits only a participation label; no reply content is generated.",
    }


def _historical_c0_summary(paths: list[Path], case_ids: set[str]) -> dict[str, Any]:
    batches = []
    for path in paths:
        packet = _json(path)
        reply_hits = 0
        reply_total = 0
        silence_hits = 0
        silence_total = 0
        for case in packet.get("cases", []):
            if case.get("case_id") not in case_ids:
                continue
            expected = case.get("expected_action")
            for run in case.get("runs", {}).get("C0", []):
                action = run.get("action")
                if expected == "reply":
                    reply_total += 1
                    reply_hits += action == "reply"
                elif expected == "silence":
                    silence_total += 1
                    silence_hits += action == "silence"
        batches.append(
            {
                "source": str(path.relative_to(PROJECT_ROOT)),
                "reply_hits": reply_hits,
                "reply_total": reply_total,
                "silence_hits": silence_hits,
                "silence_total": silence_total,
            }
        )
    return {"batches": batches}


def main() -> None:
    parser = argparse.ArgumentParser(description="离线探测二阶段参与门控")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--few-shots", action="store_true")
    parser.add_argument("--decision-few-shots", action="store_true")
    parser.add_argument("--recent-chain-rule", action="store_true")
    parser.add_argument("--final-gap-rule", action="store_true")
    parser.add_argument(
        "--decision-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/participation_decision_few_shot_cases.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/participation_recent_thread_ab_cases.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/participation_gate_probe_replay.json",
    )
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError("runs must be positive")
    if args.few_shots and args.decision_few_shots:
        raise ValueError("choose one few-shot strategy")

    _load_env()
    for key, value in REPLAY_ENV.items():
        os.environ[key] = value
    manifest = _json(args.manifest)
    cases = _build_cases(manifest)
    decision_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    decision_content = ""
    decision_ids: list[str] = []
    decision_leakage: dict[str, Any] = {}
    if args.decision_few_shots:
        decision_pairs = _decision_cases(_json(args.decision_manifest))
        decision_content, decision_ids = _render_decision_examples(decision_pairs)
        decision_leakage = _decision_leakage(decision_pairs, manifest, cases)
    c0_source = (PROJECT_ROOT / "src/reply/generator.py").read_text(encoding="utf-8")
    c0_sha = _sha256_text(c0_source)
    if c0_sha != PROTECTED_C0_SHA256 or c0_sha != manifest["c0_generator_sha256"]:
        raise ValueError("current generator no longer matches protected C0")

    case_ids = {case["case_id"] for case in cases}
    leakage = _few_shot_leakage(
        [case for case in cases if case["expected_action"] == "reply"]
    )
    historical_c0 = _historical_c0_summary(
        [
            PROJECT_ROOT / "data/reports/private/participation_recent_thread_ab_replay.json",
            PROJECT_ROOT / "data/reports/private/participation_self_context_ab_replay.json",
            PROJECT_ROOT / "data/reports/private/participation_decision_context_ab_replay.json",
        ],
        case_ids,
    )
    validation = {
        "case_count": len(cases),
        "reply_cases": sum(case["expected_action"] == "reply" for case in cases),
        "silence_cases": sum(case["expected_action"] == "silence" for case in cases),
        "runs": args.runs,
        "c0_generator_sha256": c0_sha,
        "c0_untouched": True,
        "candidate_kind": "standalone participation gate probe",
        "persona_few_shots_enabled": args.few_shots,
        "decision_few_shots_enabled": args.decision_few_shots,
        "recent_chain_rule_enabled": args.recent_chain_rule,
        "final_gap_rule_enabled": args.final_gap_rule,
        "decision_example_count": len(decision_pairs),
        "decision_examples_balanced": bool(decision_pairs) and len(decision_pairs) == 12,
        "decision_few_shot_leakage": decision_leakage,
        "persona_few_shot_leakage": leakage,
        "human_replies_excluded_from_model_input": True,
        "temperature": 0,
        "seed_supported_by_current_client": False,
        "bot_constructed": False,
        "reply_content_generated": False,
        "judge_worker_enabled": False,
        "historical_c0": historical_c0,
    }
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return

    report: dict[str, Any] = {
        "schema_version": 1,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "benchmark_kind": "participation_gate_probe",
        "scope": "participation decision only; this probe does not generate reply content",
        "status": "running",
        "validation": validation,
        "gate_system_prompt": GATE_SYSTEM_PROMPT
        + (RECENT_CHAIN_RULE if args.recent_chain_rule else "")
        + (FINAL_GAP_RULE if args.final_gap_rule else ""),
        "cases": [
            {
                "case_id": case["case_id"],
                "expected_action": case["expected_action"],
                "human_evidence": case["human_evidence"],
                "runs": [],
            }
            for case in cases
        ],
    }
    _write_private(args.output, report)
    report_by_id = {case["case_id"]: case for case in report["cases"]}
    for run_number in range(1, args.runs + 1):
        for case in cases:
            result = _run_gate(
                case,
                use_few_shots=args.few_shots,
                decision_few_shots=decision_content,
                decision_few_shot_ids=decision_ids,
                recent_chain_rule=args.recent_chain_rule,
                final_gap_rule=args.final_gap_rule,
            )
            result["run"] = run_number
            report_by_id[case["case_id"]]["runs"].append(result)
            _write_private(args.output, report)

    report["status"] = "completed"
    report["completed_runs"] = args.runs * len(cases)
    report["final_c0_generator_sha256"] = _sha256_text(
        (PROJECT_ROOT / "src/reply/generator.py").read_text(encoding="utf-8")
    )
    if report["final_c0_generator_sha256"] != c0_sha:
        raise ValueError("protected C0 changed during probe")
    report["summary"] = _summarize(report)
    _write_private(args.output, report)
    print(json.dumps({"status": report["status"], **validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
