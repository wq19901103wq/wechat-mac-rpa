#!/usr/bin/env python3
"""Run an isolated participation holdout A/B without constructing WeChat Bot."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from replay_human_participation_cases import (
    PROJECT_ROOT,
    REPLAY_ENV,
    _build_window_case,
    _display_message,
    _generator_class,
    _json,
    _load_env,
    _sha256_text,
    _write_private,
    _chat_message,
)

CANDIDATE_RULE = (
    "            \"<rule>没有 @、问号或点名不等于必须沉默；仅当 history 明确表明你正持续参与同一件事，"
    "且 unread 是对你上一轮参与的直接承接、共同决定或结果直接关系你，或对方明显在等你确认时才回复。"
    "普通群聊、广播、链接、图片和他人之间对话仍沉默。</rule>\",\n"
)
ANCHOR = (
    "            \"<rule>先扫描全部 unread 再决定是否参与，不能只看最后一条；"
    "只要其中有明确指向你的提问、追问、叫人或请求，就应回应。</rule>\",\n"
)


def _message_by_local_id(messages: list[dict[str, Any]], local_id: int) -> dict[str, Any]:
    matches = [item for item in messages if item.get("localId") == local_id]
    if len(matches) != 1:
        raise ValueError(f"expected one message for localId={local_id}, got {len(matches)}")
    return matches[0]


def _build_reply_window_case(spec: dict[str, Any]) -> dict[str, Any]:
    source_path = PROJECT_ROOT / spec["source"]
    packet = _json(source_path)
    messages = packet.get("messages", [])
    context_start = _message_by_local_id(messages, spec["context_start_local_id"])
    unread_start = _message_by_local_id(messages, spec["unread_start_local_id"])
    unread_end = _message_by_local_id(messages, spec["unread_end_local_id"])
    human_reply = _message_by_local_id(messages, spec["human_reply_local_id"])
    start_id = context_start["localId"]
    unread_start_id = unread_start["localId"]
    unread_end_id = unread_end["localId"]
    human_reply_id = human_reply["localId"]
    if not (start_id <= unread_start_id <= unread_end_id < human_reply_id):
        raise ValueError(f"invalid localId ordering: {spec['case_id']}")
    selected = [item for item in messages if start_id <= item.get("localId", -1) <= unread_end_id]
    unread = [item for item in selected if unread_start_id <= item["localId"] <= unread_end_id]
    if not selected or not unread or not human_reply.get("isSend"):
        raise ValueError(f"invalid reply window: {spec['case_id']}")
    if any(item.get("isSend") for item in unread):
        raise ValueError(f"human reply leaked into unread: {spec['case_id']}")
    session = packet.get("session") or {}
    chat_name = str(session.get("displayName") or session.get("name") or source_path.stem)
    evaluation_time = datetime.strptime(str(unread_end["formattedTime"]), "%Y-%m-%d %H:%M:%S")
    return {
        "case_id": spec["case_id"],
        "expected_action": "reply",
        "is_group": True,
        "evaluation_time": evaluation_time,
        "all_messages": [_chat_message(item, chat_name) for item in selected],
        "unreplied": [_chat_message(item, chat_name) for item in unread],
        "human_evidence": {
            "decision": "reply",
            "source": spec["source"],
            "context_original": [_display_message(item) for item in selected],
            "target_unread_original": [_display_message(item) for item in unread],
            "human_reply_original": _display_message(human_reply),
            "human_reply_excluded_from_model_input": True,
        },
    }


def _build_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    cases = []
    for spec in manifest.get("cases", []):
        if spec.get("source_kind") == "weflow_reply_window":
            case = _build_reply_window_case(spec)
        elif spec.get("source_kind") == "weflow_window":
            case = _build_window_case(spec)
        else:
            raise ValueError(f"unsupported source kind: {spec.get('source_kind')}")
        if case["expected_action"] != spec.get("expected_action"):
            raise ValueError(f"expected action mismatch: {spec['case_id']}")
        if case["evaluation_time"] >= datetime(2026, 4, 1):
            raise ValueError(f"case is not before cutoff: {spec['case_id']}")
        cases.append(case)
    if len(cases) != 11:
        raise ValueError(f"expected 11 cases, got {len(cases)}")
    if sum(case["expected_action"] == "reply" for case in cases) != 5:
        raise ValueError("holdout must contain 5 reply cases")
    if sum(case["expected_action"] == "silence" for case in cases) != 6:
        raise ValueError("holdout must contain 6 silence cases")
    return cases


def _candidate_source(c0_source: str) -> str:
    if c0_source.count(ANCHOR) != 1:
        raise ValueError("candidate insertion anchor did not match exactly once")
    return c0_source.replace(ANCHOR, ANCHOR + CANDIDATE_RULE, 1)


def _few_shot_leakage(cases: list[dict[str, Any]]) -> dict[str, Any]:
    configured = os.environ.get("PERSONA_FEW_SHOT_PATH", "")
    path = PROJECT_ROOT / configured
    text = path.read_text(encoding="utf-8") if configured and path.exists() else ""
    matches = {}
    for case in cases:
        if case["expected_action"] != "reply":
            continue
        evidence = case["human_evidence"]
        context_texts = [str(item.get("text") or "").strip() for item in evidence["target_unread_original"]]
        reply_text = str(evidence["human_reply_original"].get("text") or "").strip()
        matches[case["case_id"]] = {
            "target_and_human_reply_both_present": bool(
                context_texts and all(value in text for value in context_texts) and reply_text in text
            )
        }
    if any(item["target_and_human_reply_both_present"] for item in matches.values()):
        raise ValueError("holdout target and answer found together in active few-shot file")
    return {"configured_path": configured, "case_matches": matches}


def _run_with_few_shot(generator_class, case: dict[str, Any]) -> dict[str, Any]:
    from src.tools.tool_registry import ToolRegistry
    from src.utils.qwen_client import QwenClient

    started = time.monotonic()
    result: dict[str, Any] = {"status": "failed", "action": None, "replies": []}
    try:
        client = QwenClient()
        generator = generator_class(
            llm_client=client,
            memory_engine=None,
            tool_registry=ToolRegistry(),
            judge_worker=None,
        )
        replies = generator.generate(
            case["unreplied"],
            case["all_messages"],
            is_group=case["is_group"],
            tick_id=0,
            evaluation_time=case["evaluation_time"],
        )
        generation_failed = bool(generator.last_generation_failed)
        result.update(
            {
                "status": "completed",
                "action": (
                    "generation_failed"
                    if generation_failed
                    else ("reply" if replies else "silence")
                ),
                "replies": replies,
                "generation_failed": generation_failed,
                "model": client.model,
                "active_llm": generator.last_active_llm,
                "few_shot_ids": list(generator.last_few_shot_ids),
                "loaded_skills": generator.last_loaded_skills,
                "llm_call_count": len(generator.last_llm_calls),
                "tool_call_count": len(generator.last_tool_calls),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="隔离回放参与判断 holdout A/B")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/participation_holdout_cases.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/participation_holdout_ab_replay.json",
    )
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError("runs must be positive")
    _load_env()
    for key, value in REPLAY_ENV.items():
        os.environ[key] = value
    manifest = _json(args.manifest)
    cases = _build_cases(manifest)
    c0_source = (PROJECT_ROOT / "src/reply/generator.py").read_text(encoding="utf-8")
    c0_sha = _sha256_text(c0_source)
    if c0_sha != manifest["c0_generator_sha256"]:
        raise ValueError("current generator no longer matches protected C0")
    candidate_source = _candidate_source(c0_source)
    candidate_sha = _sha256_text(candidate_source)
    leakage = _few_shot_leakage(cases)
    validation = {
        "case_count": len(cases),
        "reply_cases": 5,
        "silence_cases": 6,
        "c0_generator_sha256": c0_sha,
        "candidate_generator_sha256": candidate_sha,
        "candidate_rule": CANDIDATE_RULE.strip(),
        "candidate_is_in_memory_only": True,
        "all_cases_before_cutoff": True,
        "human_replies_excluded_from_model_input": True,
        "few_shot_leakage": leakage,
        "bot_constructed": False,
        "action_tools_registered": False,
        "memory_enabled": False,
        "judge_worker_enabled": False,
        "model": os.environ.get("LLM_MODEL"),
        "config": REPLAY_ENV,
    }
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return
    classes = {
        "C0": _generator_class(c0_source, "holdout_c0"),
        "H1": _generator_class(candidate_source, "holdout_h1"),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "benchmark_kind": "participation_holdout_ab_replay",
        "scope": "participation decision only; reply wording is not judged",
        "status": "running",
        "validation": validation,
        "cases": [
            {
                "case_id": case["case_id"],
                "expected_action": case["expected_action"],
                "human_evidence": case["human_evidence"],
                "runs": {"C0": [], "H1": []},
            }
            for case in cases
        ],
    }
    _write_private(args.output, report)
    report_by_id = {item["case_id"]: item for item in report["cases"]}
    for run_number in range(1, args.runs + 1):
        for version in ("C0", "H1"):
            for case in cases:
                result = _run_with_few_shot(classes[version], case)
                result["run"] = run_number
                report_by_id[case["case_id"]]["runs"][version].append(result)
                _write_private(args.output, report)
    report["status"] = "completed"
    report["completed_runs"] = args.runs * len(cases) * len(classes)
    report["expected_runs"] = report["completed_runs"]
    _write_private(args.output, report)
    print(json.dumps({"status": report["status"], **validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
