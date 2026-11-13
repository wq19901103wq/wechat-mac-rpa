#!/usr/bin/env python3
"""A/B a narrow recent-thread participation rule without constructing WeChat Bot."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from replay_human_participation_cases import (
    PROJECT_ROOT,
    REPLAY_ENV,
    _build_window_case,
    _generator_class,
    _json,
    _load_env,
    _sha256_text,
    _write_private,
)
from replay_participation_holdout_ab import _build_reply_window_case
from replay_participation_missed_reply_probe import _few_shot_leakage, _run_c0


CANDIDATE_RULE = (
    "            \"<rule>群聊参与例外：若你在最近几分钟内主动发起或明确参与了当前话题，且 unread "
    "在同一话题内直接回答、补充、延续你刚才的内容或推进共同安排，应视为对话仍在继续，即使没有 @、"
    "问号或点名也应参与。只有语义直接承接才适用；已经换题、只剩他人互聊、纯图片表情、广播，或只能"
    "硬接梗时仍应沉默。</rule>\",\n"
)
CANDIDATE_LABEL = "C1"
ANCHOR = (
    "            \"<rule>先扫描全部 unread 再决定是否参与，不能只看最后一条；"
    "只要其中有明确指向你的提问、追问、叫人或请求，就应回应。</rule>\",\n"
)


def _candidate_source(c0_source: str) -> str:
    if c0_source.count(ANCHOR) != 1:
        raise ValueError("candidate insertion anchor did not match exactly once")
    return c0_source.replace(ANCHOR, ANCHOR + CANDIDATE_RULE, 1)


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
    expected = int(manifest["expected_case_count"])
    if len(cases) != expected or len({case["case_id"] for case in cases}) != expected:
        raise ValueError("case count or uniqueness mismatch")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="离线 A/B 近期本人话题直接承接规则")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/participation_recent_thread_ab_cases.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/participation_recent_thread_ab_replay.json",
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
    reply_cases = [case for case in cases if case["expected_action"] == "reply"]
    silence_cases = [case for case in cases if case["expected_action"] == "silence"]
    leakage = _few_shot_leakage(reply_cases)
    validation = {
        "case_count": len(cases),
        "reply_cases": len(reply_cases),
        "silence_cases": len(silence_cases),
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
        "C0": _generator_class(c0_source, "recent_thread_c0"),
        CANDIDATE_LABEL: _generator_class(candidate_source, f"recent_thread_{CANDIDATE_LABEL.lower()}"),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "benchmark_kind": "participation_recent_thread_ab_replay",
        "scope": "participation decision only; reply wording is not judged",
        "status": "running",
        "validation": validation,
        "cases": [
            {
                "case_id": case["case_id"],
                "expected_action": case["expected_action"],
                "human_evidence": case.get("human_evidence"),
                "runs": {"C0": [], CANDIDATE_LABEL: []},
            }
            for case in cases
        ],
    }
    _write_private(args.output, report)
    report_by_id = {item["case_id"]: item for item in report["cases"]}
    for run_number in range(1, args.runs + 1):
        for version in ("C0", CANDIDATE_LABEL):
            for case in cases:
                result = _run_c0(classes[version], case)
                result["run"] = run_number
                report_by_id[case["case_id"]]["runs"][version].append(result)
                _write_private(args.output, report)
    report["status"] = "completed"
    report["completed_runs"] = args.runs * len(cases) * len(classes)
    _write_private(args.output, report)
    print(json.dumps({"status": report["status"], **validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
