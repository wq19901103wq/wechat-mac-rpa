#!/usr/bin/env python3
"""Probe protected C0 for stable missed replies without constructing WeChat Bot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from replay_human_participation_cases import (
    PROJECT_ROOT,
    REPLAY_ENV,
    _generator_class,
    _json,
    _load_env,
    _sha256_text,
    _write_private,
)
from replay_participation_holdout_ab import (
    _build_reply_window_case,
    _run_with_few_shot,
)


def _build_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    cases = []
    for spec in manifest.get("cases", []):
        if spec.get("source_kind") != "weflow_reply_window":
            raise ValueError(f"unsupported source kind: {spec.get('source_kind')}")
        case = _build_reply_window_case(spec)
        if case["expected_action"] != "reply":
            raise ValueError(f"expected action mismatch: {spec['case_id']}")
        cases.append(case)
    expected_case_count = int(manifest.get("expected_case_count", 12))
    if len(cases) != expected_case_count:
        raise ValueError(f"expected {expected_case_count} probe cases, got {len(cases)}")
    if len({item["case_id"] for item in cases}) != len(cases):
        raise ValueError("duplicate case id")
    return cases


def _few_shot_leakage(cases: list[dict[str, Any]]) -> dict[str, Any]:
    configured = os.environ.get("PERSONA_FEW_SHOT_PATH", "")
    path = PROJECT_ROOT / configured
    rows = []
    if configured and path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = {}
    for case in cases:
        evidence = case["human_evidence"]
        targets = [str(item.get("text") or "").strip() for item in evidence["target_unread_original"]]
        answer = str(evidence["human_reply_original"].get("text") or "").strip()
        leaked_rows = []
        for row in rows:
            contexts = [str(value).strip() for value in row.get("context", [])]
            replies = [str(value).strip() for value in row.get("reply", [])]
            if targets and all(target in contexts for target in targets) and answer in replies:
                leaked_rows.append(row.get("id"))
        matches[case["case_id"]] = leaked_rows
    if any(matches.values()):
        raise ValueError("holdout target and human answer found together in active few-shot row")
    return {
        "configured_path": configured,
        "active_row_count": len(rows),
        "case_matching_row_ids": matches,
    }


def _meaningful_reply(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(bool(str(item).strip()) for item in value.values())
    return bool(value)


def _run_c0(generator_class, case: dict[str, Any]) -> dict[str, Any]:
    result = _run_with_few_shot(generator_class, case)
    raw_replies = result.get("replies") or []
    if result.get("status") == "completed" and not result.get("generation_failed"):
        meaningful = [item for item in raw_replies if _meaningful_reply(item)]
        result["empty_reply_count"] = len(raw_replies) - len(meaningful)
        result["action"] = "reply" if meaningful else "silence"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="离线探测 C0 稳定漏回案例")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/participation_missed_reply_probe_cases.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/participation_missed_reply_probe_replay.json",
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
    leakage = _few_shot_leakage(cases)
    validation = {
        "case_count": len(cases),
        "source_chat_count": len({item["human_evidence"]["source"] for item in cases}),
        "c0_generator_sha256": c0_sha,
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

    generator_class = _generator_class(c0_source, "missed_reply_probe_c0")
    report: dict[str, Any] = {
        "schema_version": 1,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "benchmark_kind": "participation_missed_reply_probe",
        "scope": "C0 participation decision only; reply wording is not scored",
        "status": "running",
        "validation": validation,
        "cases": [
            {
                "case_id": case["case_id"],
                "expected_action": "reply",
                "human_evidence": case["human_evidence"],
                "runs": [],
            }
            for case in cases
        ],
    }
    _write_private(args.output, report)
    report_by_id = {item["case_id"]: item for item in report["cases"]}
    for run_number in range(1, args.runs + 1):
        for case in cases:
            result = _run_c0(generator_class, case)
            result["run"] = run_number
            report_by_id[case["case_id"]]["runs"].append(result)
            _write_private(args.output, report)
    report["status"] = "completed"
    report["completed_runs"] = args.runs * len(cases)
    _write_private(args.output, report)
    print(json.dumps({"status": report["status"], **validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
