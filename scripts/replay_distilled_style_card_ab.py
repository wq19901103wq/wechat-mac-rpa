#!/usr/bin/env python3
"""Run an isolated C0 vs distilled-human-style-card offline A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import MethodType
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.replay_same_chat_style_ab import (  # noqa: E402
    PROTECTED_C0_SHA256,
    _load_env,
    _serialize_message,
    _write_private,
)

S4_CASE_IDS = (
    "896b288cb9d301358fbe",  # arrival/status
    "92265673acf1754be333",  # arrival/status
    "1b4902a8f6433cdc1194",  # family gratitude
    "5bddcd3d7bb60ddce1ce",  # family praise
    "0344721ce96c1a046e99",  # wording-based banter
    "136c6cb4d37afdfc34d3",  # familiar identity teasing
    "01b7b9adcdee5f1039f5",  # familiar teasing
    "28ed94e582bae0f82c29",  # factual troubleshooting
    "0e281ad9f57d50716e71",  # fact-sensitive forced answer
    "05c3b5f96ae199f1ed1c",  # believable action and stale-joke risk
    "00b4a6bfdee914f498e2",  # stale-joke risk
    "15f03ea1550042cb5267",  # serious safety context
)

S4_STYLE_CARD = """
<experimental_distilled_human_style_card>
This card was distilled from complete human chats recorded before 2026-04-01. It contains response actions only; it contains no reusable facts or historical lines.

Apply it only after the existing reply-decision, factuality, relationship, safety, and freshness rules allow a reply. If no action clearly fits, follow the existing behavior unchanged.

1. For ordinary arrival or status updates, prefer one very short immediate reaction over a summary, explanation, or invented detail.
2. For praise or thanks from family and familiar people, lightly acknowledge or understate it; do not force a punchline.
3. Answer genuine information needs directly before considering humor. Never trade factual reliability for personality.
4. In light banter with an appropriate relationship, humor may reverse the exact premise or wording in the current unread messages. Do not import people, numbers, objects, events, transactions, or situations from history.
5. Use at most one fresh comic turn, then stop. Do not explain the joke, add a second version, or continue a joke already used by self in history.
6. Do not use humor in serious distress or safety-sensitive contexts. Do not make believable promises to transfer money, send red packets, attend, meet, or perform another real-world action.
</experimental_distilled_human_style_card>
""".strip()


def _install_style_card(generator: Any) -> None:
    original_system_prompt = generator._system_prompt

    def styled_system_prompt(_generator: Any, *args: Any, **kwargs: Any) -> str:
        return original_system_prompt(*args, **kwargs) + "\n\n" + S4_STYLE_CARD

    generator._system_prompt = MethodType(styled_system_prompt, generator)


def _run_one(case: dict[str, Any], version: str) -> dict[str, Any]:
    from src.reply.generator import ReplyGenerator
    from src.tools.tool_registry import ToolRegistry
    from src.utils.qwen_client import QwenClient

    started = time.monotonic()
    result: dict[str, Any] = {"status": "failed", "action": None, "replies": []}
    try:
        client = QwenClient()
        generator = ReplyGenerator(
            llm_client=client,
            memory_engine=None,
            tool_registry=ToolRegistry(),
            judge_worker=None,
        )
        if version == "S4":
            _install_style_card(generator)
        replies = generator.generate(
            case["unreplied"],
            case["all_messages"],
            is_group=case["is_group"],
            tick_id=case["source_tick_id"],
            evaluation_time=case["evaluation_time"],
        )
        generation_failed = bool(generator.last_generation_failed)
        result.update({
            "status": "completed",
            "action": "generation_failed" if generation_failed else ("reply" if replies else "silence"),
            "replies": replies,
            "generation_failed": generation_failed,
            "model": client.model,
            "few_shot_ids": list(generator.last_few_shot_ids),
            "loaded_skills": list(generator.last_loaded_skills),
            "llm_call_count": len(generator.last_llm_calls),
            "tool_call_count": len(generator.last_tool_calls),
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    return result


def _load_human_evidence(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cutoff = 1774972800
    if not rows or any(float(row["timestamp"]) >= cutoff for row in rows):
        raise ValueError("human style evidence must be non-empty and before 2026-04-01")
    moves = Counter(row["semantic_profile"]["response_move"] for row in rows)
    tones = Counter(row["semantic_profile"]["tone"] for row in rows)
    summary = {
        "row_count": len(rows),
        "latest_timestamp": max(float(row["timestamp"]) for row in rows),
        "response_moves": dict(sorted(moves.items())),
        "tones": dict(sorted(tones.items())),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "usage": "private review evidence only; no raw human line is injected into C0 or S4",
    }
    return rows, summary


def _select_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {case["case_id"]: case for case in cases}
    missing = [case_id for case_id in S4_CASE_IDS if case_id not in by_id]
    if missing:
        raise ValueError(f"fixed S4 cases missing: {missing}")
    return [by_id[case_id] for case_id in S4_CASE_IDS]


def main() -> None:
    parser = argparse.ArgumentParser(description="隔离回放 C0 与 S4 抽象真人表达动作卡")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data/cases.db")
    parser.add_argument("--scenarios", type=Path, default=PROJECT_ROOT / "data/reports/private/reply_system_scenarios.json")
    parser.add_argument("--selection", type=Path, default=PROJECT_ROOT / "data/reports/private/reply_quality_gold_cases.json")
    parser.add_argument("--human-evidence", type=Path, default=PROJECT_ROOT / "data/few_shot_v5/verified_human_humor.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/reports/private/distilled_style_card_s4_ab_replay.json")
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError("runs must be positive")

    _load_env()
    os.environ["ENABLE_SELF_REFINE"] = "0"
    os.environ["ENABLE_FACT_CHECK"] = "1"

    from src.benchmarks.reply_system_replay import load_replay_cases

    generator_path = PROJECT_ROOT / "src/reply/generator.py"
    c0_sha = hashlib.sha256(generator_path.read_bytes()).hexdigest()
    if c0_sha != PROTECTED_C0_SHA256:
        raise ValueError("current generator no longer matches protected C0")

    source_fingerprint, all_cases = load_replay_cases(args.db, args.scenarios, args.selection)
    cases = _select_cases(all_cases)
    human_rows, human_summary = _load_human_evidence(args.human_evidence)
    validation = {
        "case_count": len(cases),
        "case_ids": list(S4_CASE_IDS),
        "runs": args.runs,
        "source_fingerprint": source_fingerprint,
        "c0_generator_sha256": c0_sha,
        "candidate_version": "S4",
        "only_intended_difference": "append abstract response-action card distilled from pre-2026-04 complete human chats",
        "historical_raw_lines_in_candidate_prompt": False,
        "human_evidence": human_summary,
        "bot_constructed": False,
        "action_tools_registered": False,
        "memory_enabled": False,
        "judge_worker_enabled": False,
        "self_refine": False,
    }
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return

    report: dict[str, Any] = {
        "schema_version": 1,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "benchmark_kind": "distilled_style_card_s4_ab_replay",
        "scope": "human likeness and humor; reply-rate differences are neutral unless clearly harmful",
        "status": "running",
        "validation": validation,
        "distilled_style_card": S4_STYLE_CARD,
        "human_style_evidence_original": human_rows,
        "cases": [
            {
                "case_id": case["case_id"],
                "is_group": case["is_group"],
                "evaluation_time": case["evaluation_time"].isoformat(),
                "context_original": [_serialize_message(message) for message in case["all_messages"]],
                "target_original": [_serialize_message(message) for message in case["unreplied"]],
                "evidence_usage": "judge may cite original human evidence by response action; generator never receives it",
                "runs": {"C0": [], "S4": []},
            }
            for case in cases
        ],
    }
    _write_private(args.output, report)
    report_by_id = {item["case_id"]: item for item in report["cases"]}
    for run_number in range(1, args.runs + 1):
        for case in cases:
            for version in ("C0", "S4"):
                result = _run_one(case, version)
                result["run"] = run_number
                report_by_id[case["case_id"]]["runs"][version].append(result)
                _write_private(args.output, report)
    results = [
        result
        for item in report["cases"]
        for version in ("C0", "S4")
        for result in item["runs"][version]
    ]
    generation_failures = sum(
        result.get("status") != "completed" or result.get("generation_failed", False)
        for result in results
    )
    report["status"] = "completed" if generation_failures == 0 else "completed_with_generation_failures"
    report["completed_runs"] = len(results)
    report["successful_runs"] = len(results) - generation_failures
    report["generation_failure_count"] = generation_failures
    report["expected_runs"] = args.runs * len(cases) * 2
    _write_private(args.output, report)
    print(json.dumps({"status": report["status"], **validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
