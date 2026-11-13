#!/usr/bin/env python3
"""Compare current retrieval with exact-move-gated same-chat preference."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_human_ai_judge import _write_private  # noqa: E402
from scripts.run_situation_c0_ab import (  # noqa: E402
    DEFAULT_DEVELOPMENT,
    DEFAULT_EXPOSED,
    DEFAULT_FEW_SHOT,
    DEFAULT_FIXED_TEST,
    DEFAULT_PREVIOUS_AB,
    _excluded_case_ids,
    _select_cases,
)
from src.reply.few_shot import PersonaFewShotRetriever, _latest_turn_form  # noqa: E402


DEFAULT_OUTPUT = ROOT / "data/reports/private/same_chat_exact_move_gate_fixed80_v1/results.json"


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(percentile * len(ordered)) - 1)], 3)


def _timing(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50_ms": round(statistics.median(values), 3) if values else None,
        "p95_ms": _percentile(values, 0.95),
        "mean_ms": round(statistics.mean(values), 3) if values else None,
    }


def run(args: argparse.Namespace) -> None:
    development = json.loads(args.development.read_text(encoding="utf-8"))
    exposed = json.loads(args.exposed.read_text(encoding="utf-8"))
    fixed_test = json.loads(args.fixed_test.read_text(encoding="utf-8"))
    previous = json.loads(args.previous_ab.read_text(encoding="utf-8"))
    excluded = _excluded_case_ids(exposed, [previous], fixed_test)
    if args.case_manifest:
        manifest = json.loads(args.case_manifest.read_text(encoding="utf-8"))
        rows_by_id = {str(row["id"]): row for row in development["cases"]}
        case_ids = [str(case_id) for case_id in manifest["case_ids"]]
        missing = [case_id for case_id in case_ids if case_id not in rows_by_id]
        if missing:
            raise ValueError(f"case manifest contains {len(missing)} missing development ids")
        rows = [rows_by_id[case_id] for case_id in case_ids[: args.max_cases]]
    else:
        rows = _select_cases(
            development["cases"],
            excluded,
            args.max_cases,
            "same-chat-after-situation",
        )
    candidate_options = (
        {"prefer_latest_turn_form": True}
        if args.feature == "latest-turn-form"
        else {"enable_information_gap_response_move": True}
        if args.feature == "information-gap-response-move"
        else {"prefer_same_chat_after_exact_move": True}
    )
    retrievers = {
        "baseline": PersonaFewShotRetriever(
            args.few_shot,
            enable_precise_situation=True,
        ),
        "candidate": PersonaFewShotRetriever(
            args.few_shot,
            enable_precise_situation=True,
            **candidate_options,
        ),
    }
    for retriever in retrievers.values():
        retriever.retrieve("warmup", "", False, limit=args.few_shot_count)

    timings: dict[str, list[float]] = {name: [] for name in retrievers}
    cases = []
    for row in rows:
        query = "\n".join(str(value) for value in row.get("context") or [])
        conditions = {}
        for name, retriever in retrievers.items():
            selected = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                selected = retriever.retrieve(
                    query=query,
                    chat_name="",
                    chat_id=str(row.get("chat_id") or ""),
                    is_group=row.get("relationship") == "group",
                    relationship=str(row.get("relationship") or "acquaintance"),
                    limit=args.few_shot_count,
                    exclude_ids={str(row["id"])},
                    use_situation=True,
                )
                timings[name].append((time.perf_counter() - started) * 1000)
            rendered, selected_ids = retriever.render(selected, max_chars=args.max_chars)
            conditions[name] = {
                "selected_ids": selected_ids,
                "same_chat_count": sum(item.get("chat_id") == row.get("chat_id") for item in selected),
                "rendered_chars": len(rendered),
                "latest_form_matches": sum(
                    _latest_turn_form(str((item.get("context") or [""])[-1]))
                    == _latest_turn_form(str((row.get("context") or [""])[-1]))
                    for item in selected
                ),
                "examples": [
                    {
                        "id": item.get("id"),
                        "chat_id": item.get("chat_id"),
                        "context": item.get("context") or [],
                        "reply": item.get("reply") or [],
                    }
                    for item in selected
                ],
            }
        cases.append({
            "case_id": row["id"],
            "chat_id": row.get("chat_id"),
            "relationship": row.get("relationship"),
            "context": row.get("context") or [],
            "human_reply": row.get("reply") or [],
            "changed": conditions["baseline"]["selected_ids"] != conditions["candidate"]["selected_ids"],
            "conditions": conditions,
        })

    result = {
        "schema_version": 1,
        "purpose": "development-only retrieval and RT check; not a formal evaluation result",
        "only_variable": (
            "rank exact latest-turn-form matches +4 and mismatches -1"
            if args.feature == "latest-turn-form"
            else "classify source, role, evidence, and required-field gaps as verify_or_request_detail; all other retrieval settings remain unchanged"
            if args.feature == "information-gap-response-move"
            else "prefer current-chat examples only when situation tags overlap and expected_response_move matches exactly; cross-chat fallback remains enabled"
        ),
        "fixed_240_excluded": True,
        "few_shot_pool": str(args.few_shot.relative_to(ROOT)),
        "few_shot_count": args.few_shot_count,
        "few_shot_max_chars": args.max_chars,
        "summary": {
            "cases": len(cases),
            "group_cases": sum(case["relationship"] == "group" for case in cases),
            "private_cases": sum(case["relationship"] != "group" for case in cases),
            "changed_cases": sum(case["changed"] for case in cases),
            "baseline_same_chat_examples": sum(case["conditions"]["baseline"]["same_chat_count"] for case in cases),
            "candidate_same_chat_examples": sum(case["conditions"]["candidate"]["same_chat_count"] for case in cases),
            "baseline_latest_form_matches": sum(case["conditions"]["baseline"]["latest_form_matches"] for case in cases),
            "candidate_latest_form_matches": sum(case["conditions"]["candidate"]["latest_form_matches"] for case in cases),
            "timing": {name: _timing(values) for name, values in timings.items()},
        },
        "cases": cases,
    }
    _write_private(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--exposed", type=Path, default=DEFAULT_EXPOSED)
    parser.add_argument("--fixed-test", type=Path, default=DEFAULT_FIXED_TEST)
    parser.add_argument("--previous-ab", type=Path, default=DEFAULT_PREVIOUS_AB)
    parser.add_argument("--few-shot", type=Path, default=DEFAULT_FEW_SHOT)
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cases", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--few-shot-count", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=2500)
    parser.add_argument(
        "--feature",
        choices=("same-chat-exact-move", "latest-turn-form", "information-gap-response-move"),
        default="same-chat-exact-move",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
