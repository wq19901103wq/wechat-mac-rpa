#!/usr/bin/env python3
"""Preflight the one canonical C0/Judge iteration flow without calling a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.badcase.judge_config import (  # noqa: E402
    current_judge_baseline_id,
    load_current_judge,
    missing_official_baseline_fields,
    require_official_judge_runnable,
)
from src.reply.c0_config import current_c0_baseline_id, load_current_c0  # noqa: E402


def preflight(iteration_object: str, dataset: str, single_change: str) -> dict[str, Any]:
    if iteration_object not in {"c0", "judge"}:
        raise ValueError("iteration_object must be c0 or judge")
    if dataset not in {"development", "fixed_test"}:
        raise ValueError("dataset must be development or fixed_test")
    if not single_change.strip():
        raise ValueError("single_change must be non-empty")

    load_current_c0()
    result: dict[str, Any] = {
        "status": "ready",
        "model_calls_started": 0,
        "iteration_object": iteration_object,
        "dataset": dataset,
        "dataset_purpose": "tuning_and_selection" if dataset == "development" else "independent_validation_only",
        "single_change": single_change.strip(),
        "current_c0_baseline_id": current_c0_baseline_id(),
    }
    if iteration_object == "c0":
        result["next_command"] = "scripts/run_situation_c0_ab.py"
        return result

    judge = load_current_judge()
    judge_baseline_id = (
        current_judge_baseline_id()
        if judge.get("status") == "adopted" and judge.get("official_judge_baseline") is True
        else None
    )
    result.update({
        "current_judge_baseline_id": judge_baseline_id,
        "official_judge_target_cases": 240,
        "official_judge_runnable": judge.get("runnable_as_official") is True,
        "missing_official_baseline_fields": missing_official_baseline_fields(judge),
        "next_command": "scripts/evaluate_human_ai_judge.py --run-codex-api-judge --official-judge",
    })
    require_official_judge_runnable(judge)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-object", choices=("c0", "judge"), required=True)
    parser.add_argument("--dataset", choices=("development", "fixed_test"), required=True)
    parser.add_argument("--single-change", required=True)
    args = parser.parse_args()
    try:
        result = preflight(args.iteration_object, args.dataset, args.single_change)
    except (RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "model_calls_started": 0, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(2) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
