#!/usr/bin/env python3
"""Evaluate Judge calibration and optional Bot A/B results without model calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.reply_quality_evaluation import (  # noqa: E402
    evaluate_ab,
    evaluate_judge_calibration,
)


def _read(path: Path | None, default):
    return json.loads(path.read_text(encoding="utf-8")) if path else default


def _cases(value):
    return value.get("cases", value) if isinstance(value, dict) else value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="计算回复质量 Judge 校准与 A/B 门禁")
    parser.add_argument(
        "--gold",
        type=Path,
        default=PROJECT_ROOT / "data/reports/private/reply_quality_gold_cases.json",
    )
    parser.add_argument("--judge-predictions", type=Path)
    parser.add_argument("--judge-policy", type=Path)
    parser.add_argument(
        "--judge-output",
        type=Path,
        default=PROJECT_ROOT / "data/reports/reply_quality_judge.json",
    )
    parser.add_argument("--ab-comparisons", type=Path)
    parser.add_argument("--ab-policy", type=Path)
    parser.add_argument(
        "--ab-output",
        type=Path,
        default=PROJECT_ROOT / "data/reports/reply_quality_ab.json",
    )
    args = parser.parse_args()

    gold = _cases(_read(args.gold, []))
    predictions = _cases(_read(args.judge_predictions, []))
    judge_report = evaluate_judge_calibration(
        gold, predictions, _read(args.judge_policy, {})
    )
    _write(args.judge_output, judge_report)
    print(f"Judge report: {args.judge_output}")
    print(f"Judge status: {judge_report['status']}")

    if args.ab_comparisons:
        comparisons = _cases(_read(args.ab_comparisons, []))
        ab_report = evaluate_ab(comparisons, judge_report, _read(args.ab_policy, {}))
        _write(args.ab_output, ab_report)
        print(f"A/B report: {args.ab_output}")
        print(f"A/B status: {ab_report['status']}")


if __name__ == "__main__":
    main()
