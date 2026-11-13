#!/usr/bin/env python3
"""Freeze the Judge training boundary and one reproducible fixed test set."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


QUOTAS = {
    "fixed_test": {
        "specified_common": 20,
        "specified_youzi": 20,
        "other_group_complex": 62,
        "other_group": 58,
        "private": 80,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _category(case: dict[str, Any]) -> str:
    specified = case.get("specified_group")
    if specified == "correct_common_prosperity_with_luxixi":
        return "specified_common"
    if specified == "youzi_group_2":
        return "specified_youzi"
    if case.get("chat_type") == "private":
        return "private"
    if case.get("complex_large_group"):
        return "other_group_complex"
    return "other_group"


def freeze_splits(source: dict[str, Any], *, seed: str) -> dict[str, Any]:
    cases = source.get("cases") or []
    if len(cases) != 240:
        raise ValueError(f"expected 240 source cases, found {len(cases)}")

    buckets: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        buckets.setdefault(_category(case), []).append(case)
    for category, values in buckets.items():
        values.sort(
            key=lambda item: hashlib.sha256(
                f"{seed}|{item['internal_id']}|{item['event_cluster']}".encode()
            ).hexdigest()
        )

    offsets = {category: 0 for category in buckets}
    split_cases: dict[str, list[dict[str, Any]]] = {}
    for split, quotas in QUOTAS.items():
        selected: list[dict[str, Any]] = []
        for category, count in quotas.items():
            start = offsets.get(category, 0)
            values = buckets.get(category, [])
            chosen = values[start : start + count]
            if len(chosen) != count:
                raise ValueError(
                    f"insufficient {category} cases for {split}: "
                    f"needed {count}, found {len(chosen)}"
                )
            offsets[category] = start + count
            selected.extend(chosen)
        split_cases[split] = selected

    if any(offsets[name] != len(values) for name, values in buckets.items()):
        raise ValueError("quota plan did not consume every source case exactly once")

    all_ids = [case["internal_id"] for values in split_cases.values() for case in values]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("case overlap detected across splits")
    all_events = [case["event_cluster"] for values in split_cases.values() for case in values]
    if len(all_events) != len(set(all_events)):
        raise ValueError("event-cluster overlap detected across splits")

    result_splits: dict[str, Any] = {}
    for split, values in split_cases.items():
        result_splits[split] = {
            "status": "proposed_pending_exposure_audit",
            "purpose": "fixed independent acceptance only; never use case-level errors for rules, prompts, or few-shot selection",
            "count": len(values),
            "category_counts": {
                category: sum(_category(case) == category for case in values)
                for category in QUOTAS[split]
            },
            "case_ids": [case["internal_id"] for case in values],
            "event_clusters": [case["event_cluster"] for case in values],
        }

    return {
        "schema_version": 1,
        "status": "proposed_pending_exposure_audit",
        "seed": seed,
        "policy": {
            "development": "all previously viewed, discussed, scored, or prompt-tuning cases",
            "fixed_test": "the only candidate sealed acceptance set; activate only after a clean exposure audit",
            "no_cross_split_event_overlap": True,
            "development_only_for_error_analysis_and_few_shot_selection": True,
            "fixed_test_case_level_feedback_forbidden": True,
        },
        "development": {
            "status": "exposed_development_only",
            "count": 180,
            "sources": [
                "data/reports/private/human_ai_judge_benchmark_v2/dev_replay.json",
                "data/reports/private/human_ai_judge_benchmark_v2/validation_replay.json",
                "data/reports/private/human_ai_judge_benchmark_v2/final_replay.json",
            ],
            "reason": "answers and/or case-level results were already used during Judge iteration",
        },
        "source_case_count": len(cases),
        "splits": result_splits,
    }


def _public_summary(ledger: dict[str, Any], *, source_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": ledger["schema_version"],
        "status": ledger["status"],
        "frozen_at": ledger["frozen_at"],
        "source_sha256": source_sha256,
        "development": ledger["development"],
        "splits": {
            name: {
                "status": value["status"],
                "purpose": value["purpose"],
                "count": value["count"],
                "category_counts": value["category_counts"],
            }
            for name, value in ledger["splits"].items()
        },
        "privacy": "case IDs, event IDs, contexts, and answers are omitted",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/reports/private/adversarial_stage2_holdout_v1/manifest.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/reports/private/judge_dataset_splits_v1"),
    )
    parser.add_argument("--seed", default="judge-fixed-splits-v1")
    args = parser.parse_args()

    source = json.loads(args.source.read_text())
    ledger = freeze_splits(source, seed=args.seed)
    ledger["frozen_at"] = datetime.now(timezone.utc).isoformat()
    ledger["source"] = str(args.source)
    ledger["source_sha256"] = _sha256(args.source)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    private_path = args.out_dir / "private_ledger.json"
    public_path = args.out_dir / "public_summary.json"
    private_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n")
    public_path.write_text(
        json.dumps(
            _public_summary(ledger, source_sha256=ledger["source_sha256"]),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(_public_summary(ledger, source_sha256=ledger["source_sha256"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
