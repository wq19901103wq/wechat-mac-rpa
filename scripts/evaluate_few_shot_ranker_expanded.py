#!/usr/bin/env python3
"""Compare frozen expanded ranker selections with production top-1 labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.badcase.judge_config import current_judge_baseline_id, load_current_judge  # noqa: E402
from src.reply.c0_config import current_c0_baseline_id  # noqa: E402


DEFAULT_SELECTIONS = (
    ROOT / "data/reports/private/few_shot_ranker_xgboost_expanded480_plan/selections.jsonl"
)
DEFAULT_SAMPLES = (
    ROOT / "data/reports/private/few_shot_ranker_xgboost_expanded480_label_samples/samples.jsonl",
)
DEFAULT_LABELS = tuple(
    ROOT / f"data/reports/private/few_shot_ranker_xgboost_expanded480_labels/shard{index}/labels.jsonl"
    for index in range(4)
)
DEFAULT_OUTPUT = (
    ROOT / "data/reports/private/few_shot_ranker_xgboost_expanded480_evaluation/result.json"
)
PRIVATE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700
BOOTSTRAP_ITERATIONS = 2_000
SUPPORTED_VARIANTS = ("lr", "xgboost")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recorded_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _validate_sources(
    selections_path: Path,
    sample_paths: tuple[Path, ...],
    label_paths: tuple[Path, ...],
    *,
    candidate_variant: str,
    judge_baseline_id: str,
) -> None:
    plan_manifest_path = selections_path.parent / "manifest.json"
    plan_manifest = json.loads(plan_manifest_path.read_text(encoding="utf-8"))
    if candidate_variant not in str(plan_manifest.get("kind") or ""):
        raise ValueError("selection plan does not match the requested candidate variant")
    if str((plan_manifest.get("files") or {}).get("selections_sha256") or "") != _sha256_file(
        selections_path
    ):
        raise ValueError("selection plan hash differs from its manifest")

    supplied_samples = {path.resolve(): _sha256_file(path) for path in sample_paths}
    label_owners: dict[str, Path] = {}
    for label_path in label_paths:
        manifest_path = label_path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_path = _resolve_recorded_path(str(manifest.get("samples_path") or ""))
        if source_path not in supplied_samples:
            raise ValueError("label index refers to a sample source that was not supplied")
        if str(manifest.get("samples_sha256") or "") != supplied_samples[source_path]:
            raise ValueError("label sample hash differs from its manifest")
        if str((manifest.get("judge_config") or {}).get("baseline_id") or "") != judge_baseline_id:
            raise ValueError("label index was not produced by the current Judge")
        for row in _read_jsonl(label_path):
            sample_id = str(row.get("sample_id") or "")
            owner = label_owners.get(sample_id)
            if sample_id and owner is not None and owner != label_path:
                raise ValueError("sample occurs in multiple label indexes")
            if sample_id:
                label_owners[sample_id] = label_path


def _sample_index(sample_paths: Iterable[Path]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for path in sample_paths:
        for row in _read_jsonl(path):
            sample_id = str(row.get("sample_id") or "")
            target = row.get("target")
            candidate = row.get("candidate")
            if (
                not sample_id
                or sample_id in result
                or not isinstance(target, dict)
                or not isinstance(candidate, dict)
                or not str(target.get("case_id") or "")
                or str(target.get("relationship") or "") not in {"group", "private"}
                or not str(candidate.get("id") or "")
            ):
                raise ValueError("label sample source has an invalid identity")
            result[sample_id] = row
    return result


def _latest_successful_labels(label_paths: Iterable[Path]) -> dict[str, Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    for path in label_paths:
        for row in _read_jsonl(path):
            sample_id = str(row.get("sample_id") or "")
            if not sample_id:
                raise ValueError("label record has no sample id")
            previous = latest.get(sample_id)
            if (
                previous is None
                or row.get("status") == "ok"
                or previous.get("status") != "ok"
            ):
                latest[sample_id] = row
    return {sample_id: row for sample_id, row in latest.items() if row.get("status") == "ok"}


def _labels_by_pair(
    sample_paths: tuple[Path, ...],
    label_paths: tuple[Path, ...],
) -> dict[tuple[str, str], tuple[int, float]]:
    samples = _sample_index(sample_paths)
    result: dict[tuple[str, str], tuple[int, float]] = {}
    for sample_id, label in _latest_successful_labels(label_paths).items():
        sample = samples.get(sample_id)
        if sample is None:
            raise ValueError("successful label is absent from the supplied sample sources")
        target = sample["target"]
        candidate = sample["candidate"]
        relationship = str(target["relationship"])
        candidate_id = str(candidate["id"])
        value = label.get("label")
        probability = label.get("probability")
        if (
            value not in {0, 1}
            or not isinstance(probability, (int, float))
            or not 0.0 <= float(probability) <= 1.0
            or str(label.get("relationship") or "") != relationship
            or str(label.get("injected_candidate_id") or "") != candidate_id
        ):
            raise ValueError("successful label does not match its bound sample")
        key = (str(target["case_id"]), candidate_id)
        pair_value = (int(value), float(probability))
        if key in result and result[key] != pair_value:
            raise ValueError("the same target/candidate pair has conflicting labels")
        result[key] = pair_value
    return result


def _bootstrap_interval(differences: np.ndarray) -> list[float]:
    if not len(differences):
        return [0.0, 0.0]
    generator = np.random.default_rng(0)
    means = np.asarray([
        float(np.mean(generator.choice(differences, size=len(differences), replace=True)))
        for _ in range(BOOTSTRAP_ITERATIONS)
    ])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "targets": 0,
            "changed_selections": 0,
            "baseline_human_like_count": 0,
            "candidate_human_like_count": 0,
            "improved": 0,
            "regressed": 0,
            "unchanged": 0,
            "net_human_like_gain": 0,
        }
    baseline = np.asarray([int(row["baseline_label"]) for row in rows], dtype=float)
    candidate = np.asarray([int(row["candidate_label"]) for row in rows], dtype=float)
    baseline_probability = np.asarray(
        [float(row["baseline_probability"]) for row in rows], dtype=float
    )
    candidate_probability = np.asarray(
        [float(row["candidate_probability"]) for row in rows], dtype=float
    )
    differences = candidate - baseline
    improved = int(np.sum((baseline == 0) & (candidate == 1)))
    regressed = int(np.sum((baseline == 1) & (candidate == 0)))
    net = int(np.sum(differences))
    return {
        "targets": len(rows),
        "changed_selections": sum(bool(row["selection_changed"]) for row in rows),
        "baseline_human_like_count": int(np.sum(baseline)),
        "candidate_human_like_count": int(np.sum(candidate)),
        "baseline_human_like_rate": float(np.mean(baseline)),
        "candidate_human_like_rate": float(np.mean(candidate)),
        "improved": improved,
        "regressed": regressed,
        "unchanged": len(rows) - improved - regressed,
        "net_human_like_gain": net,
        "human_like_rate_delta": float(np.mean(differences)),
        "human_like_rate_delta_ci95": _bootstrap_interval(differences),
        "judge_detection_rate_delta": float(-np.mean(differences)),
        "baseline_mean_human_probability": float(np.mean(baseline_probability)),
        "candidate_mean_human_probability": float(np.mean(candidate_probability)),
        "mean_human_probability_delta": float(
            np.mean(candidate_probability - baseline_probability)
        ),
    }


def evaluate(
    selections_path: Path,
    sample_paths: tuple[Path, ...],
    label_paths: tuple[Path, ...],
    *,
    candidate_variant: str,
    c0_baseline_id: str,
    judge_baseline_id: str,
) -> dict[str, Any]:
    if candidate_variant not in SUPPORTED_VARIANTS:
        raise ValueError("candidate variant must be lr or xgboost")
    if not sample_paths or not label_paths:
        raise ValueError("sample and label sources are required")
    _validate_sources(
        selections_path,
        sample_paths,
        label_paths,
        candidate_variant=candidate_variant,
        judge_baseline_id=judge_baseline_id,
    )
    labels = _labels_by_pair(sample_paths, label_paths)
    selections = _read_jsonl(selections_path)
    complete: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    for selection in selections:
        target_id = str(selection.get("target_id") or "")
        relationship = str(selection.get("relationship") or "")
        production_candidate_id = str(selection.get("production_candidate_id") or "")
        candidate_id = str(selection.get(f"{candidate_variant}_candidate_id") or "")
        if (
            not target_id
            or target_id in seen_targets
            or relationship not in {"group", "private"}
            or not production_candidate_id
            or not candidate_id
        ):
            raise ValueError("selection plan has an invalid target or candidate identity")
        seen_targets.add(target_id)
        baseline = labels.get((target_id, production_candidate_id))
        candidate = labels.get((target_id, candidate_id))
        if baseline is None or candidate is None:
            excluded.append({
                "target_id": target_id,
                "relationship": relationship,
                "missing_production_label": baseline is None,
                "missing_candidate_label": candidate is None,
                "reason": "incomplete_verified_context_replay",
            })
            continue
        complete.append({
            "target_id": target_id,
            "relationship": relationship,
            "selection_changed": production_candidate_id != candidate_id,
            "baseline_label": baseline[0],
            "candidate_label": candidate[0],
            "baseline_probability": baseline[1],
            "candidate_probability": candidate[1],
        })
    if not selections:
        raise ValueError("selection plan is empty")

    metrics = {
        "overall": _metrics(complete),
        "group": _metrics([row for row in complete if row["relationship"] == "group"]),
        "private": _metrics([row for row in complete if row["relationship"] == "private"]),
    }
    coverage_complete = not excluded
    directional_gate_passed = (
        metrics["overall"]["net_human_like_gain"] > 0
        and metrics["group"]["net_human_like_gain"] >= 0
        and metrics["private"]["net_human_like_gain"] >= 0
    )
    eligible_for_quality_review = coverage_complete and directional_gate_passed
    if not coverage_complete:
        decision = "hold_incomplete_paired_coverage"
    elif directional_gate_passed:
        decision = "advance_to_explicit_quality_review"
    else:
        decision = "reject_no_net_improvement"
    improvements = [
        row["target_id"]
        for row in complete
        if row["baseline_label"] == 0 and row["candidate_label"] == 1
    ]
    regressions = [
        row["target_id"]
        for row in complete
        if row["baseline_label"] == 1 and row["candidate_label"] == 0
    ]
    return {
        "schema_version": 1,
        "kind": "few_shot_ranker_expanded_frozen_selection_paired_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_variant": candidate_variant,
        "status": "completed" if coverage_complete else "completed_with_exclusions",
        "current_defaults": {
            "c0_baseline_id": c0_baseline_id,
            "judge_baseline_id": judge_baseline_id,
        },
        "protocol": {
            "selection_is_frozen_before_labels": True,
            "paired_by_target_and_candidate": True,
            "successful_retry_overrides_transient_error": True,
            "higher_human_like_rate_is_better_for_c0": True,
            "missing_arm_excluded_and_blocks_promotion": True,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        },
        "sources": {
            "selections": {
                "path": _recorded_path(selections_path),
                "sha256": _sha256_file(selections_path),
            },
            "sample_indexes": [
                {"path": _recorded_path(path), "sha256": _sha256_file(path)}
                for path in sample_paths
            ],
            "label_indexes": [
                {"path": _recorded_path(path), "sha256": _sha256_file(path)}
                for path in label_paths
            ],
        },
        "coverage": {
            "planned_targets": len(selections),
            "complete_targets": len(complete),
            "excluded_targets": excluded,
            "excluded_relationship_counts": dict(sorted(Counter(
                row["relationship"] for row in excluded
            ).items())),
        },
        "metrics": metrics,
        "paired_outcomes": {
            "improved_target_ids": improvements,
            "regressed_target_ids": regressions,
        },
        "promotion_gate": {
            "coverage_complete": coverage_complete,
            "overall_net_improvement": metrics["overall"]["net_human_like_gain"] > 0,
            "group_not_worse": metrics["group"]["net_human_like_gain"] >= 0,
            "private_not_worse": metrics["private"]["net_human_like_gain"] >= 0,
            "directional_gate_passed": directional_gate_passed,
            "eligible_for_quality_review": eligible_for_quality_review,
            "entered_fixed_240": False,
            "adopted": False,
            "decision": decision,
        },
    }


def write_result(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIR_MODE)
    os.chmod(path.parent, PRIVATE_DIR_MODE)
    path.write_text(
        json.dumps(dict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, PRIVATE_MODE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--samples", type=Path, action="append")
    parser.add_argument("--labels", type=Path, action="append")
    parser.add_argument("--candidate-variant", choices=SUPPORTED_VARIANTS, default="xgboost")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    c0_id = current_c0_baseline_id()
    judge = load_current_judge(require_runnable=True)
    if str(judge.get("required_current_c0_baseline_id") or "") != c0_id:
        raise ValueError("current Judge is not bound to the current C0")
    judge_id = current_judge_baseline_id()
    result = evaluate(
        args.selections,
        tuple(args.samples) if args.samples else DEFAULT_SAMPLES,
        tuple(args.labels) if args.labels else DEFAULT_LABELS,
        candidate_variant=args.candidate_variant,
        c0_baseline_id=c0_id,
        judge_baseline_id=judge_id,
    )
    write_result(args.output, result)
    print(json.dumps({
        "candidate_variant": args.candidate_variant,
        "complete_targets": result["coverage"]["complete_targets"],
        "net_human_like_gain": result["metrics"]["overall"]["net_human_like_gain"],
        "decision": result["promotion_gate"]["decision"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
