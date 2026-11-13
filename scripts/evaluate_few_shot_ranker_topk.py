#!/usr/bin/env python3
"""Evaluate frozen production and ranker top-k few-shot arms by target."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_few_shot_ranker_expanded import (  # noqa: E402
    _metrics,
    _read_jsonl,
    _recorded_path,
    _resolve_recorded_path,
    _sha256_file,
    write_result,
)
from src.badcase.judge_config import current_judge_baseline_id, load_current_judge  # noqa: E402
from src.reply.c0_config import current_c0_baseline_id  # noqa: E402


DEFAULT_SELECTIONS = (
    ROOT / "data/reports/private/few_shot_ranker_lr_validation160_top3_plan/selections.jsonl"
)
DEFAULT_SAMPLES = (
    ROOT / "data/reports/private/few_shot_ranker_lr_validation160_top3_label_samples/samples.jsonl",
)
DEFAULT_LABELS = tuple(
    ROOT / f"data/reports/private/few_shot_ranker_lr_validation160_top3_labels/shard{index}/labels.jsonl"
    for index in range(4)
)
DEFAULT_OUTPUT = (
    ROOT / "data/reports/private/few_shot_ranker_lr_validation160_top3_evaluation/result.json"
)
SUPPORTED_VARIANTS = ("lr", "xgboost")


def _validate_sources(
    selections_path: Path,
    sample_paths: tuple[Path, ...],
    label_paths: tuple[Path, ...],
    *,
    candidate_variant: str,
    selection_count: int,
    c0_baseline_id: str,
    judge_baseline_id: str,
) -> Mapping[str, Any]:
    plan_manifest = json.loads(
        (selections_path.parent / "manifest.json").read_text(encoding="utf-8")
    )
    expected_kind_fragment = f"{candidate_variant}_top{selection_count}_selection_plan"
    if expected_kind_fragment not in str(plan_manifest.get("kind") or ""):
        raise ValueError("selection plan does not match the requested top-k variant")
    if int((plan_manifest.get("protocol") or {}).get("selection_count") or 0) != selection_count:
        raise ValueError("selection plan count differs from the requested top-k count")
    if str((plan_manifest.get("files") or {}).get("selections_sha256") or "") != _sha256_file(
        selections_path
    ):
        raise ValueError("selection plan hash differs from its manifest")

    supplied_samples = {path.resolve(): _sha256_file(path) for path in sample_paths}
    label_owners: dict[str, Path] = {}
    for label_path in label_paths:
        manifest = json.loads((label_path.parent / "manifest.json").read_text(encoding="utf-8"))
        source_path = _resolve_recorded_path(str(manifest.get("samples_path") or ""))
        if source_path not in supplied_samples:
            raise ValueError("label index refers to a sample source that was not supplied")
        if str(manifest.get("samples_sha256") or "") != supplied_samples[source_path]:
            raise ValueError("label sample hash differs from its manifest")
        if str((manifest.get("judge_config") or {}).get("baseline_id") or "") != judge_baseline_id:
            raise ValueError("label index was not produced by the current Judge")
        if str((manifest.get("c0_config") or {}).get("baseline_id") or "") != c0_baseline_id:
            raise ValueError("label index was not produced by the current C0")
        if int((manifest.get("parameters") or {}).get("bound_few_shot_count") or 0) != selection_count:
            raise ValueError("label index was not produced with the requested top-k count")
        for row in _read_jsonl(label_path):
            sample_id = str(row.get("sample_id") or "")
            owner = label_owners.get(sample_id)
            if sample_id and owner is not None and owner != label_path:
                raise ValueError("sample occurs in multiple label indexes")
            if sample_id:
                label_owners[sample_id] = label_path
    return plan_manifest


def _sample_index(
    sample_paths: Iterable[Path],
    *,
    selection_count: int,
    candidate_variant: str,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    sample_ids: set[str] = set()
    for path in sample_paths:
        for row in _read_jsonl(path):
            sample_id = str(row.get("sample_id") or "")
            target = row.get("target")
            candidates = row.get("candidates")
            generation = row.get("generation")
            variant = str(row.get("evaluation_variant") or "")
            if (
                not sample_id
                or sample_id in sample_ids
                or not isinstance(target, dict)
                or not isinstance(candidates, list)
                or len(candidates) != selection_count
                or not all(isinstance(candidate, dict) for candidate in candidates)
                or not isinstance(generation, dict)
                or variant not in {"production", candidate_variant}
            ):
                raise ValueError("top-k label sample source has an invalid shape")
            target_id = str(target.get("case_id") or "")
            relationship = str(target.get("relationship") or "")
            candidate_ids = [str(candidate.get("id") or "") for candidate in candidates]
            if (
                not target_id
                or relationship not in {"group", "private"}
                or any(not candidate_id for candidate_id in candidate_ids)
                or len(candidate_ids) != len(set(candidate_ids))
                or generation.get("few_shot_count") != selection_count
                or generation.get("static_few_shot_count") != 0
                or generation.get("injected_candidate_ids") != candidate_ids
            ):
                raise ValueError("top-k label sample source has an invalid identity")
            key = (target_id, variant)
            if key in result:
                raise ValueError("top-k label samples contain a duplicate target arm")
            sample_ids.add(sample_id)
            result[key] = row
    return result


def _latest_successful_labels(
    label_paths: Iterable[Path],
    samples_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[int, float]]:
    latest: dict[str, Mapping[str, Any]] = {}
    for path in label_paths:
        for row in _read_jsonl(path):
            sample_id = str(row.get("sample_id") or "")
            if not sample_id:
                raise ValueError("label record has no sample id")
            previous = latest.get(sample_id)
            if previous is None or row.get("status") == "ok" or previous.get("status") != "ok":
                latest[sample_id] = row
    result: dict[str, tuple[int, float]] = {}
    for sample_id, label in latest.items():
        if label.get("status") != "ok":
            continue
        sample = samples_by_id.get(sample_id)
        if sample is None:
            raise ValueError("successful label is absent from the supplied sample sources")
        candidate_ids = [str(candidate["id"]) for candidate in sample["candidates"]]
        value = label.get("label")
        probability = label.get("probability")
        if (
            value not in {0, 1}
            or not isinstance(probability, (int, float))
            or not 0.0 <= float(probability) <= 1.0
            or str(label.get("relationship") or "") != str(sample["target"]["relationship"])
            or label.get("injected_candidate_ids") != candidate_ids
        ):
            raise ValueError("successful label does not match its bound top-k sample")
        result[sample_id] = (int(value), float(probability))
    return result


def evaluate(
    selections_path: Path,
    sample_paths: tuple[Path, ...],
    label_paths: tuple[Path, ...],
    *,
    candidate_variant: str,
    selection_count: int,
    c0_baseline_id: str,
    judge_baseline_id: str,
    evaluation_scope: str = "development",
    quality_review_path: Path | None = None,
) -> dict[str, Any]:
    if candidate_variant not in SUPPORTED_VARIANTS:
        raise ValueError("candidate variant must be lr or xgboost")
    if selection_count < 2 or not sample_paths or not label_paths:
        raise ValueError("top-k evaluation requires k >= 2 plus sample and label sources")
    if evaluation_scope not in {"development", "formal1000"}:
        raise ValueError("evaluation scope must be development or formal1000")
    plan_manifest = _validate_sources(
        selections_path,
        sample_paths,
        label_paths,
        candidate_variant=candidate_variant,
        selection_count=selection_count,
        c0_baseline_id=c0_baseline_id,
        judge_baseline_id=judge_baseline_id,
    )
    arms = _sample_index(
        sample_paths,
        selection_count=selection_count,
        candidate_variant=candidate_variant,
    )
    samples_by_id = {str(row["sample_id"]): row for row in arms.values()}
    labels = _latest_successful_labels(label_paths, samples_by_id)
    selections = _read_jsonl(selections_path)
    if not selections:
        raise ValueError("selection plan is empty")

    complete: list[dict[str, Any]] = []
    replay_excluded: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    same_key = f"same_top{selection_count}"
    for selection in selections:
        target_id = str(selection.get("target_id") or "")
        relationship = str(selection.get("relationship") or "")
        if (
            not target_id
            or target_id in seen_targets
            or relationship not in {"group", "private"}
            or int(selection.get("selection_count") or 0) != selection_count
            or not isinstance(selection.get(same_key), bool)
        ):
            raise ValueError("selection plan has an invalid target identity")
        seen_targets.add(target_id)
        production_ids = [str(value) for value in selection.get("production_candidate_ids") or ()]
        candidate_ids = [
            str(value) for value in selection.get(f"{candidate_variant}_candidate_ids") or ()
        ]
        if len(production_ids) != selection_count or len(candidate_ids) != selection_count:
            raise ValueError("selection plan has an incomplete top-k arm")
        baseline_arm = arms.get((target_id, "production"))
        candidate_arm = (
            baseline_arm if selection[same_key] else arms.get((target_id, candidate_variant))
        )
        if baseline_arm is None or candidate_arm is None:
            raise ValueError("selection plan is absent from its bound top-k sample source")
        baseline_bound_ids = [str(candidate["id"]) for candidate in baseline_arm["candidates"]]
        candidate_bound_ids = [str(candidate["id"]) for candidate in candidate_arm["candidates"]]
        if (
            str(baseline_arm["target"]["relationship"]) != relationship
            or str(candidate_arm["target"]["relationship"]) != relationship
            or baseline_bound_ids != production_ids
            or candidate_bound_ids != candidate_ids
            or bool(production_ids == candidate_ids) != bool(selection[same_key])
        ):
            raise ValueError("bound top-k arm differs from its frozen selection")
        baseline = labels.get(str(baseline_arm["sample_id"]))
        candidate = labels.get(str(candidate_arm["sample_id"]))
        if baseline is None or candidate is None:
            replay_excluded.append({
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
            "selection_changed": not selection[same_key],
            "baseline_label": baseline[0],
            "candidate_label": candidate[0],
            "baseline_probability": baseline[1],
            "candidate_probability": candidate[1],
        })

    structural_excluded = list((plan_manifest.get("counts") or {}).get("excluded_targets") or ())
    expanded_targets = int((plan_manifest.get("counts") or {}).get("expanded_evaluation_targets") or 0)
    if expanded_targets != len(selections) + len(structural_excluded):
        raise ValueError("selection plan target coverage does not reconcile")
    structural_ids = {str(row.get("target_id") or "") for row in structural_excluded}
    if not structural_ids or "" in structural_ids or structural_ids & seen_targets:
        if structural_excluded:
            raise ValueError("selection plan has invalid structural exclusions")

    metrics = {
        "overall": _metrics(complete),
        "group": _metrics([row for row in complete if row["relationship"] == "group"]),
        "private": _metrics([row for row in complete if row["relationship"] == "private"]),
    }
    paired_coverage_complete = not replay_excluded
    fixed_dev_coverage_complete = paired_coverage_complete and not structural_excluded
    relationship_counts = Counter(row["relationship"] for row in complete)
    formal_size_complete = (
        expanded_targets == 1000
        and len(complete) == 1000
        and relationship_counts == Counter({"group": 667, "private": 333})
        and fixed_dev_coverage_complete
    )
    directional_signal_positive = (
        metrics["overall"]["net_human_like_gain"] > 0
        and metrics["group"]["net_human_like_gain"] >= 0
        and metrics["private"]["net_human_like_gain"] >= 0
    )
    changed_target_ids = sorted(
        row["target_id"] for row in complete if row["selection_changed"]
    )
    quality_review: dict[str, Any] | None = None
    quality_review_complete = False
    new_hard_fact_errors: int | None = None
    if quality_review_path is not None:
        quality_review = json.loads(quality_review_path.read_text(encoding="utf-8"))
        reviewed_target_ids = quality_review.get("reviewed_target_ids")
        new_hard_fact_errors = quality_review.get("new_hard_fact_errors")
        quality_review_complete = (
            quality_review.get("status") == "completed"
            and isinstance(reviewed_target_ids, list)
            and sorted(str(value) for value in reviewed_target_ids) == changed_target_ids
            and type(new_hard_fact_errors) is int
            and new_hard_fact_errors >= 0
            and quality_review.get("selections_sha256") == _sha256_file(selections_path)
        )

    formal_promotion_eligible = (
        evaluation_scope == "formal1000"
        and formal_size_complete
        and metrics["overall"]["net_human_like_gain"] > 0
        and quality_review_complete
        and new_hard_fact_errors == 0
    )
    if not paired_coverage_complete:
        decision = "hold_incomplete_paired_replay"
    elif evaluation_scope == "formal1000" and not formal_size_complete:
        decision = "hold_incomplete_formal1000_coverage"
    elif metrics["overall"]["net_human_like_gain"] <= 0:
        decision = "reject_no_net_human_likeness_improvement"
    elif evaluation_scope == "formal1000" and not quality_review_complete:
        decision = "hold_pending_complete_quality_review"
    elif evaluation_scope == "formal1000" and new_hard_fact_errors != 0:
        decision = "reject_new_hard_fact_errors"
    elif formal_promotion_eligible:
        decision = "promote_update_default_c0"
    else:
        decision = "retain_positive_learning_signal_not_formally_promotable"
    return {
        "schema_version": 1,
        "kind": f"few_shot_ranker_top{selection_count}_frozen_selection_paired_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_variant": candidate_variant,
        "status": (
            "completed"
            if fixed_dev_coverage_complete
            else "completed_with_exclusions" if paired_coverage_complete else "incomplete"
        ),
        "current_defaults": {
            "c0_baseline_id": c0_baseline_id,
            "judge_baseline_id": judge_baseline_id,
            "few_shot_count": selection_count,
        },
        "protocol": {
            "ranker_training_uses_single_candidate_labels": True,
            "paired_replay_uses_full_topk_arms": True,
            "selection_is_frozen_before_labels": True,
            "candidate_order_is_frozen": True,
            "successful_retry_overrides_transient_error": True,
            "higher_human_like_rate_is_better_for_c0": True,
            "evaluation_scope": evaluation_scope,
            "quality_gate_deferred_for_current_learning_scope": evaluation_scope == "development",
            "missing_arm_blocks_conclusion": True,
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
            "quality_review": (
                {
                    "path": _recorded_path(quality_review_path),
                    "sha256": _sha256_file(quality_review_path),
                }
                if quality_review_path is not None else None
            ),
        },
        "coverage": {
            "fixed_dev_targets": expanded_targets,
            "planned_targets": len(selections),
            "complete_targets": len(complete),
            "structurally_excluded_targets": structural_excluded,
            "replay_excluded_targets": replay_excluded,
            "complete_relationship_counts": dict(sorted(relationship_counts.items())),
        },
        "metrics": metrics,
        "paired_outcomes": {
            "improved_target_ids": [
                row["target_id"] for row in complete
                if row["baseline_label"] == 0 and row["candidate_label"] == 1
            ],
            "regressed_target_ids": [
                row["target_id"] for row in complete
                if row["baseline_label"] == 1 and row["candidate_label"] == 0
            ],
        },
        "learning_gate": {
            "paired_coverage_complete": paired_coverage_complete,
            "fixed_dev_coverage_complete": fixed_dev_coverage_complete,
            "formal1000_coverage_complete": formal_size_complete,
            "overall_net_improvement": metrics["overall"]["net_human_like_gain"] > 0,
            "group_not_worse": metrics["group"]["net_human_like_gain"] >= 0,
            "private_not_worse": metrics["private"]["net_human_like_gain"] >= 0,
            "directional_signal_positive": directional_signal_positive,
            "quality_review_complete": quality_review_complete,
            "reviewed_changed_targets": len(changed_target_ids) if quality_review_complete else 0,
            "new_hard_fact_errors": new_hard_fact_errors,
            "formal_promotion_eligible": formal_promotion_eligible,
            "entered_fixed_240": False,
            "entered_formal1000": evaluation_scope == "formal1000",
            "adopted": False,
            "decision": decision,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--samples", type=Path, action="append")
    parser.add_argument("--labels", type=Path, action="append")
    parser.add_argument("--candidate-variant", choices=SUPPORTED_VARIANTS, default="lr")
    parser.add_argument("--selection-count", type=int, default=3)
    parser.add_argument(
        "--evaluation-scope", choices=("development", "formal1000"), default="development"
    )
    parser.add_argument("--quality-review", type=Path)
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
        selection_count=args.selection_count,
        c0_baseline_id=c0_id,
        judge_baseline_id=judge_id,
        evaluation_scope=args.evaluation_scope,
        quality_review_path=args.quality_review,
    )
    write_result(args.output, result)
    print(json.dumps({
        "complete_targets": result["coverage"]["complete_targets"],
        "net_human_like_gain": result["metrics"]["overall"]["net_human_like_gain"],
        "decision": result["learning_gate"]["decision"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
