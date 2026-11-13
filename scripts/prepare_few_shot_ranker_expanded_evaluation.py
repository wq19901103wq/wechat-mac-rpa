#!/usr/bin/env python3
"""Fit relation-specific rankers and select expanded evaluation top-k sets.

This is deliberately a local preparation step.  It fits only on the frozen,
labelled training targets, then scores the already-built expanded pool.  It
does not generate a C0 reply or invoke the Judge.  The following label pass
can therefore reuse the production top-1 and evaluate only targets whose
ranker selection actually differs.
"""

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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_few_shot_ranker_samples import _candidate_view, _target_view  # noqa: E402
from scripts.evaluate_few_shot_ranker import (  # noqa: E402
    LabeledSample,
    _add_local_feature_crosses,
    _add_production_trace_features,
    _by_target,
    _fit_train_dev_ranker,
    _ranking_feature_names,
    _recorded_path,
    _sha256_file,
    _training_rows,
    _ranker_model_description,
)


DEFAULT_TRAIN_SAMPLES = (
    ROOT / "data/reports/private/few_shot_ranker_samples_v7_question_turn_match_train160/samples.jsonl"
)
DEFAULT_TRAIN_LABELS = tuple(
    ROOT / f"data/reports/private/few_shot_ranker_labels_v3_train160_partitioned_shard{index}/labels.jsonl"
    for index in range(1, 5)
)
DEFAULT_EVALUATION_SAMPLES = (
    ROOT / "data/reports/private/few_shot_ranker_samples_v8_xgboost_expanded480/samples.jsonl"
)
DEFAULT_OUTPUT = ROOT / "data/reports/private/few_shot_ranker_xgboost_expanded480_plan"
PRIVATE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700
RELATIONSHIPS = ("group", "private")
QUESTION_TURN_CROSS = "context_latest_is_question__candidate_context_latest_is_question"
V8_ONLY_LOCAL_FEATURES = {
    "candidate_reply_has_first_person",
    "candidate_reply_overlap_current_ge2",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    content = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    path.write_text(content, encoding="utf-8")
    path.chmod(PRIVATE_MODE)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _question_turn_cross(sample: Mapping[str, Any]) -> bool:
    target_payload = sample.get("target")
    candidate_payload = sample.get("candidate")
    if not isinstance(target_payload, dict) or not isinstance(candidate_payload, dict):
        raise ValueError("expanded sample is missing a target or candidate")
    target = _target_view({**target_payload, "replies": []})
    candidate = _candidate_view(candidate_payload)
    return (
        target["latest_form"].endswith("question")
        and candidate["latest_form"].endswith("question")
    )


def _source_local_schema(samples: list[Mapping[str, Any]]) -> frozenset[str]:
    schemas = {
        frozenset(str(name) for name in (sample.get("local_features") or {}))
        for sample in samples
    }
    if len(schemas) != 1 or not schemas or not next(iter(schemas)):
        raise ValueError("sample source has local-feature schema drift")
    return next(iter(schemas))


def _expanded_rows(
    samples: list[Mapping[str, Any]],
    *,
    frozen_local_schema: frozenset[str],
) -> list[LabeledSample]:
    """Normalize v8 to the feature schema learned from v7 only."""
    v8_schema = _source_local_schema(samples)
    if v8_schema == frozen_local_schema:
        added_features: set[str] = set()
        missing_features: set[str] = set()
    else:
        added_features = set(v8_schema - frozen_local_schema)
        missing_features = set(frozen_local_schema - v8_schema)
    required_difference = (V8_ONLY_LOCAL_FEATURES, {QUESTION_TURN_CROSS})
    actual_difference = (added_features, missing_features)
    if actual_difference != (set(), set()) and actual_difference != required_difference:
        raise ValueError(
            "expanded local-feature schema is not the expected v7/v8 drift: "
            f"got added={sorted(actual_difference[0])}, missing={sorted(actual_difference[1])}"
        )

    rows: list[LabeledSample] = []
    for sample in samples:
        target = sample.get("target")
        candidate = sample.get("candidate")
        local_features = sample.get("local_features")
        trace = sample.get("production_retrieval_features")
        if not isinstance(target, dict) or not isinstance(candidate, dict):
            raise ValueError("expanded sample target or candidate is invalid")
        if not isinstance(local_features, dict) or not isinstance(trace, dict):
            raise ValueError("expanded sample features or production trace is invalid")
        if any(type(value) is not bool for value in local_features.values()):
            raise ValueError("expanded sample local features must be booleans")
        relationship = str(target.get("relationship") or "")
        if relationship not in RELATIONSHIPS:
            raise ValueError("expanded sample has an unsupported relationship")
        candidate_rank = int(sample.get("candidate_rank") or 0)
        if candidate_rank < 1:
            raise ValueError("expanded sample has an invalid candidate rank")
        features = {
            name: (
                _question_turn_cross(sample)
                if name == QUESTION_TURN_CROSS
                else bool(local_features[name])
            )
            for name in frozen_local_schema
        }
        features["candidate_is_production_top1"] = candidate_rank == 1
        features["candidate_is_production_rank_2_or_3"] = candidate_rank in {2, 3}
        features["candidate_is_production_rank_4_to_8"] = 4 <= candidate_rank <= 8
        rows.append(LabeledSample(
            sample_id=str(sample.get("sample_id") or ""),
            target_id=str(target.get("case_id") or ""),
            candidate_id=str(candidate.get("id") or ""),
            relationship=relationship,
            candidate_relationship=str(candidate.get("relationship") or ""),
            candidate_rank=candidate_rank,
            features=_add_local_feature_crosses(features),
            label=0,
            retrieval_features=trace,
        ))
    if any(not row.sample_id or not row.target_id or not row.candidate_id for row in rows):
        raise ValueError("expanded sample has an incomplete ranking identity")
    return _add_production_trace_features(rows)


def _assert_target_groups(groups: Mapping[str, list[LabeledSample]]) -> None:
    for target_id, rows in groups.items():
        if not rows:
            raise ValueError("empty target candidate group")
        if len({row.relationship for row in rows}) != 1:
            raise ValueError(f"target {target_id} has mixed relationships")
        ranks = [row.candidate_rank for row in rows]
        if len(ranks) != len(set(ranks)) or min(ranks) != 1:
            raise ValueError(f"target {target_id} has invalid production ranks")


def _selection_records(
    rows: list[LabeledSample],
    scores: Mapping[str, float],
    *,
    model_kind: str,
    selection_count: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selections: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for target_id, target_rows in sorted(_by_target(rows).items()):
        production_rows = sorted(
            target_rows,
            key=lambda row: (row.candidate_rank, row.candidate_id),
        )
        if len(production_rows) < selection_count:
            excluded.append({
                "target_id": target_id,
                "relationship": production_rows[0].relationship,
                "available_candidates": len(production_rows),
                "required_candidates": selection_count,
                "reason": "insufficient_safe_historical_candidates",
            })
            continue
        ranked = sorted(
            target_rows,
            key=lambda row: (-scores[row.sample_id], row.candidate_rank, row.candidate_id),
        )
        production = production_rows[:selection_count]
        selected = ranked[:selection_count]
        if selection_count == 1:
            selections.append({
                "target_id": target_id,
                "relationship": production[0].relationship,
                "production_sample_id": production[0].sample_id,
                "production_candidate_id": production[0].candidate_id,
                "production_candidate_rank": production[0].candidate_rank,
                f"{model_kind}_sample_id": selected[0].sample_id,
                f"{model_kind}_candidate_id": selected[0].candidate_id,
                f"{model_kind}_candidate_rank": selected[0].candidate_rank,
                f"{model_kind}_score": scores[selected[0].sample_id],
                "same_top1": production[0].candidate_id == selected[0].candidate_id,
            })
            continue
        production_ids = [row.candidate_id for row in production]
        selected_ids = [row.candidate_id for row in selected]
        selections.append({
            "target_id": target_id,
            "relationship": production[0].relationship,
            "selection_count": selection_count,
            "production_sample_ids": [row.sample_id for row in production],
            "production_candidate_ids": production_ids,
            "production_candidate_ranks": [row.candidate_rank for row in production],
            f"{model_kind}_sample_ids": [row.sample_id for row in selected],
            f"{model_kind}_candidate_ids": selected_ids,
            f"{model_kind}_candidate_ranks": [row.candidate_rank for row in selected],
            f"{model_kind}_scores": [scores[row.sample_id] for row in selected],
            f"same_top{selection_count}": production_ids == selected_ids,
        })
    return selections, excluded


def _apply_relationship_policy(
    selections: list[dict[str, Any]],
    scores: Mapping[str, float],
    *,
    model_kind: str,
    selection_count: int,
    ranker_enabled: bool,
) -> list[dict[str, Any]]:
    if ranker_enabled:
        return selections
    result: list[dict[str, Any]] = []
    for selection in selections:
        row = dict(selection)
        if selection_count == 1:
            row[f"{model_kind}_sample_id"] = row["production_sample_id"]
            row[f"{model_kind}_candidate_id"] = row["production_candidate_id"]
            row[f"{model_kind}_candidate_rank"] = row["production_candidate_rank"]
            row[f"{model_kind}_score"] = scores[str(row["production_sample_id"])]
            row["same_top1"] = True
        else:
            sample_ids = list(row["production_sample_ids"])
            row[f"{model_kind}_sample_ids"] = sample_ids
            row[f"{model_kind}_candidate_ids"] = list(row["production_candidate_ids"])
            row[f"{model_kind}_candidate_ranks"] = list(row["production_candidate_ranks"])
            row[f"{model_kind}_scores"] = [scores[str(sample_id)] for sample_id in sample_ids]
            row[f"same_top{selection_count}"] = True
        result.append(row)
    return result


def prepare(
    train_samples_path: Path,
    train_labels_paths: tuple[Path, ...],
    evaluation_samples_path: Path,
    output_dir: Path,
    *,
    model_kind: str = "xgboost",
    selection_count: int = 1,
    ranker_relationships: tuple[str, ...] = RELATIONSHIPS,
) -> dict[str, Any]:
    """Fit group/private rankers and write a no-LLM top-k evaluation plan."""
    if model_kind not in {"lr", "xgboost"}:
        raise ValueError("model kind must be lr or xgboost")
    if not 1 <= selection_count <= 8:
        raise ValueError("selection count must be between 1 and 8")
    if (
        not ranker_relationships
        or len(ranker_relationships) != len(set(ranker_relationships))
        or not set(ranker_relationships) <= set(RELATIONSHIPS)
    ):
        raise ValueError("ranker relationships must be a non-empty unique subset")
    if output_dir.exists():
        raise ValueError(f"output already exists: {output_dir}")
    training_rows, training_excluded, training_pairwise_only = _training_rows(
        train_samples_path,
        train_labels_paths,
        include_incomplete_pairwise_targets=False,
    )
    if training_pairwise_only:
        raise AssertionError("strict top-8 training unexpectedly included partial targets")
    training_rows = _add_production_trace_features(training_rows)
    train_groups = _by_target(training_rows)
    _assert_target_groups(train_groups)

    train_source = _read_jsonl(train_samples_path)
    frozen_local_schema = _source_local_schema(train_source)
    evaluation_rows = _expanded_rows(
        _read_jsonl(evaluation_samples_path),
        frozen_local_schema=frozen_local_schema,
    )
    evaluation_groups = _by_target(evaluation_rows)
    _assert_target_groups(evaluation_groups)
    train_target_ids = set(train_groups)
    evaluation_target_ids = set(evaluation_groups)
    if train_target_ids & evaluation_target_ids:
        raise ValueError("expanded evaluation targets overlap labelled training targets")

    selections: list[dict[str, Any]] = []
    excluded_selections: list[dict[str, Any]] = []
    model_summaries: dict[str, dict[str, Any]] = {}
    for relationship in RELATIONSHIPS:
        relationship_train = [row for row in training_rows if row.relationship == relationship]
        relationship_evaluation = [row for row in evaluation_rows if row.relationship == relationship]
        if not relationship_train or not relationship_evaluation:
            raise ValueError(f"{relationship} is absent from train or expanded evaluation")
        relation_train_groups = _by_target(relationship_train)
        relation_evaluation_groups = _by_target(relationship_evaluation)
        feature_names = _ranking_feature_names(relation_train_groups)
        if not feature_names:
            raise ValueError(f"{relationship} has no candidate-varying training features")
        train_schema = set(relationship_train[0].features)
        if any(set(row.features) != train_schema for row in relationship_train):
            raise ValueError(f"{relationship} training feature schema drift")
        if any(set(row.features) != train_schema for row in relationship_evaluation):
            raise ValueError(f"{relationship} expanded feature schema differs from frozen training schema")
        scores, _model, preference_count = _fit_train_dev_ranker(
            relationship_train,
            relationship_evaluation,
            feature_names,
            model_kind=model_kind,
        )
        relation_selections, relation_excluded = _selection_records(
            relationship_evaluation,
            scores,
            model_kind=model_kind,
            selection_count=selection_count,
        )
        relation_selections = _apply_relationship_policy(
            relation_selections,
            scores,
            model_kind=model_kind,
            selection_count=selection_count,
            ranker_enabled=relationship in ranker_relationships,
        )
        selections.extend(relation_selections)
        excluded_selections.extend(relation_excluded)
        same_key = "same_top1" if selection_count == 1 else f"same_top{selection_count}"
        model_summaries[relationship] = {
            "training_target_count": len(relation_train_groups),
            "training_candidate_count": len(relationship_train),
            "training_pairwise_preference_count": preference_count,
            "evaluation_target_count": len(relation_evaluation_groups),
            "evaluation_candidate_count": len(relationship_evaluation),
            "feature_count": len(feature_names),
            "feature_names": list(feature_names),
            "ranker_applied": relationship in ranker_relationships,
            f"top{selection_count}_changed_count": sum(
                not row[same_key] for row in relation_selections
            ),
            f"top{selection_count}_same_count": sum(
                row[same_key] for row in relation_selections
            ),
            "excluded_insufficient_candidate_targets": len(relation_excluded),
            "candidate_count_distribution": dict(
                sorted(Counter(len(rows) for rows in relation_evaluation_groups.values()).items())
            ),
        }

    selections.sort(key=lambda row: (row["relationship"], row["target_id"]))
    output_dir.mkdir(parents=True, mode=PRIVATE_DIR_MODE)
    os.chmod(output_dir, PRIVATE_DIR_MODE)
    selection_sha256 = _write_jsonl(output_dir / "selections.jsonl", selections)
    same_key = "same_top1" if selection_count == 1 else f"same_top{selection_count}"
    manifest = {
        "schema_version": 1,
        "kind": (
            f"few_shot_ranker_expanded_{model_kind}_selection_plan"
            if selection_count == 1
            else f"few_shot_ranker_expanded_{model_kind}_top{selection_count}_selection_plan"
        ),
        "privacy": "LOCAL PRIVATE: identifiers refer to real chat artifacts; never publish or commit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": f"local_top{selection_count}_plan_ready_for_c0_judge_pairwise_evaluation",
        "protocol": {
            "training": "strictly complete, labelled single-few-shot top-8 training targets only",
            "evaluation": "expanded, target-disjoint single-few-shot candidates; no labels were used",
            "relationship_models": f"separate group and private {model_kind} rankers",
            "selection": (
                f"production top-{selection_count} versus {model_kind} top-{selection_count}; "
                f"production rank then candidate id break {model_kind} score ties"
            ),
            "selection_count": selection_count,
            "ranker_applied_relationships": list(ranker_relationships),
            "model": _ranker_model_description(model_kind),
            "frozen_feature_source": _recorded_path(train_samples_path),
            "v8_schema_handling": {
                "dropped_evaluation_only_local_features": sorted(V8_ONLY_LOCAL_FEATURES),
                "locally_reconstructed_frozen_feature": QUESTION_TURN_CROSS,
            },
            "calls": {"candidate_llm": 0, "c0": 0, "judge": 0},
            "next_label_pass": {
                f"production_top{selection_count}_c0_judge_runs": len(selections),
                f"additional_{model_kind}_top{selection_count}_c0_judge_runs": sum(
                    not row[same_key] for row in selections
                ),
                f"reuse_for_same_top{selection_count}": True,
            },
        },
        "inputs": {
            "train_samples": {
                "path": _recorded_path(train_samples_path),
                "sha256": _sha256_file(train_samples_path),
            },
            "train_labels": [
                {"path": _recorded_path(path), "sha256": _sha256_file(path)}
                for path in train_labels_paths
            ],
            "expanded_evaluation_samples": {
                "path": _recorded_path(evaluation_samples_path),
                "sha256": _sha256_file(evaluation_samples_path),
            },
        },
        "counts": {
            "strict_training_targets": len(train_groups),
            "strict_training_candidates": len(training_rows),
            "training_excluded_incomplete_targets": len(training_excluded),
            "expanded_evaluation_targets": len(evaluation_groups),
            "planned_evaluation_targets": len(selections),
            "expanded_evaluation_candidates": len(evaluation_rows),
            f"top{selection_count}_changed": sum(not row[same_key] for row in selections),
            f"top{selection_count}_same": sum(row[same_key] for row in selections),
            "excluded_targets": excluded_selections,
            "by_relationship": model_summaries,
        },
        "files": {
            "selections": "selections.jsonl",
            "selections_sha256": selection_sha256,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.chmod(PRIVATE_MODE)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=Path, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--train-labels", type=Path, nargs="+", default=DEFAULT_TRAIN_LABELS)
    parser.add_argument("--evaluation-samples", type=Path, default=DEFAULT_EVALUATION_SAMPLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", choices=("lr", "xgboost"), default="xgboost")
    parser.add_argument("--selection-count", type=int, default=1)
    parser.add_argument(
        "--ranker-relationship",
        action="append",
        choices=RELATIONSHIPS,
        dest="ranker_relationships",
    )
    args = parser.parse_args()
    print(json.dumps(
        prepare(
            args.train_samples,
            tuple(args.train_labels),
            args.evaluation_samples,
            args.output,
            model_kind=args.model,
            selection_count=args.selection_count,
            ranker_relationships=tuple(args.ranker_relationships or RELATIONSHIPS),
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
