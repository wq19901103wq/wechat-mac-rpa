#!/usr/bin/env python3
"""Bind a ranker's frozen top-k selections into label samples.

The selection plan is local-only: each target gets its production arm, and
only targets whose ranker arm differs get a second arm.  This script
materialises exactly those rows for the existing C0 + formal-Judge labeler.
It does not generate a reply or invoke the Judge.  The default top-1 format is
kept for compatibility; top-k arms bind their candidates in frozen order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTIONS = ROOT / "data/reports/private/few_shot_ranker_xgboost_expanded480_plan/selections.jsonl"
DEFAULT_SAMPLES = ROOT / "data/reports/private/few_shot_ranker_samples_v8_xgboost_expanded480/samples.jsonl"
DEFAULT_OUTPUT = ROOT / "data/reports/private/few_shot_ranker_xgboost_expanded480_label_samples"
PRIVATE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700
SUPPORTED_CANDIDATE_VARIANTS = ("lr", "xgboost")


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
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    content = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(content, encoding="utf-8")
    path.chmod(PRIVATE_MODE)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _bound_sample(
    source: Mapping[str, Any],
    *,
    variant: str,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    if variant not in {"production", *SUPPORTED_CANDIDATE_VARIANTS}:
        raise ValueError("unsupported selection variant")
    target = source.get("target")
    candidate = source.get("candidate")
    generation = source.get("generation")
    if not isinstance(target, dict) or not isinstance(candidate, dict) or not isinstance(generation, dict):
        raise ValueError("selected source sample is incomplete")
    expected_sample_id = str(selection[f"{variant}_sample_id"])
    expected_candidate_id = str(selection[f"{variant}_candidate_id"])
    if str(source.get("sample_id") or "") != expected_sample_id:
        raise ValueError("selection source sample id mismatch")
    if str(target.get("case_id") or "") != str(selection.get("target_id") or ""):
        raise ValueError("selection source target mismatch")
    if str(candidate.get("id") or "") != expected_candidate_id:
        raise ValueError("selection source candidate mismatch")
    if str(target.get("relationship") or "") != str(selection.get("relationship") or ""):
        raise ValueError("selection source relationship mismatch")
    if (
        generation.get("few_shot_count") != 1
        or generation.get("static_few_shot_count") != 0
        or str(generation.get("injected_candidate_id") or "") != expected_candidate_id
    ):
        raise ValueError("selection source is not a bound single-few-shot row")

    result = dict(source)
    # The source id identifies candidate/target; the prefix records which arm
    # caused its C0/Judge replay and prevents future label-index collisions.
    result["sample_id"] = f"{variant}:{expected_sample_id}"
    result["evaluation_variant"] = variant
    result["selection_source_sample_id"] = expected_sample_id
    return result


def _selection_count(selections: list[Mapping[str, Any]]) -> int:
    counts = {int(selection.get("selection_count") or 1) for selection in selections}
    if len(counts) != 1 or next(iter(counts)) < 1:
        raise ValueError("selection plan has inconsistent selection counts")
    return next(iter(counts))


def _topk_sample_id(variant: str, target_id: str, candidate_ids: list[str]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            [target_id, candidate_ids],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:20]
    return f"{variant}:top{len(candidate_ids)}:{digest}"


def _bound_topk_sample(
    sources: list[Mapping[str, Any]],
    *,
    variant: str,
    selection: Mapping[str, Any],
    selection_count: int,
) -> dict[str, Any]:
    if variant not in {"production", *SUPPORTED_CANDIDATE_VARIANTS}:
        raise ValueError("unsupported selection variant")
    expected_sample_ids = selection.get(f"{variant}_sample_ids")
    expected_candidate_ids = selection.get(f"{variant}_candidate_ids")
    expected_candidate_ranks = selection.get(f"{variant}_candidate_ranks")
    if not all(
        isinstance(values, list) and len(values) == selection_count
        for values in (expected_sample_ids, expected_candidate_ids, expected_candidate_ranks)
    ):
        raise ValueError("top-k selection arm is incomplete")
    sample_ids = [str(value) for value in expected_sample_ids]
    candidate_ids = [str(value) for value in expected_candidate_ids]
    candidate_ranks = [int(value) for value in expected_candidate_ranks]
    if (
        len(sources) != selection_count
        or any(not value for value in sample_ids + candidate_ids)
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        raise ValueError("top-k selection arm has invalid identities")

    target_id = str(selection.get("target_id") or "")
    relationship = str(selection.get("relationship") or "")
    target: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    for source, sample_id, candidate_id, candidate_rank in zip(
        sources, sample_ids, candidate_ids, candidate_ranks
    ):
        source_target = source.get("target")
        candidate = source.get("candidate")
        generation = source.get("generation")
        if (
            not isinstance(source_target, dict)
            or not isinstance(candidate, dict)
            or not isinstance(generation, dict)
            or str(source.get("sample_id") or "") != sample_id
            or str(source_target.get("case_id") or "") != target_id
            or str(source_target.get("relationship") or "") != relationship
            or str(candidate.get("id") or "") != candidate_id
            or int(source.get("candidate_rank") or 0) != candidate_rank
            or generation.get("few_shot_count") != 1
            or generation.get("static_few_shot_count") != 0
            or str(generation.get("injected_candidate_id") or "") != candidate_id
        ):
            raise ValueError("top-k selection source does not match its frozen arm")
        if target is None:
            target = dict(source_target)
        elif source_target != target:
            raise ValueError("top-k selection sources disagree on target payload")
        candidates.append(dict(candidate))
    if target is None:
        raise ValueError("top-k selection arm has no target")
    return {
        "sample_id": _topk_sample_id(variant, target_id, candidate_ids),
        "evaluation_variant": variant,
        "selection_source_sample_ids": sample_ids,
        "target": target,
        "candidates": candidates,
        "generation": {
            "few_shot_count": selection_count,
            "static_few_shot_count": 0,
            "injected_candidate_ids": candidate_ids,
        },
    }


def build(
    *,
    selections_path: Path = DEFAULT_SELECTIONS,
    samples_path: Path = DEFAULT_SAMPLES,
    output_dir: Path = DEFAULT_OUTPUT,
    candidate_variant: str = "xgboost",
    reuse_selections_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    if candidate_variant not in SUPPORTED_CANDIDATE_VARIANTS:
        raise ValueError("candidate variant must be lr or xgboost")
    if output_dir.exists():
        raise ValueError(f"output already exists: {output_dir}")
    source_rows = _read_jsonl(samples_path)
    source_by_id = {str(row.get("sample_id") or ""): row for row in source_rows}
    if len(source_by_id) != len(source_rows) or "" in source_by_id:
        raise ValueError("expanded source samples have duplicate or missing ids")
    selections = _read_jsonl(selections_path)
    if not selections:
        raise ValueError("selection plan is empty")
    selection_count = _selection_count(selections)
    if selection_count > 1 and reuse_selections_paths:
        raise ValueError("top-k arms cannot reuse single-candidate labels")
    reusable_pairs: set[tuple[str, str]] = set()
    for path in reuse_selections_paths:
        for reuse in _read_jsonl(path):
            target_id = str(reuse.get("target_id") or "")
            if not target_id:
                raise ValueError("reuse selection has an incomplete target identity")
            for key, value in reuse.items():
                if key.endswith("_candidate_id") and str(value or ""):
                    reusable_pairs.add((target_id, str(value)))
    selected_targets: set[str] = set()
    rows: list[dict[str, Any]] = []
    by_variant = Counter()
    by_relationship_variant = Counter()
    reused_by_variant = Counter()
    expected_rows = 0
    for selection in selections:
        target_id = str(selection.get("target_id") or "")
        relationship = str(selection.get("relationship") or "")
        if not target_id or target_id in selected_targets or relationship not in {"group", "private"}:
            raise ValueError("selection plan has invalid target identities")
        selected_targets.add(target_id)
        same_key = "same_top1" if selection_count == 1 else f"same_top{selection_count}"
        variants = ["production"]
        if selection.get(same_key) is False:
            variants.append(candidate_variant)
        elif selection.get(same_key) is not True:
            raise ValueError(f"selection plan has invalid {same_key} value")
        for variant in variants:
            expected_rows += 1
            if selection_count == 1:
                sample_id = str(selection.get(f"{variant}_sample_id") or "")
                candidate_id = str(selection.get(f"{variant}_candidate_id") or "")
                if (target_id, candidate_id) in reusable_pairs:
                    reused_by_variant[variant] += 1
                    continue
                source = source_by_id.get(sample_id)
                if source is None:
                    raise ValueError("selection references a missing source sample")
                rows.append(_bound_sample(source, variant=variant, selection=selection))
            else:
                sample_ids = selection.get(f"{variant}_sample_ids")
                if not isinstance(sample_ids, list):
                    raise ValueError("top-k selection references invalid source samples")
                sources = [source_by_id.get(str(sample_id)) for sample_id in sample_ids]
                if any(source is None for source in sources):
                    raise ValueError("top-k selection references a missing source sample")
                rows.append(_bound_topk_sample(
                    [source for source in sources if source is not None],
                    variant=variant,
                    selection=selection,
                    selection_count=selection_count,
                ))
            by_variant[variant] += 1
            by_relationship_variant[(relationship, variant)] += 1
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise ValueError("bound label samples have duplicate ids")
    if len(rows) + sum(reused_by_variant.values()) != expected_rows:
        raise AssertionError("bound label count does not match selection plan")

    output_dir.mkdir(parents=True, mode=PRIVATE_DIR_MODE)
    os.chmod(output_dir, PRIVATE_DIR_MODE)
    samples_sha256 = _write_jsonl(output_dir / "samples.jsonl", rows)
    counts = {
        "targets": len(selections),
        "samples": len(rows),
        "by_variant": dict(sorted(by_variant.items())),
        "by_relationship_variant": {
            f"{relationship}_{variant}": by_relationship_variant[(relationship, variant)]
            for relationship in ("group", "private")
            for variant in ("production", candidate_variant)
        },
        "by_relationship": dict(sorted(Counter(
            str(row["target"]["relationship"]) for row in rows
        ).items())),
        "reused_by_variant": dict(sorted(reused_by_variant.items())),
    }
    manifest = {
        "schema_version": 1,
        "kind": (
            f"few_shot_ranker_expanded_{candidate_variant}_label_samples"
            if selection_count == 1
            else f"few_shot_ranker_expanded_{candidate_variant}_top{selection_count}_label_samples"
        ),
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": f"bound_top{selection_count}_few_shot_samples_ready_for_c0_judge",
        "protocol": {
            "comparison": (
                f"production top-{selection_count} versus "
                f"{candidate_variant} top-{selection_count} by target"
            ),
            "single_few_shot_only": selection_count == 1,
            "bound_few_shot_count": selection_count,
            "static_few_shot_count": 0,
            f"reuse_same_top{selection_count}": True,
            "reuse_existing_candidate_labels": bool(reuse_selections_paths),
            "evaluation_relationships": ["group", "private"],
            "calls_in_this_step": {"c0": 0, "judge": 0},
        },
        "inputs": {
            "selections": {
                "path": _recorded_path(selections_path),
                "sha256": _sha256_file(selections_path),
            },
            "expanded_samples": {
                "path": _recorded_path(samples_path),
                "sha256": _sha256_file(samples_path),
            },
            "reuse_selections": [
                {"path": _recorded_path(path), "sha256": _sha256_file(path)}
                for path in reuse_selections_paths
            ],
        },
        "counts": counts,
        "files": {"samples": "samples.jsonl", "samples_sha256": samples_sha256},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.chmod(PRIVATE_MODE)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-variant", choices=SUPPORTED_CANDIDATE_VARIANTS, default="xgboost")
    parser.add_argument("--reuse-selections", type=Path, nargs="*", default=())
    args = parser.parse_args()
    result = build(
        selections_path=args.selections,
        samples_path=args.samples,
        output_dir=args.output,
        candidate_variant=args.candidate_variant,
        reuse_selections_paths=tuple(args.reuse_selections),
    )
    print(json.dumps(result["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
