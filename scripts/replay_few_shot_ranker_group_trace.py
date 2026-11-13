#!/usr/bin/env python3
"""Replay one relationship's retrieval traces for existing ranker samples.

This makes an independent validation set usable by a relationship-only reranker
without regenerating replies or calling the Judge.  The replay is fail-closed:
every target's candidate ids and order must exactly match the historical sample
before any output is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_few_shot_ranker_samples import (
    DEFAULT_C0_CONFIG,
    DEFAULT_POOL,
    _candidate_view,
    _load_retrieval_settings,
    _retrievers_for_settings,
    _target_view,
    select_candidates,
)


DEFAULT_SAMPLES = ROOT / "data/reports/private/few_shot_ranker_samples_v3_validation160_for_test120/samples.jsonl"
DEFAULT_OUTPUT = ROOT / "data/reports/private/few_shot_ranker_samples_v3_validation160_group_trace_replay"
PRIVATE_MODE = 0o600


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recorded_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_private(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    path.write_text(content, encoding="utf-8")
    path.chmod(PRIVATE_MODE)


def replay(
    samples_path: Path,
    pool_path: Path,
    output_dir: Path,
    *,
    c0_config_path: Path = DEFAULT_C0_CONFIG,
    relationship: str = "group",
) -> dict[str, Any]:
    """Attach traces for one relationship, preserving the other rows exactly."""
    if relationship not in {"group", "private"}:
        raise ValueError("relationship must be group or private")
    if output_dir.exists():
        raise ValueError(f"output already exists: {output_dir}")
    samples = _read_jsonl(samples_path)
    if not samples:
        raise ValueError("source sample file is empty")
    targets: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    untouched_count = 0
    for index, sample in enumerate(samples):
        target = sample.get("target") or {}
        candidate = sample.get("candidate") or {}
        target_id = str(target.get("case_id") or "")
        sample_relationship = str(target.get("relationship") or "")
        if not target_id or not str(candidate.get("id") or ""):
            raise ValueError("source sample has incomplete target or candidate identity")
        if sample_relationship == relationship:
            targets[target_id].append((index, sample))
        elif sample_relationship in {"group", "private"}:
            untouched_count += 1
        else:
            raise ValueError(f"unsupported relationship in source samples: {sample_relationship!r}")
    if not targets:
        raise ValueError(f"source samples have no {relationship} target")

    settings, config_bytes = _load_retrieval_settings(c0_config_path)
    pool_bytes = pool_path.read_bytes()
    raw_pool = [json.loads(line) for line in pool_bytes.decode("utf-8").splitlines() if line.strip()]
    candidate_relationship = "group" if relationship == "group" else "acquaintance"
    candidate_pool = [
        _candidate_view(row)
        for row in raw_pool
        if row.get("relationship") == candidate_relationship
    ]
    if not candidate_pool:
        raise ValueError(f"few-shot pool has no {candidate_relationship} candidates")
    retriever = _retrievers_for_settings(pool_path, settings)[relationship]

    traces_by_sample_id: dict[str, dict[str, Any]] = {}
    mismatches: list[dict[str, Any]] = []
    for target_id, indexed_rows in sorted(targets.items()):
        indexed_rows.sort(key=lambda pair: int(pair[1].get("candidate_rank") or 0))
        ranks = [int(sample.get("candidate_rank") or 0) for _, sample in indexed_rows]
        if ranks != list(range(1, len(indexed_rows) + 1)):
            raise ValueError(f"{relationship} target {target_id} has non-contiguous candidate ranks")
        source_target = indexed_rows[0][1].get("target") or {}
        if any((sample.get("target") or {}) != source_target for _, sample in indexed_rows):
            raise ValueError(f"{relationship} target {target_id} has inconsistent target metadata")
        # Original human replies are intentionally absent from ranker samples.
        # Exact replay equality below prevents that absence from silently
        # changing the selected production top-k.
        target = _target_view({**source_target, "replies": []})
        selected, _ = select_candidates(
            target,
            candidate_pool,
            retriever=retriever,
            candidates_per_target=len(indexed_rows),
        )
        expected_ids = [str((sample.get("candidate") or {}).get("id") or "") for _, sample in indexed_rows]
        replayed_ids = [str(row["candidate"]["id"]) for row in selected]
        if replayed_ids != expected_ids:
            mismatches.append({
                "target_id": target_id,
                "expected_candidate_ids": expected_ids,
                "replayed_candidate_ids": replayed_ids,
            })
            continue
        for (_, sample), selected_row in zip(indexed_rows, selected):
            traces_by_sample_id[str(sample["sample_id"])] = selected_row["production_retrieval_features"]
    if mismatches:
        preview = ", ".join(row["target_id"] for row in mismatches[:5])
        raise ValueError(
            f"production retrieval replay mismatched {len(mismatches)} {relationship} targets ({preview}); no output written"
        )

    replayed_samples: list[dict[str, Any]] = []
    for sample in samples:
        copied = dict(sample)
        sample_id = str(copied["sample_id"])
        if sample_id in traces_by_sample_id:
            copied["production_retrieval_features"] = traces_by_sample_id[sample_id]
        replayed_samples.append(copied)
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in replayed_samples)
    manifest = {
        "schema_version": 1,
        "kind": "few_shot_ranker_production_trace_replay",
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "replayed_exact_candidate_order",
        "sources": {
            "samples": {"path": _recorded_path(samples_path), "sha256": _sha256_file(samples_path)},
            "few_shot_pool": {"path": _recorded_path(pool_path), "sha256": _sha256_file(pool_path)},
            "c0_config": {"path": _recorded_path(c0_config_path), "sha256": hashlib.sha256(config_bytes).hexdigest()},
        },
        "protocol": {
            "relationship_replayed": relationship,
            "other_relationship_rows": "copied unchanged; not used by the relationship-only learned ranker",
            "retrieval_flags": settings,
            "candidate_order_requirement": "exact candidate ids and rank order must match source samples",
            "c0_calls": 0,
            "judge_calls": 0,
            "candidate_llm_calls": 0,
        },
        "counts": {
            "source_samples": len(samples),
            f"{relationship}_targets_replayed": len(targets),
            f"{relationship}_samples_with_trace": len(traces_by_sample_id),
            f"{'private' if relationship == 'group' else 'group'}_samples_copied_unchanged": untouched_count,
        },
        "files": {"samples": "samples.jsonl", "samples_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()},
    }
    _write_private(output_dir / "samples.jsonl", content)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").chmod(PRIVATE_MODE)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--c0-config", type=Path, default=DEFAULT_C0_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--relationship", choices=("group", "private"), default="group")
    args = parser.parse_args()
    print(json.dumps(
        replay(
            args.samples,
            args.pool,
            args.output,
            c0_config_path=args.c0_config,
            relationship=args.relationship,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
