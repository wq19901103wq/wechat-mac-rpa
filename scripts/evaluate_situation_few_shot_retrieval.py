#!/usr/bin/env python3
"""Compare the previous lexical-heavy retrieval with situation-first retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reply.few_shot import (  # noqa: E402
    PersonaFewShotRetriever,
    _chat_id,
    _profile_compatibility,
    _row_situation_profile,
    _row_situation_tags,
    _situation_profile,
    _situation_tags,
    _terms,
)

DEFAULT_POOL = PROJECT_ROOT / "data/reports/private/style_same_chat_v1/persona_examples.jsonl"
DEFAULT_DEVELOPMENT = PROJECT_ROOT / "data/reports/private/c0_development_pool_v1/manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/reports/private/situation_few_shot_retrieval_v1/results.json"


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


def _old_retrieve(
    rows: list[dict[str, Any]],
    query: str,
    *,
    relationship: str,
    chat_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    query_terms = Counter(_terms(query))
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for row in rows:
        if relationship != row.get("relationship"):
            continue
        sample_text = " ".join(row["context"])
        sample_terms = Counter(_terms(sample_text))
        overlap = sum(min(count, sample_terms.get(term, 0)) for term, count in query_terms.items())
        lexical = overlap / max(1, sum(query_terms.values()))
        length = 1.0 / (1.0 + abs(len(query) - len(sample_text)) / 20.0)
        same_chat = bool(chat_id and row.get("chat_id") == chat_id)
        same_relationship = row.get("relationship") == relationship
        score = lexical * 8.0 + length + (1.5 if same_chat else 0.0) + (2.0 if same_relationship else 0.0)
        scored.append((-score, str(row["id"]), row))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [row for _, _, row in scored[: max(0, limit)]]


def _summary(cases: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    top1 = 0
    top3 = 0
    returned = 0
    same_chat_wrong_situation = 0
    top1_profile_compatible = 0
    for case in cases:
        selected = case[condition]
        tags = set(case["situation_tags"])
        selected_tags = [set(row["situation_tags"]) for row in selected]
        returned += len(selected)
        top1 += bool(selected_tags and tags & selected_tags[0])
        top3 += any(tags & row_tags for row_tags in selected_tags[:3])
        same_chat_wrong_situation += sum(
            bool(row["same_chat"] and not tags & set(row["situation_tags"]))
            for row in selected
        )
        top1_profile_compatible += bool(
            selected and selected[0]["profile_compatibility"] > 0
        )
    total = len(cases)
    return {
        "cases": total,
        "top1_same_situation": top1,
        "top1_same_situation_rate": round(top1 / total, 4) if total else None,
        "top3_has_same_situation": top3,
        "top3_has_same_situation_rate": round(top3 / total, 4) if total else None,
        "same_chat_wrong_situation_rows": same_chat_wrong_situation,
        "top1_profile_compatible": top1_profile_compatible,
        "top1_profile_compatible_rate": round(top1_profile_compatible / total, 4) if total else None,
        "returned_rows": returned,
    }


def _view(
    rows: list[dict[str, Any]],
    query_chat_id: str,
    query_profile: dict[str, str],
) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "context": row["context"],
            "reply": row["reply"],
            "situation_tags": sorted(_row_situation_tags(row)),
            "situation_profile": _row_situation_profile(row),
            "profile_compatibility": _profile_compatibility(query_profile, _row_situation_profile(row)),
            "same_chat": bool(query_chat_id and row.get("chat_id") == query_chat_id),
        }
        for row in rows
    ]


def run(
    pool: Path,
    development: Path,
    output: Path,
    max_cases: int,
    limit: int,
    case_ids_from: Path | None = None,
    skip_old: bool = False,
) -> None:
    all_library_rows = PersonaFewShotRetriever(pool)._load()
    fixed_case_ids = None
    if case_ids_from is not None:
        fixed_case_ids = [
            str(row["case_id"])
            for row in json.loads(case_ids_from.read_text(encoding="utf-8")).get("cases", [])
        ][:max_cases]
    library_rows = all_library_rows
    temporary_dir = tempfile.TemporaryDirectory(prefix="situation-retrieval-")
    retrieval_path = Path(temporary_dir.name) / "persona_examples.jsonl"
    retrieval_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in library_rows),
        encoding="utf-8",
    )
    source_index = pool.with_name("persona_embeddings.npz")
    if source_index.exists():
        selected_ids = {str(row["id"]) for row in library_rows}
        with np.load(source_index, allow_pickle=False) as data:
            source_ids = [str(value) for value in data["ids"]]
            keep = np.array([row_id in selected_ids for row_id in source_ids], dtype=bool)
            np.savez(
                retrieval_path.with_name("persona_embeddings.npz"),
                ids=data["ids"][keep],
                embeddings=data["embeddings"][keep],
                examples_sha256=np.array(hashlib.sha256(retrieval_path.read_bytes()).hexdigest()),
            )
    retriever = PersonaFewShotRetriever(retrieval_path)
    library_pairs = {
        (tuple(row.get("context") or []), tuple(row.get("reply") or []))
        for row in library_rows
    }
    development_rows = json.loads(development.read_text(encoding="utf-8"))["cases"]
    development_by_id = {str(row.get("id")): row for row in development_rows}
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in development_rows:
        tags = _situation_tags("\n".join(row.get("context") or []))
        if len(tags) != 1:
            continue
        if (tuple(row.get("context") or []), tuple(row.get("reply") or [])) in library_pairs:
            continue
        buckets[(str(row.get("relationship")), next(iter(tags)))].append(row)
    for rows in buckets.values():
        rows.sort(key=lambda row: str(row["id"]))

    selected_cases: list[dict[str, Any]] = []
    if fixed_case_ids is not None:
        selected_cases = [development_by_id[case_id] for case_id in fixed_case_ids if case_id in development_by_id]
    else:
        keys = sorted(buckets)
        while len(selected_cases) < max_cases and any(buckets.values()):
            moved = False
            for key in keys:
                if buckets[key] and len(selected_cases) < max_cases:
                    selected_cases.append(buckets[key].pop(0))
                    moved = True
            if not moved:
                break

    results: list[dict[str, Any]] = []
    old_timings: list[float] = []
    new_timings: list[float] = []
    retriever.retrieve("warmup", "", False, limit=limit)
    for row in selected_cases:
        query = "\n".join(str(value) for value in row.get("context") or [])
        query_profile = _situation_profile(query)
        relationship = str(row.get("relationship"))
        query_chat_id = str(row.get("chat_id") or "")
        old: list[dict[str, Any]] = []
        if not skip_old:
            started = time.perf_counter()
            old = _old_retrieve(
                library_rows,
                query,
                relationship=relationship,
                chat_id=query_chat_id,
                limit=limit,
            )
            old_timings.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        new = retriever.retrieve(
            query,
            "",
            is_group=relationship == "group",
            limit=limit,
            relationship=relationship,
            chat_id=query_chat_id,
            exclude_ids={
                str(candidate["id"])
                for candidate in library_rows
                if (
                    tuple(candidate.get("context") or []) == tuple(row.get("context") or [])
                    and tuple(candidate.get("reply") or []) == tuple(row.get("reply") or [])
                )
            },
        )
        new_timings.append((time.perf_counter() - started) * 1000)
        results.append({
            "case_id": row["id"],
            "relationship": relationship,
            "context": row["context"],
            "human_reply": row["reply"],
            "situation_tags": sorted(_situation_tags(query)),
            "situation_profile": query_profile,
            "old": _view(old, query_chat_id, query_profile),
            "new": _view(new, query_chat_id, query_profile),
        })

    packet = {
        "schema_version": 1,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "purpose": "development-only leave-one-out retrieval comparison; frozen test set not used",
        "pool": str(pool),
        "pool_rows": len(library_rows),
        "pool_rows_before_case_exclusion": len(all_library_rows),
        "fixed_case_ids_from": str(case_ids_from) if case_ids_from else None,
        "development_pool": str(development),
        "selection": "leave-one-out per case: the current exact human context/reply is excluded; other verified-human development history remains available",
        "case_distribution": {
            "total": len(results),
            "relationship": dict(sorted(Counter(row["relationship"] for row in results).items())),
            "situation": dict(sorted(Counter(
                row["situation_tags"][0] if row["situation_tags"] else "untagged"
                for row in results
            ).items())),
        },
        "old": None if skip_old else {**_summary(results, "old"), "retrieval_timing": _timing(old_timings)},
        "new": {**_summary(results, "new"), "retrieval_timing": _timing(new_timings)},
        "cases": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.chmod(0o600)
    temporary_dir.cleanup()
    print(json.dumps({key: packet[key] for key in ("case_distribution", "old", "new")}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cases", type=int, default=240)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--case-ids-from", type=Path)
    parser.add_argument("--skip-old", action="store_true")
    args = parser.parse_args()
    run(args.pool, args.development, args.output, args.max_cases, args.limit, args.case_ids_from, args.skip_old)


if __name__ == "__main__":
    main()
