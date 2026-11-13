#!/usr/bin/env python3
"""Add one local group-context × reply-style fact to frozen ranker samples.

This replay deliberately does not retrieve candidates or call C0, the Judge,
or an LLM.  It writes a new immutable sample set only after proving that every
pre-existing local fact and candidate identity still matches the frozen input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_few_shot_ranker_samples import (  # noqa: E402
    FEATURE_SCHEMA_VERSION,
    _candidate_view,
    _target_view,
    local_boolean_features,
)


FEATURE_NAME = "context_multiple_other_speakers__candidate_reply_has_first_person"
COMPONENT_FEATURES = (
    "context_multiple_other_speakers",
    "candidate_reply_has_first_person",
)
# The source was frozen before this independent local fact was added to the
# builder.  Keeping it absent is intentional: this replay must add exactly one
# tested feature, not silently bundle an earlier experimental direction.
_UNEMITTED_CURRENT_FEATURES = {"candidate_reply_overlap_current_ge2"}
PRIVATE_MODE = 0o600


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


def replay(samples_path: Path, output_dir: Path) -> dict[str, Any]:
    """Write frozen samples with only ``FEATURE_NAME`` appended."""
    if output_dir.exists():
        raise ValueError(f"output already exists: {output_dir}")
    samples = _read_jsonl(samples_path)
    if not samples:
        raise ValueError("source sample file is empty")

    replayed: list[dict[str, Any]] = []
    relationship_counts: Counter[str] = Counter()
    feature_true_counts: Counter[str] = Counter()
    for sample in samples:
        target_payload = sample.get("target")
        candidate_payload = sample.get("candidate")
        old_features = sample.get("local_features")
        trace = sample.get("production_retrieval_features")
        if not isinstance(target_payload, dict) or not isinstance(candidate_payload, dict):
            raise ValueError("source sample is missing target or candidate")
        if not isinstance(old_features, dict) or FEATURE_NAME in old_features:
            raise ValueError("source local feature schema is not the expected frozen schema")
        if trace is not None and not isinstance(trace, dict):
            raise ValueError("source production retrieval trace is not a mapping")
        if any(type(value) is not bool for value in old_features.values()):
            raise ValueError("source local features must all be booleans")
        if any(name not in old_features for name in COMPONENT_FEATURES):
            raise ValueError("source local feature schema lacks a cross component")

        target = _target_view({**target_payload, "replies": []})
        candidate = _candidate_view(candidate_payload)
        recomputed = local_boolean_features(target, candidate)
        unexpected_current = (
            set(recomputed) - set(old_features) - _UNEMITTED_CURRENT_FEATURES
        )
        if not set(old_features) <= set(recomputed) or unexpected_current:
            raise ValueError("recomputed local-feature schema differs from frozen source")
        changed_existing = [
            name for name in old_features
            if recomputed[name] != old_features[name]
        ]
        if changed_existing:
            raise ValueError(
                "recomputed local features differ from frozen source: "
                + ", ".join(sorted(changed_existing))
            )

        feature_value = all(old_features[name] for name in COMPONENT_FEATURES)
        copied = dict(sample)
        copied["local_features"] = {**old_features, FEATURE_NAME: feature_value}
        if (
            copied["sample_id"] != sample["sample_id"]
            or copied["candidate"] != sample["candidate"]
            or copied["candidate_rank"] != sample["candidate_rank"]
            or copied.get("production_retrieval_features")
            != sample.get("production_retrieval_features")
        ):
            raise AssertionError("frozen sample identity or production trace changed")
        replayed.append(copied)
        relationship = str(target_payload.get("relationship") or "")
        relationship_counts[relationship] += 1
        if feature_value:
            feature_true_counts[relationship] += 1

    content = "".join(
        json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n"
        for sample in replayed
    )
    manifest = {
        "schema_version": 1,
        "kind": "few_shot_ranker_local_feature_replay",
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "recomputed_from_frozen_samples",
        "sources": {
            "frozen_samples": _recorded_path(samples_path),
            "frozen_samples_sha256": _sha256_file(samples_path),
        },
        "protocol": {
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "sample_id_candidate_id_rank_and_production_trace": "copied exactly from frozen source",
            "allowed_local_feature_diff": {
                "added": [FEATURE_NAME],
                "removed": [],
                "changed_existing_values": [],
            },
            "retriever_calls": 0,
            "c0_calls": 0,
            "judge_calls": 0,
            "llm_calls": 0,
        },
        "counts": {
            "samples": len(replayed),
            "relationship": dict(sorted(relationship_counts.items())),
            f"{FEATURE_NAME}_true": dict(sorted(feature_true_counts.items())),
        },
        "files": {
            "samples": "samples.jsonl",
            "samples_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        },
    }
    output_dir.mkdir(parents=True)
    samples_output = output_dir / "samples.jsonl"
    samples_output.write_text(content, encoding="utf-8")
    samples_output.chmod(PRIVATE_MODE)
    manifest_output = output_dir / "manifest.json"
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_output.chmod(PRIVATE_MODE)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.samples, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
