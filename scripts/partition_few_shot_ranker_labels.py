#!/usr/bin/env python3
"""Partition append-only few-shot labels by their source target.

This keeps raw worker indexes intact.  It is used when interrupted or
overlapping workers have written a sample to more than one raw index: the
evaluator requires that a sample belongs to exactly one index.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


PRIVATE_FILE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def partition(
    *,
    samples_path: Path,
    label_paths: list[Path],
    output_prefix: Path,
    shard_count: int,
) -> dict[str, int]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if any(not path.is_file() for path in label_paths):
        raise ValueError("each label index must exist")

    samples = _read_jsonl(samples_path)
    source_target_by_sample: dict[str, str] = {}
    for sample in samples:
        sample_id = str(sample.get("sample_id") or "")
        target_id = str((sample.get("target") or {}).get("case_id") or "")
        if not sample_id or not target_id or sample_id in source_target_by_sample:
            raise ValueError("source samples have an invalid or duplicate identity")
        source_target_by_sample[sample_id] = target_id

    target_ids = sorted(set(source_target_by_sample.values()))
    if len(target_ids) % shard_count:
        raise ValueError("target count is not divisible by shard_count")
    targets_per_shard = len(target_ids) // shard_count
    shard_by_target = {
        target_id: index // targets_per_shard + 1
        for index, target_id in enumerate(target_ids)
    }

    output_paths = [
        Path(f"{output_prefix}_shard{index}") / "labels.jsonl"
        for index in range(1, shard_count + 1)
    ]
    if any(path.parent.exists() for path in output_paths):
        raise ValueError("clean output directories already exist")
    for path in output_paths:
        path.parent.mkdir(mode=PRIVATE_DIR_MODE)
        os.chmod(path.parent, PRIVATE_DIR_MODE)

    handles = [path.open("x", encoding="utf-8") for path in output_paths]
    written = Counter()
    try:
        for label_path in label_paths:
            for row in _read_jsonl(label_path):
                sample_id = str(row.get("sample_id") or "")
                target_id = source_target_by_sample.get(sample_id)
                if target_id is None:
                    raise ValueError("label references an unknown sample")
                shard = shard_by_target[target_id]
                handles[shard - 1].write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                )
                written[shard] += 1
    finally:
        for handle, path in zip(handles, output_paths):
            handle.close()
            os.chmod(path, PRIVATE_FILE_MODE)
    return {str(index): written[index] for index in range(1, shard_count + 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--labels", type=Path, action="append", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()
    result = partition(
        samples_path=args.samples,
        label_paths=args.labels,
        output_prefix=args.output_prefix,
        shard_count=args.shards,
    )
    print(json.dumps({"written_records_by_shard": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
