#!/usr/bin/env python3
"""Reuse the baseline embeddings and encode only newly added retrieval rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_persona_few_shot_index import embedding_text  # noqa: E402
from src.memory.history_search import _BGEEncoder, _model_path, _try_import_encoder_deps  # noqa: E402


DEFAULT_EXAMPLES = ROOT / "data/reports/private/situation_few_shot_library_expanded_holdout_excluded_v1/persona_examples.jsonl"
DEFAULT_BASE_INDEX = ROOT / "data/reports/private/situation_few_shot_library_v1/persona_embeddings.npz"


def build(examples: Path, base_index: Path, output: Path, batch_size: int) -> dict[str, int | str]:
    rows = [json.loads(line) for line in examples.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples_sha256 = hashlib.sha256(examples.read_bytes()).hexdigest()
    with np.load(base_index, allow_pickle=False) as data:
        base_by_id = {
            str(row_id): embedding for row_id, embedding in zip(data["ids"], data["embeddings"])
        }
    missing = [row for row in rows if str(row["id"]) not in base_by_id]
    encoded_by_id: dict[str, np.ndarray] = {}
    if missing:
        backend = _try_import_encoder_deps()
        if backend is None or not _model_path().exists():
            raise RuntimeError("BGE encoder is unavailable")
        encoder = _BGEEncoder(_model_path(), backend)
        for start in range(0, len(missing), max(1, batch_size)):
            batch = missing[start:start + batch_size]
            vectors = encoder.encode([embedding_text(row) for row in batch])
            encoded_by_id.update({str(row["id"]): vector for row, vector in zip(batch, vectors)})
    embeddings = np.vstack([
        base_by_id.get(str(row["id"]), encoded_by_id.get(str(row["id"]))) for row in rows
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        ids=np.array([str(row["id"]) for row in rows]),
        embeddings=embeddings,
        examples_sha256=np.array(examples_sha256),
        index_version=np.array(2),
    )
    return {
        "examples": len(rows),
        "reused": len(rows) - len(missing),
        "encoded": len(missing),
        "dim": int(embeddings.shape[1]),
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--base-index", type=Path, default=DEFAULT_BASE_INDEX)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    output = args.output or args.examples.with_name("persona_embeddings.npz")
    print(json.dumps(build(args.examples, args.base_index, output, args.batch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
