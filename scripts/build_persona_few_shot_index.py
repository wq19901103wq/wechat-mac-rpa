#!/usr/bin/env python3
"""为 persona few-shot 上下文生成本地 BGE 语义索引。"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory.history_search import _BGEEncoder, _model_path, _try_import_encoder_deps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    output = args.output or args.examples.with_name("persona_embeddings.npz")
    rows = [json.loads(line) for line in args.examples.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples_sha256 = hashlib.sha256(args.examples.read_bytes()).hexdigest()
    backend = _try_import_encoder_deps()
    if backend is None or not _model_path().exists():
        raise SystemExit("BGE 编码器不可用")
    encoder = _BGEEncoder(_model_path(), backend)
    texts = ["\n".join(row["context"]) for row in rows]
    batches = []
    for start in range(0, len(texts), max(1, args.batch_size)):
        batches.append(encoder.encode(texts[start:start + args.batch_size]))
    embeddings = np.vstack(batches) if batches else np.zeros((0, 512), dtype=np.float32)
    np.savez_compressed(
        output,
        ids=np.array([row["id"] for row in rows]),
        embeddings=embeddings,
        examples_sha256=np.array(examples_sha256),
    )
    print(json.dumps({"examples": len(rows), "dim": embeddings.shape[1] if len(rows) else 0, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
