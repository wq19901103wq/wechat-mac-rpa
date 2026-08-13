#!/usr/bin/env python3
"""用真实时间留出集评估 persona few-shot 召回。"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reply.few_shot import PersonaFewShotRetriever


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def evaluate(examples_path: Path, holdout_path: Path, limit: int = 5) -> tuple[dict, list[dict]]:
    retriever = PersonaFewShotRetriever(examples_path)
    cases = [json.loads(line) for line in holdout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts: Counter[str] = Counter()
    results = []
    for case in cases:
        query = "\n".join(case["context"])
        rows = retriever.retrieve(
            query=query,
            chat_name="",
            is_group=case["relationship"] == "group",
            limit=limit,
            relationship=case["relationship"],
            chat_id=case["chat_id"],
        )
        counts["cases"] += 1
        if rows:
            counts["retrieved_cases"] += 1
            counts["top1_same_chat"] += rows[0].get("chat_id") == case["chat_id"]
        counts["top5_same_chat"] += any(row.get("chat_id") == case["chat_id"] for row in rows)
        results.append({
            "holdout_id": case["id"],
            "query": case["context"],
            "actual_reply": case["expected_reply"],
            "expected": {"relationship": case["relationship"]},
            "retrieved": [
                {
                    "rank": rank,
                    "id": row["id"],
                    "same_chat": row.get("chat_id") == case["chat_id"],
                    "context": row.get("context", []),
                    "reply": row.get("reply", []),
                }
                for rank, row in enumerate(rows, 1)
            ],
        })
    total = counts["cases"]
    retrieved_total = counts["retrieved_cases"]
    summary = {
        "case_count": total,
        "coverage": _rate(retrieved_total, total),
        "top1": {
            "same_chat": _rate(counts["top1_same_chat"], total),
        },
        "top5": {
            "same_chat": _rate(counts["top5_same_chat"], total),
        },
    }
    return summary, results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    summary, results = evaluate(args.examples, args.holdout, args.limit)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "holdout_evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "holdout_retrieval_cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
