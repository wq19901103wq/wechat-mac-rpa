#!/usr/bin/env python3
"""用真实时间留出集评估 persona few-shot 召回。"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reply.few_shot import PersonaFewShotRetriever, _query_response_mode, _row_response_mode


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def evaluate(examples_path: Path, holdout_path: Path, limit: int = 5) -> tuple[dict, list[dict]]:
    retriever = PersonaFewShotRetriever(examples_path)
    cases = [json.loads(line) for line in holdout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts: Counter[str] = Counter()
    results = []
    for case in cases:
        query = "\n".join(case["context"])
        desired_mode = case.get("desired_response_mode")
        if desired_mode in (None, "", "auto"):
            desired_mode = _query_response_mode(query)
        rows = retriever.retrieve(
            query=query,
            chat_name="",
            is_group=case["relationship"] == "group",
            limit=limit,
            relationship=case["relationship"],
            chat_id=case["chat_id"],
        )
        expected_topic = case["topic"]
        expected_intent = case["intent"]
        expected_mode = case["response_mode"]
        counts["cases"] += 1
        if rows:
            counts["retrieved_cases"] += 1
            counts["top1_topic"] += rows[0].get("topic") == expected_topic
            counts["top1_intent"] += rows[0].get("intent") == expected_intent
            counts["top1_joint"] += (
                rows[0].get("topic") == expected_topic and rows[0].get("intent") == expected_intent
            )
            counts["top1_same_chat"] += rows[0].get("chat_id") == case["chat_id"]
            counts["top1_response_mode"] += _row_response_mode(rows[0]) == expected_mode
        counts["top5_topic"] += any(row.get("topic") == expected_topic for row in rows)
        counts["top5_intent"] += any(row.get("intent") == expected_intent for row in rows)
        counts["top5_joint"] += any(
            row.get("topic") == expected_topic and row.get("intent") == expected_intent for row in rows
        )
        counts["top5_same_chat"] += any(row.get("chat_id") == case["chat_id"] for row in rows)
        counts["top5_response_mode"] += any(_row_response_mode(row) == expected_mode for row in rows)
        if desired_mode == "sincere":
            counts["sincere_cases"] += 1
            counts["sincere_top1"] += bool(rows and _row_response_mode(rows[0]) == "sincere")
            counts["sincere_top5"] += any(_row_response_mode(row) == "sincere" for row in rows)
            counts["sincere_playful_leak"] += any(_row_response_mode(row) == "playful" for row in rows)
            counts["sincere_top1_nonplayful"] += bool(rows and _row_response_mode(rows[0]) != "playful")
            counts["sincere_top1_topic_nonplayful"] += bool(
                rows and rows[0].get("topic") == expected_topic and _row_response_mode(rows[0]) != "playful"
            )
        results.append({
            "holdout_id": case["id"],
            "query": case["context"],
            "actual_reply": case["expected_reply"],
            "expected": {
                "relationship": case["relationship"],
                "intent": expected_intent,
                "topic": expected_topic,
                "response_mode": expected_mode,
            },
            "desired_response_mode": desired_mode,
            "retrieved": [
                {
                    "rank": rank,
                    "id": row["id"],
                    "same_chat": row.get("chat_id") == case["chat_id"],
                    "intent": row.get("intent"),
                    "topic": row.get("topic"),
                    "response_mode": _row_response_mode(row),
                    "context": row.get("context", []),
                    "reply": row.get("reply", []),
                }
                for rank, row in enumerate(rows, 1)
            ],
        })
    total = counts["cases"]
    retrieved_total = counts["retrieved_cases"]
    sincere_total = counts["sincere_cases"]
    summary = {
        "case_count": total,
        "coverage": _rate(retrieved_total, total),
        "top1": {
            "topic_match": _rate(counts["top1_topic"], total),
            "intent_match": _rate(counts["top1_intent"], total),
            "topic_intent_joint_match": _rate(counts["top1_joint"], total),
            "same_chat": _rate(counts["top1_same_chat"], total),
            "response_mode_match": _rate(counts["top1_response_mode"], total),
        },
        "top5": {
            "topic_match": _rate(counts["top5_topic"], total),
            "intent_match": _rate(counts["top5_intent"], total),
            "topic_intent_joint_match": _rate(counts["top5_joint"], total),
            "same_chat": _rate(counts["top5_same_chat"], total),
            "response_mode_match": _rate(counts["top5_response_mode"], total),
        },
        "top1_when_retrieved": {
            "topic_match": _rate(counts["top1_topic"], retrieved_total),
            "intent_match": _rate(counts["top1_intent"], retrieved_total),
            "topic_intent_joint_match": _rate(counts["top1_joint"], retrieved_total),
        },
        "sincerity": {
            "case_count": sincere_total,
            "top1_sincere": _rate(counts["sincere_top1"], sincere_total),
            "top5_has_sincere": _rate(counts["sincere_top5"], sincere_total),
            "top5_playful_leak": _rate(counts["sincere_playful_leak"], sincere_total),
            "top1_nonplayful": _rate(counts["sincere_top1_nonplayful"], sincere_total),
            "top1_topic_nonplayful": _rate(counts["sincere_top1_topic_nonplayful"], sincere_total),
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
