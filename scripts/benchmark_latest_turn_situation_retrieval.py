#!/usr/bin/env python3
"""Compare current retrieval with latest-turn situation/profile matching."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_human_ai_judge import _write_private  # noqa: E402
from scripts.run_situation_c0_ab import (  # noqa: E402
    DEFAULT_DEVELOPMENT,
    DEFAULT_EXPOSED,
    DEFAULT_FEW_SHOT,
    DEFAULT_FIXED_TEST,
    DEFAULT_NAMED_CONTEXT_FEW_SHOT,
    DEFAULT_PREVIOUS_AB,
    _excluded_case_ids,
    _select_cases,
)
from src.reply.few_shot import (  # noqa: E402
    PersonaFewShotRetriever,
    _row_situation_profile,
    _situation_profile,
)


DEFAULT_OUTPUT = ROOT / "data/reports/private/latest_turn_situation_retrieval_v1/results.json"


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


def _latest(texts: list[Any]) -> str:
    values = [str(value).strip() for value in texts if str(value).strip()]
    return values[-1] if values else ""


def run(args: argparse.Namespace) -> None:
    development = json.loads(args.development.read_text(encoding="utf-8"))
    exposed = json.loads(args.exposed.read_text(encoding="utf-8"))
    fixed_test = json.loads(args.fixed_test.read_text(encoding="utf-8"))
    previous = json.loads(args.previous_ab.read_text(encoding="utf-8"))
    excluded = _excluded_case_ids(exposed, [previous], fixed_test)
    if args.case_ids_from_results:
        prior = json.loads(args.case_ids_from_results.read_text(encoding="utf-8"))
        ordered_ids = [str(row["case_id"]) for row in prior.get("cases") or []]
        rows_by_id = {str(row["id"]): row for row in development["cases"]}
        rows = [rows_by_id[row_id] for row_id in ordered_ids if row_id in rows_by_id and row_id not in excluded]
        rows = rows[:args.max_cases]
    else:
        rows = _select_cases(development["cases"], excluded, args.max_cases, "latest-turn-situation")
    candidate_options = {
        "enable_precise_situation": True,
        "prioritize_latest_turn": args.candidate_mode in {
            "latest-turn-situation",
            "latest-turn-same-chat-sparse",
        },
        "prefer_same_chat_after_situation": args.candidate_mode == "latest-turn-same-chat-sparse",
        "avoid_concrete_examples_for_sparse_query": args.candidate_mode == "latest-turn-same-chat-sparse",
        "prefer_response_move_before_relevance": args.candidate_mode == "response-move-before-relevance",
        "prefer_multi_message_reply_coverage": args.candidate_mode == "multi-message-reply-coverage",
        "increase_semantic_topic_weight": args.candidate_mode == "semantic-topic-weight-8",
    }
    retrievers = {
        "baseline": PersonaFewShotRetriever(
            args.few_shot,
            render_path=args.named_context_few_shot,
            enable_precise_situation=True,
        ),
        "candidate": PersonaFewShotRetriever(
            args.few_shot,
            render_path=args.named_context_few_shot,
            **candidate_options,
        ),
    }
    for retriever in retrievers.values():
        retriever.retrieve("warmup", "", False, limit=args.candidate_few_shot_count)

    timings: dict[str, list[float]] = {name: [] for name in retrievers}
    cases = []
    for row in rows:
        context = row.get("context") or []
        query = "\n".join(str(value) for value in context)
        query_move = _situation_profile(_latest(context)).get("expected_response_move")
        conditions = {}
        shared_ranked = None
        shared_retrieval_ms = None
        if args.candidate_mode == "few-shot-count-5":
            started = time.perf_counter()
            shared_ranked = retrievers["candidate"].retrieve(
                query=query,
                chat_name="",
                chat_id=str(row.get("chat_id") or ""),
                is_group=row.get("relationship") == "group",
                relationship=str(row.get("relationship") or "acquaintance"),
                limit=args.candidate_few_shot_count,
                exclude_ids={str(row["id"])},
                use_situation=True,
            )
            shared_retrieval_ms = (time.perf_counter() - started) * 1000
        for name, retriever in retrievers.items():
            few_shot_count = args.few_shot_count if name == "baseline" else args.candidate_few_shot_count
            selected = list(shared_ranked[:few_shot_count]) if shared_ranked is not None else []
            if shared_retrieval_ms is not None:
                timings[name].append(shared_retrieval_ms)
            else:
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    selected = retriever.retrieve(
                        query=query,
                        chat_name="",
                        chat_id=str(row.get("chat_id") or ""),
                        is_group=row.get("relationship") == "group",
                        relationship=str(row.get("relationship") or "acquaintance"),
                        limit=few_shot_count,
                        exclude_ids={str(row["id"])},
                        use_situation=True,
                    )
                    timings[name].append((time.perf_counter() - started) * 1000)
            rendered, selected_ids = retriever.render(selected, max_chars=args.max_chars)
            examples = []
            for item in selected:
                latest_context = _latest(item.get("context") or [])
                latest_move = _situation_profile(latest_context).get("expected_response_move")
                examples.append({
                    "id": item.get("id"),
                    "chat_id": item.get("chat_id"),
                    "context": item.get("context") or [],
                    "reply": item.get("reply") or [],
                    "latest_response_move": latest_move,
                    "latest_move_match": bool(query_move and latest_move == query_move),
                })
            conditions[name] = {
                "requested_count": few_shot_count,
                "selected_ids": selected_ids,
                "rendered_chars": len(rendered),
                "latest_move_matches": sum(item["latest_move_match"] for item in examples),
                "examples": examples,
            }
        cases.append({
            "case_id": row["id"],
            "chat_id": row.get("chat_id"),
            "relationship": row.get("relationship"),
            "context": context,
            "human_reply": row.get("reply") or [],
            "latest_response_move": query_move,
            "changed": conditions["baseline"]["selected_ids"] != conditions["candidate"]["selected_ids"],
            "conditions": conditions,
        })

    result = {
        "schema_version": 1,
        "purpose": "development-only retrieval and RT check; not a formal evaluation result",
        "only_variable": (
            "the maximum number of situation-ranked few-shot examples entering the prompt: baseline 3, candidate 5"
            if args.candidate_mode == "few-shot-count-5"
            else
            "rank expected response-move match before generic lexical and semantic relevance"
            if args.candidate_mode == "response-move-before-relevance"
            else
            "for queries with at least 2 messages, add 1.0 retrieval point to examples with at least "
            "2 context messages and at least 2 human reply messages"
            if args.candidate_mode == "multi-message-reply-coverage"
            else
            "increase dense semantic topic similarity weight from 4.0 to 8.0"
            if args.candidate_mode == "semantic-topic-weight-8"
            else
            "derive situation and expected response move from the latest message, then prefer current-chat examples; "
            "sparse turns exclude examples carrying concrete topic facts"
            if args.candidate_mode == "latest-turn-same-chat-sparse"
            else "derive situation and expected response move from the latest message instead of the whole context"
        ),
        "fixed_240_excluded": True,
        "few_shot_pool": str(args.few_shot.relative_to(ROOT)),
        "few_shot_count": {"baseline": args.few_shot_count, "candidate": args.candidate_few_shot_count},
        "few_shot_max_chars": args.max_chars,
        "summary": {
            "cases": len(cases),
            "group_cases": sum(case["relationship"] == "group" for case in cases),
            "private_cases": sum(case["relationship"] != "group" for case in cases),
            "changed_cases": sum(case["changed"] for case in cases),
            "baseline_latest_move_matches": sum(case["conditions"]["baseline"]["latest_move_matches"] for case in cases),
            "candidate_latest_move_matches": sum(case["conditions"]["candidate"]["latest_move_matches"] for case in cases),
            "timing": {name: _timing(values) for name, values in timings.items()},
        },
        "cases": cases,
    }
    _write_private(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--exposed", type=Path, default=DEFAULT_EXPOSED)
    parser.add_argument("--fixed-test", type=Path, default=DEFAULT_FIXED_TEST)
    parser.add_argument("--previous-ab", type=Path, default=DEFAULT_PREVIOUS_AB)
    parser.add_argument("--few-shot", type=Path, default=DEFAULT_FEW_SHOT)
    parser.add_argument("--named-context-few-shot", type=Path, default=DEFAULT_NAMED_CONTEXT_FEW_SHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-ids-from-results", type=Path)
    parser.add_argument("--max-cases", type=int, default=240)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--few-shot-count", type=int, default=3)
    parser.add_argument("--candidate-few-shot-count", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=2500)
    parser.add_argument(
        "--candidate-mode",
        choices=(
            "latest-turn-situation",
            "latest-turn-same-chat-sparse",
            "few-shot-count-5",
            "response-move-before-relevance",
            "multi-message-reply-coverage",
            "semantic-topic-weight-8",
        ),
        default="latest-turn-situation",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
