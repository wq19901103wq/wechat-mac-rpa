#!/usr/bin/env python3
"""Compare two renderings of the exact same retrieved few-shot rows."""

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

from src.reply.few_shot import PersonaFewShotRetriever  # noqa: E402


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(percentile * len(ordered)) - 1)], 3)


def _timing(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean_ms": round(statistics.mean(values), 3) if values else None,
        "p50_ms": round(statistics.median(values), 3) if values else None,
        "p95_ms": _percentile(values, 0.95),
    }


def _view(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "chat_id": row.get("chat_id"),
            "timestamp": row.get("timestamp"),
            "context": row.get("context") or [],
            "context_messages": row.get("context_messages") or [],
            "reply": row.get("reply") or [],
            "reply_messages": row.get("reply_messages") or [],
        }
        for row in rows
    ]


def _same_event(row: dict[str, Any], case: dict[str, Any]) -> bool:
    """Exclude adjacent windows from the same historical conversation event."""
    if str(row.get("chat_id") or "") != str(case.get("chat_id") or ""):
        return False
    try:
        seconds_apart = abs(float(row.get("timestamp")) - float(case.get("timestamp")))
    except (TypeError, ValueError):
        return False
    return seconds_apart <= 3600


def _without_same_event(
    rows: list[dict[str, Any]], case: dict[str, Any], limit: int
) -> tuple[list[dict[str, Any]], list[str]]:
    excluded = [str(row["id"]) for row in rows if _same_event(row, case)]
    kept = [row for row in rows if not _same_event(row, case)][:limit]
    return kept, excluded


def _load_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["id"])] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-pool", type=Path, required=True)
    parser.add_argument("--new-pool", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--case-ids-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=40)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-render-chars", type=int, default=2500)
    args = parser.parse_args()

    development = json.loads(args.development.read_text(encoding="utf-8"))
    by_id = {str(row["id"]): row for row in development["cases"]}
    source = json.loads(args.case_ids_from.read_text(encoding="utf-8"))
    case_ids = [str(row["case_id"]) for row in source.get("cases", [])]
    selected = [by_id[case_id] for case_id in case_ids if case_id in by_id][: args.max_cases]

    old = PersonaFewShotRetriever(args.old_pool)
    old.retrieve("warmup", "", False, limit=args.limit)
    named_by_id = _load_by_id(args.new_pool)

    retrieval_timings: list[float] = []
    old_render_timings: list[float] = []
    named_render_timings: list[float] = []
    missing_named_ids: set[str] = set()
    cases: list[dict[str, Any]] = []
    for row in selected:
        query = "\n".join(str(value) for value in row.get("context") or [])
        kwargs = {
            "chat_name": "",
            "is_group": row.get("relationship") == "group",
            "limit": args.limit,
            "relationship": str(row.get("relationship") or "acquaintance"),
            "chat_id": str(row.get("chat_id") or ""),
            "exclude_ids": {str(row["id"])},
        }
        started = time.perf_counter()
        retrieval_limit = max(args.limit * 10, 30)
        old_rows = old.retrieve(query, **{**kwargs, "limit": retrieval_limit})
        retrieval_timings.append((time.perf_counter() - started) * 1000)
        old_clean, old_same_event_ids = _without_same_event(old_rows, row, args.limit)
        named_clean = []
        for old_row in old_clean:
            named_row = named_by_id.get(str(old_row["id"]))
            if named_row is None:
                missing_named_ids.add(str(old_row["id"]))
                named_row = old_row
            named_clean.append(named_row)
        started = time.perf_counter()
        old_prompt, old_rendered_ids = old.render(old_clean, max_chars=args.max_render_chars)
        old_render_timings.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        named_prompt, named_rendered_ids = old.render(named_clean, max_chars=args.max_render_chars)
        named_render_timings.append((time.perf_counter() - started) * 1000)
        cases.append(
            {
                "case_id": row["id"],
                "relationship": row.get("relationship"),
                "chat_id": row.get("chat_id"),
                "timestamp": row.get("timestamp"),
                "query_context": row.get("context") or [],
                "human_reply": row.get("reply") or [],
                "old": _view(old_clean),
                "named_context": _view(named_clean),
                "rendered": {
                    "old_ids": old_rendered_ids,
                    "named_context_ids": named_rendered_ids,
                    "old_prompt": old_prompt,
                    "named_context_prompt": named_prompt,
                },
                "same_event_excluded": old_same_event_ids,
            }
        )

    result = {
        "purpose": "development-only inspection of one feature: speaker names plus continuous context",
        "not_formal_evaluation": True,
        "only_variable": "rendered few-shot content: old short context without names vs the same selected IDs with up to five continuous messages and speaker names",
        "ranking": "old pool ranks once; both variants use identical selected IDs in identical order",
        "limit": args.limit,
        "max_render_chars": args.max_render_chars,
        "same_event_exclusion": "same chat and timestamp within one hour",
        "case_count": len(cases),
        "named_pool_coverage": {
            "old_pool_rows": len(old._rows),
            "named_pool_rows": len(named_by_id),
            "missing_selected_ids": sorted(missing_named_ids),
        },
        "timing": {
            "shared_retrieval": _timing(retrieval_timings),
            "old_render": _timing(old_render_timings),
            "named_context_render": _timing(named_render_timings),
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.chmod(0o600)
    print(json.dumps({"case_count": len(cases), "timing": result["timing"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
