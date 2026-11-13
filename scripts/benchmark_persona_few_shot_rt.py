#!/usr/bin/env python3
"""Controlled RT benchmark for the 11-example and 973-example persona pools."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_human_ai_judge import (  # noqa: E402
    DEFAULT_VOLCENGINE_CONFIG,
    VOLCENGINE_DSV4F_MODEL,
    VolcengineAnthropicClient,
    _export_payloads,
    _load_env,
    _run_c0,
    load_volcengine_credentials,
)
from scripts.run_clean_human_c0_baseline import (  # noqa: E402
    EXPORTS,
    PRIVATE_DIR,
    _candidate,
    _case_id,
    _has_media_tail,
    _load_verified_rows,
)
from src.reply.few_shot import PersonaFewShotRetriever  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "data/reports/private/persona_few_shot_rt_v1/results.json"
POOLS = {
    "small_11": PROJECT_ROOT / "data/few_shot_verified_humor/persona_examples.jsonl",
    "large_973": PROJECT_ROOT / "data/reports/private/style_same_chat_v1/persona_examples.jsonl",
}


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 3)


def _metrics(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50_ms": round(statistics.median(values), 3) if values else None,
        "p95_ms": _percentile(values, 0.95),
        "mean_ms": round(statistics.mean(values), 3) if values else None,
        "min_ms": round(min(values), 3) if values else None,
        "max_ms": round(max(values), 3) if values else None,
    }


def _local_retrieval(rows: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, path in POOLS.items():
        retriever = PersonaFewShotRetriever(path)
        timings: list[float] = []
        selected_counts: list[int] = []
        rendered_chars: list[int] = []
        # Warm the JSON/file cache. Dense retrieval remains enabled when an approved index exists.
        retriever.retrieve("warmup", "", False, limit=8)
        for repeat in range(repeats):
            for row in rows:
                query = "\n".join(str(value) for value in row.get("context_original") or [])
                started = time.perf_counter()
                selected = retriever.retrieve(
                    query=query,
                    chat_name=str(row.get("source_chat_id") or ""),
                    is_group=row.get("relationship") == "group",
                    limit=8,
                )
                rendered, ids = retriever.render(selected, max_chars=2500)
                timings.append((time.perf_counter() - started) * 1000)
                selected_counts.append(len(ids))
                rendered_chars.append(len(rendered))
        results[name] = {
            "pool_path": str(path),
            "pool_rows": sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()),
            "timing": _metrics(timings),
            "selected_count_p50": statistics.median(selected_counts),
            "rendered_chars_p50": statistics.median(rendered_chars),
        }
    return results


def _select_cases(max_cases: int) -> list[dict[str, Any]]:
    rows = [row for row in _load_verified_rows() if not _has_media_tail(row)]
    groups = sorted((row for row in rows if row.get("relationship") == "group"), key=_case_id)
    privates = sorted((row for row in rows if row.get("relationship") != "group"), key=_case_id)
    selected: list[dict[str, Any]] = []
    while len(selected) < max_cases and (groups or privates):
        if groups:
            selected.append(groups.pop(0))
        if len(selected) < max_cases and privates:
            selected.append(privates.pop(0))
    return selected


def _summarize_live(entries: dict[str, Any], condition: str) -> dict[str, Any]:
    rows = [value for value in entries.values() if value.get("condition") == condition and value.get("status") == "ok"]
    return {
        "end_to_end": _metrics([float(row["elapsed_ms"]) for row in rows]),
        "llm_reported": _metrics([float(row["llm_elapsed_ms"]) for row in rows]),
        "selected_count_p50": statistics.median([row["selected_count"] for row in rows]) if rows else None,
        "rendered_chars_p50": statistics.median([row["rendered_chars"] for row in rows]) if rows else None,
        "failures": sum(value.get("condition") == condition and value.get("status") != "ok" for value in entries.values()),
    }


def run(output: Path, max_cases: int, local_repeats: int, timeout_seconds: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"schema_version": 1, "entries": {}}
    entries = payload.setdefault("entries", {})
    cases = _select_cases(max_cases)
    payload["local_retrieval"] = _local_retrieval(cases, local_repeats)
    payload["design"] = {
        "cases": len(cases),
        "group_cases": sum(row.get("relationship") == "group" for row in cases),
        "private_cases": sum(row.get("relationship") != "group" for row in cases),
        "order": "paired per case; AB/BA alternates by case",
        "model": VOLCENGINE_DSV4F_MODEL,
        "few_shot_count": 8,
        "few_shot_max_chars": 2500,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _load_env()
    api_key, base_url = load_volcengine_credentials(DEFAULT_VOLCENGINE_CONFIG)
    client = VolcengineAnthropicClient(api_key, base_url, VOLCENGINE_DSV4F_MODEL)
    from src.memory import MemoryEngine

    memory = MemoryEngine(llm_client=client)
    exports = _export_payloads(EXPORTS)
    try:
        for case_index, row in enumerate(cases):
            order = ("small_11", "large_973") if case_index % 2 == 0 else ("large_973", "small_11")
            candidate = _candidate(row)
            source = exports.get(candidate.chat_id)
            if source is None:
                continue
            for condition in order:
                key = f"{_case_id(row)}:{condition}"
                if entries.get(key, {}).get("status") == "ok":
                    continue
                os.environ["ENABLE_PERSONA_FEW_SHOTS"] = "1"
                os.environ["PERSONA_FEW_SHOT_PATH"] = str(POOLS[condition])
                os.environ["PERSONA_FEW_SHOT_COUNT"] = "8"
                os.environ["PERSONA_FEW_SHOT_MAX_CHARS"] = "2500"
                started = time.perf_counter()
                try:
                    generated = _run_c0(candidate, source, client, memory, timeout_seconds)
                    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
                    llm_calls = generated.get("trace", {}).get("llm_calls") or []
                    entries[key] = {
                        "status": "ok" if not generated.get("generation_failed") else "generation_failed",
                        "case_id": _case_id(row),
                        "relationship": row.get("relationship"),
                        "condition": condition,
                        "elapsed_ms": elapsed_ms,
                        "llm_elapsed_ms": round(sum(float(call.get("elapsed") or 0) for call in llm_calls) * 1000, 3),
                        "selected_count": len(generated.get("few_shot_ids") or []),
                        "rendered_chars": len(generated.get("trace", {}).get("few_shot_content") or ""),
                        "reply_count": len(generated.get("replies") or []),
                    }
                except Exception as exc:
                    entries[key] = {
                        "status": "error",
                        "case_id": _case_id(row),
                        "relationship": row.get("relationship"),
                        "condition": condition,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                payload["summary"] = {name: _summarize_live(entries, name) for name in POOLS}
                output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        memory.shutdown()

    small = payload["summary"]["small_11"]["end_to_end"]
    large = payload["summary"]["large_973"]["end_to_end"]
    payload["delta"] = {
        "p50_ms": round(float(large["p50_ms"]) - float(small["p50_ms"]), 3) if small["p50_ms"] is not None and large["p50_ms"] is not None else None,
        "p95_ms": round(float(large["p95_ms"]) - float(small["p95_ms"]), 3) if small["p95_ms"] is not None and large["p95_ms"] is not None else None,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": payload["summary"], "delta": payload["delta"]}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--local-repeats", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    args = parser.parse_args()
    run(args.output, args.max_cases, args.local_repeats, args.timeout_seconds)


if __name__ == "__main__":
    main()
