#!/usr/bin/env python3
"""Find fresh development cases that trigger the group-pattern C0 contract."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_human_ai_judge import _export_payloads, rebuild_generation_inputs  # noqa: E402
from scripts.run_situation_c0_ab import (  # noqa: E402
    DEFAULT_DEVELOPMENT,
    DEFAULT_EXPOSED,
    DEFAULT_EXPORTS,
    DEFAULT_FIXED_TEST,
    SOURCE_CASE_IDS,
    _candidate,
    _payload_for_case,
)
from src.reply.generator import ReplyGenerator  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT
    / "data/reports/private/situation_c0_group_pattern_participation_candidates_v1.json"
)


def _ids(source: dict[str, Any]) -> set[str]:
    values = {str(value) for value in source.get("case_ids") or []}
    for row in source.get("cases") or []:
        for key in ("id", "case_id", "internal_id", "source_case_id"):
            if row.get(key):
                values.add(str(row[key]))
    return values


def _load_exclusions(exposed: Path, fixed_test: Path) -> tuple[set[str], list[str]]:
    paths = [exposed, fixed_test]
    paths.extend(sorted((ROOT / "data/reports/private").glob("situation_c0*/manifest.json")))
    excluded = set(SOURCE_CASE_IDS)
    loaded: list[str] = []
    for path in dict.fromkeys(path.resolve() for path in paths if path.exists()):
        source = json.loads(path.read_text(encoding="utf-8"))
        excluded.update(_ids(source))
        loaded.append(str(path.relative_to(ROOT)))
    return excluded, loaded


def build(args: argparse.Namespace) -> dict[str, Any]:
    development = json.loads(args.development.read_text(encoding="utf-8"))
    exports = _export_payloads(args.exports)
    excluded, exclusion_sources = _load_exclusions(args.exposed, args.fixed_test)
    source_cache: dict[Path, dict[str, Any]] = {}
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for row in development.get("cases") or []:
        if row.get("relationship") != "group":
            continue
        counts["group_pool"] += 1
        case_id = str(row["id"])
        if case_id in excluded:
            counts["excluded_case_id"] += 1
            continue
        payload = _payload_for_case(row, "development80", exports, source_cache)
        if payload is None:
            counts["missing_export"] += 1
            continue
        source = _candidate(row)
        try:
            to_reply, all_messages, _ = rebuild_generation_inputs(
                source, payload, compact_verified_turn=True
            )
        except Exception:
            counts["rebuild_error"] += 1
            continue
        contract = ReplyGenerator._group_pattern_participation_contract(
            to_reply, all_messages
        )
        if contract is None:
            counts["not_triggered"] += 1
            continue
        counts["triggered"] += 1
        counts[f"triggered_{contract['pattern_type']}"] += 1
        selected.append({
            "case_id": case_id,
            "chat_id": str(row["chat_id"]),
            "timestamp": int(row["timestamp"]),
            "pattern_type": contract["pattern_type"],
            "pattern_evidence": contract["pattern_evidence"],
            "participants": contract["participants"],
            "human_reply": list(row.get("reply") or []),
            "timeline": [
                {
                    "role": "self" if message.sender_type.value == "self" else "other",
                    "sender": str(message.sender or ""),
                    "text": str(message.text or ""),
                    "unread": any(message is unread for unread in to_reply),
                }
                for message in all_messages
            ],
        })

    selected.sort(key=lambda row: (row["pattern_type"], row["case_id"]))
    result = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "fresh development cases for the narrow group-pattern participation C0 candidate",
        "development_pool": str(args.development.relative_to(ROOT)),
        "selection_rule": "production timeline rebuild plus the exact candidate contract; prior situation-C0, exposed, source, and fixed-test ids excluded",
        "exclusion_sources": exclusion_sources,
        "counts": dict(sorted(counts.items())),
        "case_ids": [row["case_id"] for row in selected],
        "cases": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--exports", type=Path, default=DEFAULT_EXPORTS)
    parser.add_argument("--exposed", type=Path, default=DEFAULT_EXPOSED)
    parser.add_argument("--fixed-test", type=Path, default=DEFAULT_FIXED_TEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args)
    print(json.dumps({"output": str(args.output), "counts": result["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
