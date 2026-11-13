#!/usr/bin/env python3
"""Select a balanced, trace-verified 80-case development view."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_human_ai_judge import _export_payloads, _verified_context_matches  # noqa: E402
from scripts.run_situation_c0_ab import (  # noqa: E402
    DEFAULT_DEVELOPMENT,
    DEFAULT_EXPOSED,
    DEFAULT_EXPORTS,
    DEFAULT_FIXED_TEST,
    SOURCE_CASE_IDS,
)
from src.reply.few_shot import _situation_profile  # noqa: E402
from src.utils.xml_utils import _extract_xml_text  # noqa: E402


DEFAULT_OUTPUT = ROOT / "data/reports/private/information_gap_response_move_cases_v1.json"


def _load_case_ids(path: Path) -> set[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    ids = {str(value) for value in payload.get("case_ids") or []}
    for row in payload.get("cases") or []:
        value = row.get("case_id") or row.get("internal_id") or row.get("id")
        if value:
            ids.add(str(value))
    return ids


def _excluded_ids(exposed: Path, fixed_test: Path, previous_glob: str) -> set[str]:
    result = set(SOURCE_CASE_IDS)
    result.update(_load_case_ids(exposed))
    result.update(_load_case_ids(fixed_test))
    for path in sorted(ROOT.glob(previous_glob)):
        result.update(_load_case_ids(path))
    return result


def _timestamp_seconds(value: Any) -> float:
    seconds = float(value or 0)
    return seconds / 1000 if seconds > 10_000_000_000 else seconds


def _target_contexts(payload: dict[str, Any]) -> dict[float, tuple[int, list[str]]]:
    result: dict[float, tuple[int, list[str]]] = {}
    last_texts: list[str] = []
    for index, message in enumerate(payload.get("messages") or []):
        if message.get("isSend"):
            result[_timestamp_seconds(message.get("createTime"))] = (index, list(last_texts))
        text = str(message.get("content") or "")
        if int(message.get("localType") or 0) != 1 and text.startswith("<"):
            text = _extract_xml_text(text) or text
        if text:
            last_texts = (last_texts + [text])[-4:]
    return result


def _retrieval_query(
    row: dict[str, Any],
    payload: dict[str, Any],
    target_contexts: dict[float, tuple[int, list[str]]],
) -> str:
    target_index, last_texts = target_contexts[_timestamp_seconds(row["timestamp"])]
    messages = payload.get("messages") or []
    expected = [str(value) for value in row.get("context") or []]
    if not expected:
        raise ValueError("empty verified context")
    if target_index < len(expected):
        raise ValueError("not enough messages before target")
    unread = messages[target_index - len(expected):target_index]
    if any(message.get("isSend") or int(message.get("localType") or 0) != 1 for message in unread):
        raise ValueError("verified turn is not the incoming text tail")
    replayed = [str(message.get("content") or "") for message in unread]
    if len(replayed) != len(expected) or not all(
        _verified_context_matches(left, right) for left, right in zip(expected, replayed)
    ):
        raise ValueError("verified context mismatch")
    return "\n".join(last_texts)


def _stable_key(case_id: str) -> str:
    return hashlib.sha256(f"information-gap-response-move-v1:{case_id}".encode()).hexdigest()


def run(args: argparse.Namespace) -> None:
    development = json.loads(args.development.read_text(encoding="utf-8"))
    exports = _export_payloads(args.exports)
    target_contexts = {chat_id: _target_contexts(payload) for chat_id, payload in exports.items()}
    excluded = _excluded_ids(args.exposed, args.fixed_test, args.previous_glob)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    failures: dict[str, int] = defaultdict(int)

    for row in development.get("cases") or []:
        case_id = str(row["id"])
        if case_id in excluded:
            continue
        payload = exports.get(str(row["chat_id"]))
        if payload is None:
            failures["missing_export"] += 1
            continue
        try:
            query = _retrieval_query(row, payload, target_contexts[str(row["chat_id"])])
        except (ValueError, TypeError, KeyError):
            failures["rebuild_failed"] += 1
            continue
        profile = _situation_profile(
            query,
            "",
            enable_clarification=False,
            enable_information_gap_response_move=True,
        )
        stratum = (
            "trigger"
            if profile.get("expected_response_move") == "verify_or_request_detail"
            else "regression"
        )
        relation = "group" if row.get("relationship") == "group" else "private"
        buckets[(stratum, relation)].append({
            "case_id": case_id,
            "stratum": stratum,
            "relationship": relation,
            "retrieval_query": query,
            "expected_response_move": profile.get("expected_response_move"),
        })

    selected: list[dict[str, Any]] = []
    available: dict[str, int] = {}
    for stratum in ("trigger", "regression"):
        for relation in ("group", "private"):
            key = (stratum, relation)
            rows = sorted(buckets[key], key=lambda row: _stable_key(row["case_id"]))
            available[f"{stratum}_{relation}"] = len(rows)
            if len(rows) < args.per_bucket:
                raise ValueError(f"not enough {stratum}/{relation} cases: {len(rows)} < {args.per_bucket}")
            selected.extend(rows[: args.per_bucket])

    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "trace-verified development A/B view; fixed 240 excluded",
        "feature": "information-gap-response-move",
        "selection_rule": {
            "per_bucket": args.per_bucket,
            "buckets": ["trigger_group", "trigger_private", "regression_group", "regression_private"],
            "trigger_expected_response_move": "verify_or_request_detail",
            "regression_expected_response_move": "not verify_or_request_detail",
            "stable_order_seed": "information-gap-response-move-v1",
        },
        "available": available,
        "failures": dict(failures),
        "excluded_count": len(excluded),
        "cases": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "selected": len(selected),
        "distribution": {
            key: sum(1 for row in selected if f"{row['stratum']}_{row['relationship']}" == key)
            for key in available
        },
        "available": available,
        "failures": dict(failures),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--exposed", type=Path, default=DEFAULT_EXPOSED)
    parser.add_argument("--fixed-test", type=Path, default=DEFAULT_FIXED_TEST)
    parser.add_argument("--exports", type=Path, default=DEFAULT_EXPORTS)
    parser.add_argument("--previous-glob", default="data/reports/private/situation_c0*/manifest.json")
    parser.add_argument("--per-bucket", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
