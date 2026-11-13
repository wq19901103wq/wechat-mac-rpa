#!/usr/bin/env python3
"""Audit candidate manifests for strict same-context C0 comparability."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_current_c0_historical_ab import _chat_message, _source  # noqa: E402
from scripts.build_adversarial_stage2_holdout import _anonymous_context, _fingerprint  # noqa: E402
from scripts.run_strict_historical_judge_acceptance import (  # noqa: E402
    _match_context,
    _strict_target_index,
)
from src.models.base import SenderType  # noqa: E402


def _normalize_candidate(case: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    if case.get("source_path"):
        return case
    if isinstance(case.get("human_reply"), dict) and case.get("context"):
        context_rows = list(case["context"])
        context = [str(row.get("content") or "").strip() for row in context_rows]
        reply = str(case["human_reply"].get("content") or "").strip()
        return {
            **case,
            "source_path": str(manifest_path.relative_to(ROOT)),
            "message_index": None,
            "chat_id": "specified_youzi_group_2",
            "chat_type": "group",
            "context": context,
            "context_messages": _anonymous_context(context_rows),
            "reply": reply,
            "content_fingerprints": sorted(
                {_fingerprint(reply), *(_fingerprint(value) for value in context)}
            ),
            "complex_large_group": True,
            "specified_group": "youzi_group_2",
        }
    return case


def _align_untyped_context(
    prefix: list[Any], context_rows: list[Any]
) -> tuple[list[Any], list[dict[str, str]], str]:
    wanted = [str(row).strip() for row in context_rows if str(row).strip()]
    if not wanted:
        return [], [], "manifest context is empty"
    positions: list[int] = []
    cursor = len(prefix) - 1
    for value in reversed(wanted):
        found = -1
        for index in range(cursor, max(-1, cursor - 180), -1):
            if prefix[index].text.strip() == value:
                found = index
                break
        if found < 0:
            return [], [], f"could not align untyped context item: {value[:30]}"
        positions.append(found)
        cursor = found - 1
    positions.reverse()
    messages = [prefix[index] for index in positions]
    anonymous = [
        {
            "speaker": "self" if message.sender_type is SenderType.SELF else "participant",
            "content": message.text,
        }
        for message in messages
    ]
    return messages, anonymous, "untyped context aligned to raw source"


def _audit_case(
    case: dict[str, Any], cache: dict[str, Any]
) -> tuple[str | None, dict[str, Any] | None]:
    try:
        source_key = str(case["source_path"])
        if source_key not in cache:
            rows, chat_name, chatroom_id = _source(case)
            converted = []
            prefix_counts = [0]
            for row in rows:
                if int(row.get("localType") or 0) != 10000:
                    converted.append(_chat_message(row, chat_name, chatroom_id, {}))
                prefix_counts.append(len(converted))
            cache[source_key] = rows, converted, prefix_counts
        rows, converted, prefix_counts = cache[source_key]
        target_index = _strict_target_index(rows, case)
        raw_prefix = converted[: prefix_counts[target_index]]
        context_messages = list(case.get("context_messages") or [])
        normalized = dict(case)
        if context_messages:
            event_context, alignment = _match_context(raw_prefix, context_messages)
        else:
            event_context, context_messages, alignment = _align_untyped_context(
                raw_prefix, list(case.get("context") or [])
            )
            normalized["context_messages"] = context_messages
        if not event_context:
            raise RuntimeError(alignment)
        last_self = max(
            (
                index
                for index, message in enumerate(event_context)
                if message.sender_type is SenderType.SELF
            ),
            default=-1,
        )
        if not any(
            message.sender_type is SenderType.OTHER
            for message in event_context[last_self + 1 :]
        ):
            raise RuntimeError("no independent incoming messages after the last self message")
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}", None
    return None, normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cache: dict[str, Any] = {}
    reports = []
    valid_cases = []
    for manifest_path in args.manifests:
        path = manifest_path if manifest_path.is_absolute() else ROOT / manifest_path
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases = list(payload.get("cases") or [])
        errors = Counter()
        valid = []
        invalid = []
        for case in cases:
            case = _normalize_candidate(case, path)
            error, normalized = _audit_case(case, cache)
            if error is None:
                assert normalized is not None
                valid.append(normalized)
                valid_cases.append(
                    {**normalized, "candidate_manifest": str(path.relative_to(ROOT))}
                )
            else:
                errors[error] += 1
                invalid.append({"internal_id": case.get("internal_id"), "error": error})
        reports.append(
            {
                "manifest": str(path.relative_to(ROOT)),
                "total": len(cases),
                "valid": len(valid),
                "invalid": len(invalid),
                "errors": dict(errors),
                "invalid_cases": invalid,
            }
        )

    result = {"reports": reports, "valid_cases": valid_cases}
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"reports": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
