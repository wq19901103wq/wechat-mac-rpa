#!/usr/bin/env python3
"""Assemble the one final 240-case fixed C0 test manifest."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_adversarial_stage2_holdout import (  # noqa: E402
    _anonymous_context,
    _fingerprint,
    _near_duplicate,
)

NON_YOUZI = ROOT / "data/reports/private/judge_fixed_candidates_v2/private_manifest.json"
TARGET_RESOLUTION = ROOT / "data/reports/private/adversarial_stage2_target_resolution_candidates_v2/manifest.json"
HISTORICAL = ROOT / "data/reports/private/adversarial_stage2_holdout_v1/manifest.json"
NEW_YOUZI = ROOT / "data/reports/private/adversarial_stage2_youzi_quota_v1/discovered_candidates_v2/manifest.json"
AUDIT = ROOT / "data/reports/private/fixed_c0_holdout_final/candidate_audit.json"
OUTPUT = ROOT / "data/reports/private/fixed_c0_holdout_final/manifest.json"
SEED = "fixed-c0-final-240-20260822"
MEDIA_MARKERS = ("<img", "cdnmidimgurl", "[图片]", "[表情]", "[视频]", "<type>3</type>")
SOURCE_PRIORITY = {
    str(HIDDEN := ROOT / "data/reports/private/adversarial_hidden_holdout_v1/manifest.json"): 0,
    str(NON_YOUZI): 1,
    str(TARGET_RESOLUTION): 2,
    str(NEW_YOUZI): 0,
    str(HISTORICAL): 3,
}


def _load_cases(path: Path) -> list[dict[str, Any]]:
    return list(json.loads(path.read_text(encoding="utf-8")).get("cases") or [])


def _rank(case: dict[str, Any], namespace: str) -> str:
    return hashlib.sha256(f"{SEED}|{namespace}|{case['internal_id']}".encode()).hexdigest()


def _context(case: dict[str, Any]) -> list[Any]:
    return list(case.get("context_messages") or case.get("context") or [])


def _reply(case: dict[str, Any]) -> str:
    value = case.get("reply")
    if value is None and isinstance(case.get("human_reply"), dict):
        value = case["human_reply"].get("content")
    return str(value or "").strip()


def _normalize_new_youzi(case: dict[str, Any]) -> dict[str, Any]:
    context_rows = list(case.get("context") or [])
    context = [str(row.get("content") or "").strip() for row in context_rows]
    reply = _reply(case)
    return {
        "internal_id": str(case["internal_id"]),
        "source_path": str(NEW_YOUZI.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(NEW_YOUZI.read_bytes()).hexdigest(),
        "message_index": None,
        "timestamp": int(case["timestamp"]),
        "chat_id": "specified_youzi_group_2",
        "chat_type": "group",
        "event_cluster": str(case["event_cluster"]),
        "identity_evidence": "verified self sender in machine-filtered Youzi quota manifest",
        "context_complete_within_event": True,
        "context": context,
        "context_messages": _anonymous_context(context_rows),
        "reply": reply,
        "content_fingerprints": sorted({_fingerprint(reply), *(_fingerprint(x) for x in context)}),
        "complex_large_group": True,
        "source_message_count": 20011,
        "source_participant_count": 8,
        "specified_group": "youzi_group_2",
    }


def _has_media_dependency(case: dict[str, Any]) -> bool:
    text = "\n".join(
        str(row if isinstance(row, str) else row.get("content") or "")
        for row in _context(case)[-3:]
    ).lower()
    return any(marker.lower() in text for marker in MEDIA_MARKERS)


def _readable_text_only(case: dict[str, Any]) -> dict[str, Any] | None:
    """Drop unreadable media/XML rows without changing any surviving raw text."""
    context_messages = list(case.get("context_messages") or [])
    if not context_messages:
        return None
    readable = []
    for row in context_messages:
        content = str(row.get("content") or "").strip()
        lowered = content.lower()
        if (
            not content
            or content == "[非文本消息]"
            or lowered.startswith("<?xml")
            or lowered.startswith("<msg")
            or any(marker.lower() in lowered for marker in MEDIA_MARKERS)
        ):
            continue
        readable.append({**row, "content": content})
    if not readable:
        return None
    last_self = max(
        (index for index, row in enumerate(readable) if row.get("speaker") == "self"),
        default=-1,
    )
    if not any(row.get("speaker") != "self" for row in readable[last_self + 1 :]):
        return None
    return {
        **case,
        "context": [row["content"] for row in readable],
        "context_messages": readable,
        "context_normalization": "exact readable text rows only; media/XML rows omitted",
    }


def _select_distinct(
    candidates: list[dict[str, Any]],
    count: int,
    namespace: str,
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    replies = [_reply(row) for row in selected]
    events = {str(row.get("event_cluster")) for row in selected}
    picked: list[dict[str, Any]] = []
    for row in sorted(
        candidates,
        key=lambda case: (
            SOURCE_PRIORITY.get(str(ROOT / str(case.get("candidate_manifest") or "")), 9),
            _rank(case, namespace),
        ),
    ):
        reply = _reply(row)
        event = str(row.get("event_cluster"))
        if not reply or not _context(row) or _has_media_dependency(row) or event in events:
            continue
        if any(_near_duplicate(reply, old) for old in replies):
            continue
        picked.append({**row, "final_selection_source": namespace})
        replies.append(reply)
        events.add(event)
        if len(picked) == count:
            return picked
    raise RuntimeError(f"{namespace}: only found {len(picked)} of {count} valid distinct cases")


def _validate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(row.get("internal_id")) for row in cases]
    events = [str(row.get("event_cluster")) for row in cases]
    chat_types = Counter(str(row.get("chat_type")) for row in cases)
    common = sum(row.get("specified_group") == "correct_common_prosperity_with_luxixi" for row in cases)
    youzi = sum(str(row.get("specified_group") or "").startswith("youzi_group") for row in cases)
    complex_count = sum(bool(row.get("complex_large_group")) for row in cases)
    checks = {
        "total_240": len(cases) == 240,
        "group_200": chat_types["group"] == 200,
        "private_40": chat_types["private"] == 40,
        "complex_large_group_at_least_100": complex_count >= 100,
        "common_prosperity_20": common == 20,
        "youzi_20": youzi == 20,
        "unique_case_ids": len(set(ids)) == len(ids),
        "unique_events": len(set(events)) == len(events),
        "all_contexts_nonempty": all(_context(row) for row in cases),
        "all_human_replies_nonempty": all(_reply(row) for row in cases),
        "no_media_dependency": not any(_has_media_dependency(row) for row in cases),
    }
    if not all(checks.values()):
        raise RuntimeError(f"final fixed manifest validation failed: {checks}")
    return {
        "total": len(cases),
        "group": chat_types["group"],
        "private": chat_types["private"],
        "complex_large_group": complex_count,
        "common_prosperity": common,
        "youzi": youzi,
        "checks": checks,
    }


def main() -> None:
    audited = [
        normalized
        for row in json.loads(AUDIT.read_text(encoding="utf-8")).get("valid_cases") or []
        if (normalized := _readable_text_only(row)) is not None
    ]

    selected: list[dict[str, Any]] = []
    selected.extend(
        _select_distinct(
            [row for row in audited if row.get("specified_group") == "correct_common_prosperity_with_luxixi"],
            20,
            "common_prosperity",
            selected,
        )
    )
    selected.extend(
        _select_distinct(
            [row for row in audited if str(row.get("specified_group") or "").startswith("youzi_group")],
            20,
            "youzi",
            selected,
        )
    )
    selected.extend(
        _select_distinct(
            [row for row in audited if row.get("chat_type") == "private"],
            40,
            "private",
            selected,
        )
    )
    selected.extend(
        _select_distinct(
            [
                row for row in audited
                if row.get("chat_type") == "group"
                and row.get("complex_large_group")
                and not row.get("specified_group")
            ],
            60,
            "complex_group",
            selected,
        )
    )
    selected.extend(
        _select_distinct(
            [
                row for row in audited
                if row.get("chat_type") == "group"
                and not row.get("specified_group")
            ],
            100,
            "remaining_group",
            selected,
        )
    )
    coverage = _validate(selected)
    payload = {
        "schema_version": 1,
        "status": "sealed_pending_full_generation_and_judge",
        "sealed_at": datetime.now().astimezone().isoformat(),
        "seed": SEED,
        "selection_policy": {
            "same_single_fixed_test": True,
            "one_time_rebalance": True,
            "forced_reply_required": True,
            "fixed_judge": "Codex Judge v5",
            "fixed_test_not_used_for_tuning": True,
            "media_dependent_cases_excluded": True,
        },
        "source_manifests": [
            str(NON_YOUZI.relative_to(ROOT)),
            str(TARGET_RESOLUTION.relative_to(ROOT)),
            str(HISTORICAL.relative_to(ROOT)),
            str(NEW_YOUZI.relative_to(ROOT)),
        ],
        "coverage": coverage,
        "cases": selected,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps(coverage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
