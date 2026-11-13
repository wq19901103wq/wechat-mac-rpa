#!/usr/bin/env python3
"""Audit and seal the Stage 2 Judge/Bot holdout without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable

from scripts import build_adversarial_hidden_holdout as stage1


ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_COMMON_SOURCE = (
    ROOT
    / "data/private_benchmarks/judge_bot_adversarial_sources/20260818_account20b2"
    / "correct_common_prosperity_pre_cutoff_merged.json"
)
COMMON_SOURCE = _DEFAULT_COMMON_SOURCE
COMMON_ADDITIONAL_SOURCES = (
    ROOT
    / "data/private_benchmarks/judge_bot_adversarial_sources/20260822_account20b2"
    / "correct_common_prosperity_20260217_20260303.json",
)
YOUZI_MANIFEST = ROOT / "data/reports/private/adversarial_stage2_youzi_quota_v1/manifest.json"
STAGE1_MANIFEST = ROOT / "data/reports/private/adversarial_hidden_holdout_v1/manifest.json"
PRIVATE_DIR = ROOT / "data/reports/private/adversarial_stage2_holdout_v1"
MANIFEST_PATH = PRIVATE_DIR / "manifest.json"
PUBLIC_SUMMARY_PATH = PRIVATE_DIR / "public_summary.json"
TARGET_RESOLUTION_DIR = ROOT / "data/reports/private/adversarial_stage2_target_resolution_candidates_v1"
TARGET_RESOLUTION_MANIFEST_PATH = TARGET_RESOLUTION_DIR / "manifest.json"

SELF_SENDER = "wxid_qxlscnesk92m21"
COMMON_TALKER = "18406812120@chatroom"
TARGET_TOTAL = 240
TARGET_GROUP = 160
TARGET_PRIVATE = 80
TARGET_COMPLEX = 100
TARGET_DISTINCT_CHATS = 80
TARGET_COMMON = 20
TARGET_YOUZI = 20
ORDINARY_CHAT_CAP = 3

EXPOSURE_ROOTS = stage1.EXPOSURE_ROOTS
IGNORED_EXPOSURE_PATHS = {
    STAGE1_MANIFEST.resolve(),
    (STAGE1_MANIFEST.parent / "public_summary.json").resolve(),
    YOUZI_MANIFEST.resolve(),
    (YOUZI_MANIFEST.parent / "public_summary.json").resolve(),
    MANIFEST_PATH.resolve(),
    PUBLIC_SUMMARY_PATH.resolve(),
    TARGET_RESOLUTION_MANIFEST_PATH.resolve(),
    (TARGET_RESOLUTION_DIR / "strata_labels.json").resolve(),
    (TARGET_RESOLUTION_DIR / "strata_curation_audit.json").resolve(),
}


def _is_target_resolution_work_file(path: Path) -> bool:
    return any(
        part.startswith("adversarial_stage2_target_resolution_candidates_")
        for part in path.resolve().parts
    )


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text, flags=re.UNICODE).lower()


def _fingerprint(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def collect_exposed_fingerprints() -> tuple[set[str], int]:
    fingerprints: set[str] = set()
    files_scanned = 0
    for base in EXPOSURE_ROOTS:
        if not base.exists():
            continue
        paths = [base] if base.is_file() else base.rglob("*")
        for path in paths:
            if (
                not path.is_file()
                or path.resolve() in IGNORED_EXPOSURE_PATHS
                or _is_target_resolution_work_file(path)
            ):
                continue
            if path.suffix.lower() not in {".json", ".jsonl", ".md", ".txt", ".log"}:
                continue
            try:
                raw = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            files_scanned += 1
            values: Iterable[str]
            if path.suffix.lower() == ".json":
                try:
                    values = _walk_strings(json.loads(raw))
                except json.JSONDecodeError:
                    values = raw.splitlines()
            elif path.suffix.lower() == ".jsonl":
                rows: list[str] = []
                for line in raw.splitlines():
                    try:
                        rows.extend(_walk_strings(json.loads(line)))
                    except json.JSONDecodeError:
                        rows.append(line)
                values = rows
            else:
                values = raw.splitlines()
            for value in values:
                normalized = _normalize(value)
                if 2 <= len(normalized) <= 300:
                    fingerprints.add(hashlib.sha256(normalized.encode()).hexdigest())
    return fingerprints, files_scanned


def _near_duplicate(left: str, right: str) -> bool:
    a, b = _normalize(left), _normalize(right)
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    if len(shorter) >= 6 and shorter in longer and len(shorter) / len(longer) >= 0.75:
        return True
    if min(len(a), len(b)) >= 8 and SequenceMatcher(None, a, b).ratio() >= 0.88:
        return True
    return False


def _stage1_replies() -> list[str]:
    payload = json.loads(STAGE1_MANIFEST.read_text(encoding="utf-8"))
    return [str(row["reply"]) for row in payload["cases"]]


def _stage1_events() -> set[str]:
    payload = json.loads(STAGE1_MANIFEST.read_text(encoding="utf-8"))
    return {str(row["event_cluster"]) for row in payload["cases"]}


def _event_rows(messages: list[dict[str, Any]]) -> list[tuple[int, list[dict[str, Any]]]]:
    ordered = sorted(messages, key=lambda row: (stage1._timestamp(row), int(row.get("sortSeq") or 0)))
    events: list[list[dict[str, Any]]] = []
    previous = 0
    for message in ordered:
        timestamp = stage1._timestamp(message)
        if not events or (previous and timestamp - previous > stage1.EVENT_GAP_SECONDS):
            events.append([])
        events[-1].append(message)
        if timestamp:
            previous = timestamp
    return list(enumerate(events))


def _anonymous_context(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Keep turn ownership needed for curation without exposing participant identities."""
    speakers: dict[str, str] = {}
    result: list[dict[str, str]] = []
    for row in rows:
        if row.get("isSend") or row.get("is_send"):
            speaker = "self"
        else:
            raw = str(
                row.get("senderUsername")
                or row.get("senderDisplayName")
                or row.get("sender")
                or "unknown"
            )
            if raw not in speakers:
                speakers[raw] = f"participant_{len(speakers) + 1}"
            speaker = speakers[raw]
        result.append({"speaker": speaker, "content": str(row.get("content") or "").strip()})
    return result


def _common_candidates(
    exposed: set[str],
    *,
    candidate_filter: Callable[[list[dict[str, Any]], int], bool] | None = None,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    sources = (COMMON_SOURCE,)
    if COMMON_SOURCE == _DEFAULT_COMMON_SOURCE:
        sources += tuple(path for path in COMMON_ADDITIONAL_SOURCES if path.exists())
    for source_number, source_path in enumerate(sources):
        raw = source_path.read_bytes()
        payload = json.loads(raw)
        if str(payload.get("talker")) != COMMON_TALKER:
            raise RuntimeError("correct common-prosperity source has unexpected talker")
        source_hash = hashlib.sha256(raw).hexdigest()
        source_namespace = "" if source_number == 0 else f":{source_hash[:16]}"
        for event_number, event in _event_rows(payload.get("messages") or []):
            for index, message in enumerate(event):
                reply = str(message.get("content") or "").strip()
                timestamp = stage1._timestamp(message)
                if not (
                    message.get("isSend")
                    and str(message.get("senderUsername") or "") == SELF_SENDER
                    and int(message.get("localType") or 0) == 1
                    and 0 < timestamp < stage1.HUMAN_CUTOFF
                    and stage1._safe_text(reply)
                ):
                    continue
                context_rows = [
                    row for row in event[:index]
                    if int(row.get("localType") or 0) == 1
                    and str(row.get("content") or "").strip()
                ][-12:]
                if not context_rows or not any(not row.get("isSend") for row in context_rows):
                    continue
                context = [str(row.get("content") or "").strip() for row in context_rows]
                if not all(stage1._safe_text(text) for text in context):
                    continue
                if candidate_filter is not None and not candidate_filter(event, index):
                    continue
                content_fingerprints = {_fingerprint(reply), *(_fingerprint(text) for text in context)}
                if _fingerprint(reply) in exposed:
                    continue
                event_identity = f"{COMMON_TALKER}{source_namespace}:{event_number}"
                cases.append({
                    "internal_id": hashlib.sha256(
                        f"{event_identity}:{timestamp}:{reply}".encode()
                    ).hexdigest()[:20],
                    "source_path": str(source_path.relative_to(ROOT)),
                    "source_sha256": source_hash,
                    "message_index": index,
                    "timestamp": timestamp,
                    "chat_id": "specified_common_prosperity",
                    "chat_type": "group",
                    "event_cluster": hashlib.sha256(event_identity.encode()).hexdigest()[:20],
                    "event_cluster_source_number": event_number,
                    "identity_evidence": "isSend=1 plus verified self sender before default cutoff",
                    "context_complete_within_event": True,
                    "context": context,
                    "context_messages": _anonymous_context(context_rows),
                    "reply": reply,
                    "content_fingerprints": sorted(content_fingerprints),
                    "complex_large_group": True,
                    "source_message_count": len(payload.get("messages") or []),
                    "source_participant_count": len({
                        str(row.get("senderUsername") or "")
                        for row in payload.get("messages") or [] if not row.get("isSend")
                    }),
                    "specified_group": "correct_common_prosperity_with_luxixi",
                })
                break
    return cases


def _youzi_cases() -> list[dict[str, Any]]:
    payload = json.loads(YOUZI_MANIFEST.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for case in payload["cases"]:
        reply = str(case["human_reply"]["content"])
        context = [str(row["content"]) for row in case["context"]]
        rows.append({
            "internal_id": hashlib.sha256(f"youzi:{case['event_hash']}".encode()).hexdigest()[:20],
            "source_path": str(YOUZI_MANIFEST.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(YOUZI_MANIFEST.read_bytes()).hexdigest(),
            "message_index": None,
            "timestamp": int(case["timestamp"]),
            "chat_id": "specified_youzi_group_2",
            "chat_type": "group",
            "event_cluster": "youzi_" + str(case["event_hash"])[:20],
            "identity_evidence": "verified self sender in sealed Youzi quota manifest",
            "context_complete_within_event": True,
            "context": context,
            "context_messages": _anonymous_context(case["context"]),
            "reply": reply,
            "content_fingerprints": sorted({_fingerprint(reply), *(_fingerprint(x) for x in context)}),
            "complex_large_group": True,
            "source_message_count": 20011,
            "source_participant_count": 8,
            "specified_group": "youzi_group_2",
        })
    return rows


def _filter_semantic(
    rows: list[dict[str, Any]], forbidden: list[str], *, deduplicate_within: bool = True
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    comparison = list(forbidden)
    for row in sorted(rows, key=lambda item: (item["timestamp"], item["internal_id"])):
        if any(_near_duplicate(row["reply"], previous) for previous in comparison):
            continue
        accepted.append(row)
        if deduplicate_within:
            comparison.append(row["reply"])
    return accepted


def _ordinary_candidates(
    exposed: set[str], *, per_chat_cap: int = 12
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    stage1_events = _stage1_events()
    for path in sorted(stage1.EXPORT_DIR.glob("*.json")):
        if path.stem == stage1.WRONG_COMMON_PROSPERITY_STEM:
            continue
        try:
            source_bytes = path.read_bytes()
            payload = json.loads(source_bytes)
        except (OSError, json.JSONDecodeError):
            continue
        messages = payload.get("messages") or []
        is_group = path.stem.startswith("群聊_")
        incoming_senders = {
            str(row.get("senderUsername") or row.get("senderDisplayName") or "")
            for row in messages if not row.get("isSend")
        }
        source_is_large = is_group and len(messages) >= 200 and len(incoming_senders) >= 8
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        event_number = 0
        previous_timestamp = 0
        recent: list[dict[str, Any]] = []
        selected_in_chat = 0
        for index, message in enumerate(messages):
            timestamp = stage1._timestamp(message)
            if previous_timestamp and timestamp and timestamp - previous_timestamp > stage1.EVENT_GAP_SECONDS:
                event_number += 1
                recent = []
            if timestamp:
                previous_timestamp = timestamp
            reply = str(message.get("content") or "").strip()
            if (
                message.get("isSend") and int(message.get("localType") or 0) == 1
                and 0 < timestamp < stage1.HUMAN_CUTOFF and stage1._safe_text(reply)
            ):
                context_rows = recent[-12:]
                context = [str(row.get("content") or "").strip() for row in context_rows]
                if (
                    context_rows and any(not row.get("isSend") for row in context_rows)
                    and all(stage1._safe_text(text) for text in context)
                ):
                    fingerprints = {_fingerprint(reply), *(_fingerprint(text) for text in context)}
                    event_key = hashlib.sha256(
                        f"{path.name}:{event_number}".encode()
                    ).hexdigest()[:20]
                    if _fingerprint(reply) not in exposed and event_key not in stage1_events:
                        recent_senders = {
                            str(row.get("senderUsername") or row.get("senderDisplayName") or "self")
                            for row in context_rows if not row.get("isSend")
                        }
                        candidates.append({
                            "internal_id": hashlib.sha256(
                                f"{path.name}:{index}:{timestamp}:{reply}".encode()
                            ).hexdigest()[:20],
                            "source_path": str(path.relative_to(ROOT)),
                            "source_sha256": source_hash,
                            "message_index": index,
                            "timestamp": timestamp,
                            "chat_id": stage1._chat_id(path.stem),
                            "chat_type": "group" if is_group else "private",
                            "event_cluster": event_key,
                            "identity_evidence": "isSend=1 and before default automation cutoff",
                            "context_complete_within_event": True,
                            "context": context,
                            "context_messages": _anonymous_context(context_rows),
                            "reply": reply,
                            "content_fingerprints": sorted(fingerprints),
                            "complex_large_group": bool(
                                source_is_large and (len(recent_senders) >= 2 or len(context_rows) >= 5)
                            ),
                            "source_message_count": len(messages),
                            "source_participant_count": len(incoming_senders) if is_group else 1,
                            "specified_group": None,
                        })
                        selected_in_chat += 1
                        if selected_in_chat >= per_chat_cap:
                            break
            if int(message.get("localType") or 0) == 1 and reply:
                recent.append(message)
                if len(recent) > 12:
                    recent.pop(0)
    return candidates


def _select_round_robin(
    rows: list[dict[str, Any]], count: int, selected: list[dict[str, Any]],
    chat_counts: dict[str, int], used_events: set[str], selected_replies: list[str],
) -> None:
    by_chat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: (item["timestamp"], item["internal_id"])):
        by_chat[row["chat_id"]].append(row)
    while len(selected) < count:
        progressed = False
        for chat_id in sorted(by_chat):
            if chat_counts[chat_id] >= ORDINARY_CHAT_CAP:
                continue
            while by_chat[chat_id]:
                row = by_chat[chat_id].pop(0)
                if row["event_cluster"] in used_events:
                    continue
                if any(_near_duplicate(row["reply"], old) for old in selected_replies):
                    continue
                selected.append(row)
                selected_replies.append(row["reply"])
                used_events.add(row["event_cluster"])
                chat_counts[chat_id] += 1
                progressed = True
                break
            if len(selected) >= count:
                break
        if not progressed:
            break


def select_cases(
    ordinary: list[dict[str, Any]], common: list[dict[str, Any]], youzi: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    stage1_replies = _stage1_replies()
    common = _filter_semantic(common, stage1_replies)
    if len(common) < TARGET_COMMON or len(youzi) != TARGET_YOUZI:
        return common + youzi
    selected = common[:TARGET_COMMON] + youzi
    remaining_common = common[TARGET_COMMON:]
    selected_replies = stage1_replies + [row["reply"] for row in selected]
    used_events = {row["event_cluster"] for row in selected}
    chat_counts: dict[str, int] = defaultdict(int)
    complex_rows = [row for row in ordinary if row["complex_large_group"]]
    group_rows = [row for row in ordinary if row["chat_type"] == "group"]
    private_rows = [row for row in ordinary if row["chat_type"] == "private"]
    _select_round_robin(
        complex_rows, TARGET_COMPLEX, selected, chat_counts, used_events, selected_replies
    )
    while len([row for row in selected if row["complex_large_group"]]) < TARGET_COMPLEX and remaining_common:
        row = remaining_common.pop(0)
        if any(_near_duplicate(row["reply"], old) for old in selected_replies):
            continue
        selected.append(row)
        selected_replies.append(row["reply"])
        used_events.add(row["event_cluster"])
    _select_round_robin(
        group_rows, TARGET_GROUP, selected, chat_counts, used_events, selected_replies
    )
    while sum(row["chat_type"] == "group" for row in selected) < TARGET_GROUP and remaining_common:
        row = remaining_common.pop(0)
        if any(_near_duplicate(row["reply"], old) for old in selected_replies):
            continue
        selected.append(row)
        selected_replies.append(row["reply"])
        used_events.add(row["event_cluster"])
    target_after_private = len(selected) + TARGET_PRIVATE
    _select_round_robin(
        private_rows, target_after_private, selected, chat_counts, used_events, selected_replies
    )
    return selected


def _coverage(rows: list[dict[str, Any]]) -> dict[str, int]:
    groups = [row for row in rows if row["chat_type"] == "group"]
    return {
        "total": len(rows),
        "group_chat": len(groups),
        "private_chat": len(rows) - len(groups),
        "complex_large_group": sum(row["complex_large_group"] for row in groups),
        "distinct_chats": len({row["chat_id"] for row in rows}),
        "distinct_event_clusters": len({row["event_cluster"] for row in rows}),
        "correct_common_prosperity_with_luxixi": sum(
            row.get("specified_group") == "correct_common_prosperity_with_luxixi" for row in rows
        ),
        "youzi_group": sum(row.get("specified_group") == "youzi_group_2" for row in rows),
    }


def _gaps(coverage: dict[str, int]) -> dict[str, int]:
    targets = {
        "total": TARGET_TOTAL, "group_chat": TARGET_GROUP, "private_chat": TARGET_PRIVATE,
        "complex_large_group": TARGET_COMPLEX, "distinct_chats": TARGET_DISTINCT_CHATS,
        "correct_common_prosperity_with_luxixi": TARGET_COMMON, "youzi_group": TARGET_YOUZI,
    }
    return {key: max(0, target - coverage[key]) for key, target in targets.items()}


def _atomic_private_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)


def build() -> tuple[dict[str, Any] | None, dict[str, Any]]:
    exposed, files_scanned = collect_exposed_fingerprints()
    stage1_replies = _stage1_replies()
    ordinary = _filter_semantic(
        _ordinary_candidates(exposed), stage1_replies, deduplicate_within=False
    )
    common = _common_candidates(exposed)
    youzi = _youzi_cases()
    selected = select_cases(ordinary, common, youzi)
    coverage = _coverage(selected)
    gaps = _gaps(coverage)
    ready = not any(gaps.values()) and coverage["total"] == coverage["distinct_event_clusters"]
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    audit = {
        "schema_version": 1,
        "status": "sealed_pending_user_confirmation" if ready else "blocked_insufficient_capacity",
        "audited_at": now,
        "coverage": coverage,
        "gaps": gaps,
        "capacity": {
            "ordinary_after_exact_and_semantic_screen": len(ordinary),
            "ordinary_group": sum(row["chat_type"] == "group" for row in ordinary),
            "ordinary_complex_group": sum(row["complex_large_group"] for row in ordinary),
            "ordinary_private": sum(row["chat_type"] == "private" for row in ordinary),
            "common_after_exact_screen_before_final_semantic_selection": len(common),
            "youzi_sealed_events": len(youzi),
        },
        "checks": {
            "verified_human_identity": True,
            "one_case_per_event": coverage["total"] == coverage["distinct_event_clusters"],
            "stage_1_exact_and_semantic_reply_overlap_excluded": True,
            "known_exact_answer_exposure_excluded": True,
            "input_context_not_misclassified_as_answer_leakage": True,
            "semantic_near_duplicate_screen": "deterministic conservative lexical screen; no human-review claim",
            "wrong_same_name_common_prosperity_excluded": True,
            "model_calls": 0,
            "bot_started": False,
        },
        "exposure_files_scanned": files_scanned,
        "privacy": "hidden case text, chat IDs, event IDs and answers are private",
    }
    if not ready:
        return None, audit
    manifest = {
        **audit,
        "selection_policy": {
            "default_cutoff": "before 2026-04-01T00:00:00+08:00",
            "youzi_cutoff": "through 2026-07-09T23:59:59+08:00",
            "event_gap_seconds": stage1.EVENT_GAP_SECONDS,
            "ordinary_chat_cap": ORDINARY_CHAT_CAP,
            "specified_group_cap": "one per independent event; minimum 20 each",
            "semantic_near_duplicate_claimed_human_review": False,
        },
        "cases": selected,
    }
    _atomic_private_write(MANIFEST_PATH, manifest)
    audit["manifest_sha256"] = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    _atomic_private_write(PUBLIC_SUMMARY_PATH, audit)
    return manifest, audit


def build_target_resolution_candidates(
    output_path: Path | None = None,
) -> dict[str, Any]:
    output_path = output_path or TARGET_RESOLUTION_MANIFEST_PATH
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    exposed, _files_scanned = collect_exposed_fingerprints()
    ordinary = _filter_semantic(
        _ordinary_candidates(exposed), _stage1_replies(), deduplicate_within=False
    )
    selected = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["cases"]
    selected_ids = {str(row["internal_id"]) for row in selected}
    selected_events = {str(row["event_cluster"]) for row in selected}
    selected_replies = [str(row["reply"]) for row in selected]
    candidates = [
        row for row in ordinary
        if row["internal_id"] not in selected_ids
        and row["event_cluster"] not in selected_events
        and row["chat_type"] == "group"
        and row["complex_large_group"]
        and len({
            message["speaker"] for message in row["context_messages"]
            if message["speaker"] != "self"
        }) >= 2
        and not any(_near_duplicate(row["reply"], reply) for reply in selected_replies)
    ]
    by_event = {}
    for row in candidates:
        event = str(row["event_cluster"])
        score = (
            len({
                message["speaker"] for message in row["context_messages"]
                if message["speaker"] != "self"
            }),
            len(row["context_messages"]),
            str(row["internal_id"]),
        )
        current = by_event.get(event)
        if current is None or score > current[0]:
            by_event[event] = (score, row)
    candidates = [item[1] for item in by_event.values()]
    candidates.sort(key=lambda row: (str(row["event_cluster"]), str(row["internal_id"])))
    payload = {
        "schema_version": 1,
        "status": "candidate_screening_only",
        "selection_policy": {
            "not_in_stage2_manifest": True,
            "event_disjoint_from_stage2_manifest": True,
            "group_chat": True,
            "complex_large_group": True,
            "at_least_two_anonymous_participants": True,
            "one_case_per_event": True,
            "event_choice": "most distinct non-self speakers, then longest context",
            "semantic_near_duplicate_with_stage2_excluded": True,
            "model_calls": 0,
        },
        "cases": candidates,
    }
    _atomic_private_write(output_path, payload)
    return {
        "status": payload["status"],
        "candidates": len(candidates),
        "distinct_chats": len({row["chat_id"] for row in candidates}),
        "distinct_events": len({row["event_cluster"] for row in candidates}),
        "path": str(output_path.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-resolution-candidates", action="store_true")
    parser.add_argument(
        "--target-resolution-output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.target_resolution_candidates:
        print(json.dumps(
            build_target_resolution_candidates(args.target_resolution_output),
            ensure_ascii=False,
            indent=2,
        ))
        return
    _manifest, audit = build()
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["status"] != "sealed_pending_user_confirmation":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
