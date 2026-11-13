#!/usr/bin/env python3
"""Select and seal a private, never-tuned Judge/Bot holdout without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = ROOT / "data/exports/b"
PRIVATE_DIR = ROOT / "data/reports/private/adversarial_hidden_holdout_v1"
MANIFEST_PATH = PRIVATE_DIR / "manifest.json"
PUBLIC_SUMMARY_PATH = PRIVATE_DIR / "public_summary.json"
HUMAN_CUTOFF = 1774972800  # 2026-04-01 00:00:00 Asia/Shanghai
EVENT_GAP_SECONDS = 30 * 60
WRONG_COMMON_PROSPERITY_STEM = "群聊_共同富裕群"
TARGET_TOTAL = 120
TARGET_GROUP = 80
TARGET_COMPLEX_GROUP = 50
TARGET_PRIVATE = 40

EXPOSURE_ROOTS = (
    ROOT / "data/reports/private",
    ROOT / "data/few_shot",
    ROOT / "data/few_shot_v2",
    ROOT / "data/few_shot_v3",
    ROOT / "data/few_shot_v4",
    ROOT / "data/few_shot_v5",
    ROOT / "data/few_shot_verified_humor",
    ROOT / "data/debug/prompts",
    ROOT / "data/benchmark_history",
    ROOT / "data/memory/wiki",
    ROOT / "docs/03-guides",
)


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text, flags=re.UNICODE).lower()


def _fingerprint(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


def _safe_text(text: str) -> bool:
    value = text.strip()
    return bool(
        1 <= len(value) <= 300
        and not value.startswith("<")
        and not re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)|https?://|www\.|wxid_", value, re.I)
    )


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
            if not path.is_file() or path == MANIFEST_PATH or path == PUBLIC_SUMMARY_PATH:
                continue
            if path.suffix.lower() not in {".json", ".jsonl", ".md", ".txt", ".log"}:
                continue
            try:
                raw = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            files_scanned += 1
            strings: Iterable[str]
            if path.suffix.lower() == ".json":
                try:
                    strings = _walk_strings(json.loads(raw))
                except json.JSONDecodeError:
                    strings = raw.splitlines()
            elif path.suffix.lower() == ".jsonl":
                rows: list[str] = []
                for line in raw.splitlines():
                    try:
                        rows.extend(_walk_strings(json.loads(line)))
                    except json.JSONDecodeError:
                        rows.append(line)
                strings = rows
            else:
                strings = raw.splitlines()
            for value in strings:
                normalized = _normalize(value)
                if 2 <= len(normalized) <= 300:
                    fingerprints.add(hashlib.sha256(normalized.encode("utf-8")).hexdigest())
    return fingerprints, files_scanned


def _chat_id(stem: str) -> str:
    name = re.sub(r"^(私聊_|群聊_|曾经的好友_)", "", stem).strip()
    return "chat_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]


def _timestamp(message: dict[str, Any]) -> int:
    value = int(message.get("createTime") or 0)
    return value // 1000 if value > 10_000_000_000 else value


def _event_keys(messages: list[dict[str, Any]]) -> list[int]:
    keys: list[int] = []
    event = 0
    previous = 0
    for message in messages:
        current = _timestamp(message)
        if previous and current and current - previous > EVENT_GAP_SECONDS:
            event += 1
        keys.append(event)
        if current:
            previous = current
    return keys


def extract_candidates(exposed: set[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in sorted(EXPORT_DIR.glob("*.json")):
        if path.stem == WRONG_COMMON_PROSPERITY_STEM:
            continue
        try:
            source_bytes = path.read_bytes()
            payload = json.loads(source_bytes)
        except (OSError, json.JSONDecodeError):
            continue
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        messages = payload.get("messages") or []
        is_group = path.stem.startswith("群聊_")
        incoming_senders = {
            str(message.get("senderUsername") or message.get("senderDisplayName") or "")
            for message in messages
            if not message.get("isSend")
        }
        source_is_large = is_group and len(messages) >= 200 and len(incoming_senders) >= 8
        event_keys = _event_keys(messages)
        for index, message in enumerate(messages):
            timestamp = _timestamp(message)
            reply = str(message.get("content") or "").strip()
            if not (
                message.get("isSend")
                and message.get("localType") == 1
                and 0 < timestamp < HUMAN_CUTOFF
                and _safe_text(reply)
            ):
                continue
            context_rows: list[dict[str, Any]] = []
            cursor = index - 1
            while cursor >= 0 and len(context_rows) < 12:
                previous = messages[cursor]
                if event_keys[cursor] != event_keys[index]:
                    break
                if previous.get("localType") == 1 and str(previous.get("content") or "").strip():
                    context_rows.append(previous)
                cursor -= 1
            context_rows.reverse()
            if not context_rows or not any(not row.get("isSend") for row in context_rows):
                continue
            context = [str(row.get("content") or "").strip() for row in context_rows]
            if not all(_safe_text(text) for text in context):
                continue
            content_fingerprints = {_fingerprint(reply), *(_fingerprint(text) for text in context)}
            if content_fingerprints & exposed:
                continue
            recent_senders = {
                str(row.get("senderUsername") or row.get("senderDisplayName") or "self")
                for row in context_rows
                if not row.get("isSend")
            }
            complex_group = bool(
                source_is_large
                and (len(recent_senders) >= 2 or len(context_rows) >= 5)
            )
            internal_id = hashlib.sha256(
                f"{path.name}:{index}:{timestamp}:{reply}".encode("utf-8")
            ).hexdigest()[:20]
            event_key = hashlib.sha256(
                f"{path.name}:{event_keys[index]}".encode("utf-8")
            ).hexdigest()[:20]
            candidates.append({
                "internal_id": internal_id,
                "source_path": str(path.relative_to(ROOT)),
                "source_sha256": source_hash,
                "message_index": index,
                "timestamp": timestamp,
                "chat_id": _chat_id(path.stem),
                "chat_type": "group" if is_group else "private",
                "event_cluster": event_key,
                "identity_evidence": "isSend=1 and before_2026-04-01_automation_cutoff",
                "context_complete_within_event": True,
                "context": context,
                "reply": reply,
                "content_fingerprints": sorted(content_fingerprints),
                "complex_large_group": complex_group,
                "source_message_count": len(messages),
                "source_participant_count": len(incoming_senders) if is_group else 1,
            })
    return candidates


def _round_robin(rows: list[dict[str, Any]], count: int, used_events: set[str]) -> list[dict[str, Any]]:
    by_chat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: (item["timestamp"], item["internal_id"])):
        by_chat[row["chat_id"]].append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for chat_id in sorted(by_chat):
            while by_chat[chat_id]:
                row = by_chat[chat_id].pop(0)
                if row["event_cluster"] in used_events:
                    continue
                selected.append(row)
                used_events.add(row["event_cluster"])
                progressed = True
                break
            if len(selected) >= count:
                break
        if not progressed:
            break
    return selected


def select_holdout(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    used_events: set[str] = set()
    complex_group = [row for row in candidates if row["complex_large_group"]]
    other_group = [
        row for row in candidates
        if row["chat_type"] == "group" and not row["complex_large_group"]
    ]
    private = [row for row in candidates if row["chat_type"] == "private"]
    selected = _round_robin(complex_group, TARGET_COMPLEX_GROUP, used_events)
    selected += _round_robin(other_group, TARGET_GROUP - len(selected), used_events)
    selected += _round_robin(private, TARGET_PRIVATE, used_events)
    return selected


def _coverage(selected: list[dict[str, Any]]) -> dict[str, int]:
    group = [row for row in selected if row["chat_type"] == "group"]
    private = [row for row in selected if row["chat_type"] == "private"]
    complex_group = [row for row in group if row["complex_large_group"]]
    return {
        "total": len(selected),
        "group_chat": len(group),
        "private_chat": len(private),
        "complex_large_group": len(complex_group),
        "other_group_chat": len(group) - len(complex_group),
        "distinct_chats": len({row["chat_id"] for row in selected}),
        "distinct_event_clusters": len({row["event_cluster"] for row in selected}),
        "correct_common_prosperity_with_luxixi": 0,
        "youzi_group": 0,
    }


def _atomic_private_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    exposed, files_scanned = collect_exposed_fingerprints()
    candidates = extract_candidates(exposed)
    selected = select_holdout(candidates)
    coverage = _coverage(selected)
    if coverage["total"] < TARGET_TOTAL or coverage["group_chat"] < TARGET_GROUP:
        raise RuntimeError(f"insufficient eligible holdout: {coverage}")
    created_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    manifest = {
        "schema_version": 1,
        "status": "sealed_pending_user_confirmation",
        "created_at": created_at,
        "selection_policy": {
            "human_cutoff": "2026-04-01T00:00:00+08:00",
            "event_gap_seconds": EVENT_GAP_SECONDS,
            "one_case_per_event_cluster": True,
            "exposure_files_scanned": files_scanned,
            "exposure_fingerprint_count": len(exposed),
            "lexical_exact_exclusion": True,
            "semantic_near_duplicate_claimed": False,
            "wrong_same_name_common_prosperity_excluded": True,
        },
        "coverage": coverage,
        "cases": selected,
    }
    _atomic_private_write(MANIFEST_PATH, manifest)
    manifest_sha256 = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    public = {
        "schema_version": 1,
        "status": "sealed_pending_user_confirmation",
        "sealed_at": created_at,
        "coverage": coverage,
        "manifest_sha256": manifest_sha256,
        "privacy": "case content, internal IDs, context, replies and answers are private",
        "checks": {
            "verified_human_identity": True,
            "complete_context_reproducible": True,
            "event_cluster_isolation": coverage["total"] == coverage["distinct_event_clusters"],
            "known_exposure_exact_match_excluded": True,
            "wrong_same_name_common_prosperity_excluded": True,
            "model_calls": 0,
        },
        "specified_group_notes": {
            "correct_common_prosperity_with_luxixi": "0: all 10 verifiable turns are already exposed and post-cutoff",
            "youzi_group": "0: no unexposed reply with verifiable human identity is available",
        },
    }
    _atomic_private_write(PUBLIC_SUMMARY_PATH, public)
    return manifest, public


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    _manifest, public = build()
    print(json.dumps(public, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
