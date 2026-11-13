#!/usr/bin/env python3
"""Build a chat-isolated, resumable human-vs-C0 Judge benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.build_persona_few_shots import Candidate, extract_candidates
from scripts.evaluate_human_ai_judge import (
    CUTOFF,
    PROTECTED_C0_SHA256,
    VOLCENGINE_DSV4F_MODEL,
    VolcengineAnthropicClient,
    _assert_c0,
    _case_id,
    _export_payloads,
    _few_shot_identities,
    _json_sha256,
    _normalized_reply,
    _previously_exposed_case_ids,
    _run_c0,
    _write_private,
    build_blind_packet,
    build_judge_reference_pack,
    detect_degenerate_case,
    load_volcengine_credentials,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIVATE_DIR = PROJECT_ROOT / "data/reports/private/human_ai_judge_benchmark_v2"
DEFAULT_MANIFEST = PRIVATE_DIR / "manifest.json"
DEFAULT_CHECKPOINT = PRIVATE_DIR / "generation_checkpoint.json"
DEFAULT_REFERENCE = PRIVATE_DIR / "judge_reference.json"
DEFAULT_FREEZE = PRIVATE_DIR / "freeze.json"
DEFAULT_EXISTING_REPLAY = PROJECT_ROOT / "data/reports/private/human_ai_judge_revised_replay.json"
DEFAULT_EXPORTS = PROJECT_ROOT / "data/exports/b"
DEFAULT_WIKI = PROJECT_ROOT / "data/memory/wiki/users"
DEFAULT_VOLCENGINE_CONFIG = Path.home() / ".cc-connect/config.toml"
SPLIT_TARGETS = {"dev": 50, "validation": 30, "final": 100}
EXISTING_DEV_TARGET = 30
NEW_TARGETS = {"dev": 20, "validation": 30, "final": 100}
SPLIT_ORDER = ("dev", "validation", "final")
SEED = "human-ai-judge-benchmark-v2-20260818"
MAX_CASES_PER_CHAT = 6


def stable_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_record(candidate: Candidate) -> dict[str, Any]:
    row = asdict(candidate)
    row["style_tags"] = list(candidate.style_tags)
    row["case_id"] = _case_id(candidate)
    row["relationship"] = "group" if candidate.relation == "group" else "private"
    return row


def candidate_from_record(row: dict[str, Any]) -> Candidate:
    values = dict(row)
    for key in ("case_id", "relationship"):
        values.pop(key, None)
    values["style_tags"] = tuple(values.get("style_tags") or ())
    return Candidate(**values)


def _rank(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def allocate_chat_splits(
    candidates: list[Candidate],
    *,
    excluded_chat_ids: set[str],
) -> dict[str, list[Candidate]]:
    """Assign whole chats to one split, then put each chat's first case first."""
    by_relation: dict[str, dict[str, list[Candidate]]] = defaultdict(lambda: defaultdict(list))
    for candidate in candidates:
        if candidate.chat_id in excluded_chat_ids:
            continue
        relation = "group" if candidate.relation == "group" else "private"
        by_relation[relation][candidate.chat_id].append(candidate)

    assigned: dict[str, dict[str, list[Candidate]]] = {
        split: defaultdict(list) for split in SPLIT_ORDER
    }
    total_new = sum(NEW_TARGETS.values())
    for relation, chats in sorted(by_relation.items()):
        chat_ids = sorted(chats, key=lambda chat_id: _rank(f"chat:{relation}:{chat_id}"))
        dev_end = round(len(chat_ids) * NEW_TARGETS["dev"] / total_new)
        validation_end = dev_end + round(len(chat_ids) * NEW_TARGETS["validation"] / total_new)
        boundaries = {
            "dev": chat_ids[:dev_end],
            "validation": chat_ids[dev_end:validation_end],
            "final": chat_ids[validation_end:],
        }
        for split, split_chat_ids in boundaries.items():
            for chat_id in split_chat_ids:
                assigned[split][chat_id] = sorted(
                    chats[chat_id], key=lambda row: _rank(f"case:{_case_id(row)}")
                )[:MAX_CASES_PER_CHAT]

    queues: dict[str, list[Candidate]] = {}
    for split in SPLIT_ORDER:
        chats = assigned[split]
        ordered_chat_ids = sorted(chats, key=lambda chat_id: _rank(f"queue:{split}:{chat_id}"))
        queue = []
        depth = 0
        while True:
            added = 0
            for chat_id in ordered_chat_ids:
                if depth < len(chats[chat_id]):
                    queue.append(chats[chat_id][depth])
                    added += 1
            if not added:
                break
            depth += 1
        queues[split] = queue
    return queues


def validate_chat_isolation(
    queues: dict[str, list[dict[str, Any]]], existing_dev_chat_ids: set[str]
) -> None:
    split_chats = {
        split: {str(row["chat_id"]) for row in rows}
        for split, rows in queues.items()
    }
    split_chats["dev"].update(existing_dev_chat_ids)
    for index, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[index + 1:]:
            overlap = split_chats[left] & split_chats[right]
            if overlap:
                raise ValueError(f"chat isolation failed for {left}/{right}: {len(overlap)} overlaps")


def _eligible_candidates(
    exports: Path, wiki_dir: Path, report_dir: Path
) -> tuple[list[Candidate], list[Candidate], int, int]:
    candidates = extract_candidates(exports, wiki_dir, human_before=CUTOFF)
    excluded = _few_shot_identities(PROJECT_ROOT / "data")
    exposed_case_ids = _previously_exposed_case_ids(report_dir)
    exposed_chat_ids = {
        row.chat_id for row in candidates if _case_id(row) in exposed_case_ids
    }
    eligible = [
        row for row in candidates
        if (
            tuple((row.relation, row.chat_id, tuple(row.context), tuple(row.replies))) not in excluded
            and row.chat_id not in exposed_chat_ids
            and 2 <= len("".join(row.replies)) <= 80
            and 2 <= len("".join(row.context)) <= 240
            and row.timestamp
            and row.timestamp / (1000 if row.timestamp > 10_000_000_000 else 1) < CUTOFF
        )
    ]
    return candidates, eligible, len(excluded), len(exposed_chat_ids)


def build_manifest(
    all_candidates: list[Candidate],
    new_candidates: list[Candidate],
    existing_replay: dict[str, Any],
    *,
    few_shot_excluded_count: int,
    previously_exposed_chat_count: int,
    existing_replay_sha256: str,
) -> dict[str, Any]:
    by_case = {_case_id(row): row for row in all_candidates}
    existing_case_ids = [str(row["case_id"]) for row in existing_replay.get("cases") or []]
    existing_dev_chat_ids = {
        by_case[case_id].chat_id for case_id in existing_case_ids if case_id in by_case
    }
    if len(existing_case_ids) != EXISTING_DEV_TARGET:
        raise ValueError(f"expected {EXISTING_DEV_TARGET} existing dev cases, got {len(existing_case_ids)}")
    if len(existing_dev_chat_ids) != EXISTING_DEV_TARGET:
        raise ValueError("existing dev cases do not resolve to 30 unique source chats")

    queues = allocate_chat_splits(new_candidates, excluded_chat_ids=existing_dev_chat_ids)
    queue_records = {
        split: [candidate_record(row) for row in queue]
        for split, queue in queues.items()
    }
    validate_chat_isolation(queue_records, existing_dev_chat_ids)
    all_test_chats = existing_dev_chat_ids | {
        str(row["chat_id"]) for rows in queue_records.values() for row in rows
    }
    all_target_replies = {
        _normalized_reply(row["replies"]) for rows in queue_records.values() for row in rows
    }
    reference = build_judge_reference_pack(
        all_candidates,
        excluded_chat_ids=all_test_chats,
        excluded_replies=all_target_replies,
    )
    queue_summary = {}
    for split, rows in queue_records.items():
        unique_chats = {row["chat_id"] for row in rows}
        queue_summary[split] = {
            "new_valid_target": NEW_TARGETS[split],
            "queue_cases": len(rows),
            "unique_chats": len(unique_chats),
            "relationships": dict(Counter(row["relationship"] for row in rows)),
        }
    manifest = {
        "schema_version": 2,
        "benchmark_kind": "human_ai_reply_discrimination_180",
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "protocol": {
            "total_cases": 180,
            "split_targets": SPLIT_TARGETS,
            "existing_dev_cases": EXISTING_DEV_TARGET,
            "new_case_targets": NEW_TARGETS,
            "one_case_per_chat_first": True,
            "whole_chat_split_isolation": True,
            "judge_calls": "one fresh API session per case; no tools; no cross-case context",
            "validation_visibility": "aggregate and strata only during iteration",
            "final_visibility": "sealed until Judge design is frozen; run once",
        },
        "sources": {
            "human_before": "2026-04-01T00:00:00+08:00",
            "c0_generator_sha256": _assert_c0(),
            "existing_dev_replay_sha256": existing_replay_sha256,
            "few_shot_exact_rows_excluded": few_shot_excluded_count,
            "previously_exposed_chats_excluded_from_new_cases": previously_exposed_chat_count,
            "eligible_candidates": len(new_candidates),
            "candidate_pool_sha256": _json_sha256([
                candidate_record(row) for row in new_candidates
            ]),
        },
        "existing_dev": {
            "case_ids": existing_case_ids,
            "chat_ids": sorted(existing_dev_chat_ids),
            "cases": [
                {"case_id": case_id, "chat_id": by_case[case_id].chat_id}
                for case_id in existing_case_ids
            ],
        },
        "queue_summary": queue_summary,
        "queues": queue_records,
        "reference": reference,
    }
    manifest["manifest_sha256"] = _json_sha256(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    expected_hash = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if expected_hash != _json_sha256(unhashed):
        raise ValueError("manifest hash mismatch")
    if manifest.get("sources", {}).get("c0_generator_sha256") != PROTECTED_C0_SHA256:
        raise ValueError("manifest is not bound to protected C0")
    validate_chat_isolation(
        manifest["queues"], set(manifest["existing_dev"]["chat_ids"])
    )
    seen_case_ids = []
    for rows in manifest["queues"].values():
        seen_case_ids.extend(str(row["case_id"]) for row in rows)
    if len(seen_case_ids) != len(set(seen_case_ids)):
        raise ValueError("candidate case appears more than once")


def _load_checkpoint(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if checkpoint.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("checkpoint belongs to a different manifest")
        return checkpoint
    return {
        "schema_version": 2,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "manifest_sha256": manifest["manifest_sha256"],
        "c0_generator_sha256": PROTECTED_C0_SHA256,
        "entries": {},
    }


def generate_split(
    split: str,
    manifest: dict[str, Any],
    checkpoint_path: Path,
    exports: Path,
    volcengine_config: Path,
    *,
    max_calls: int | None,
) -> dict[str, Any]:
    from src.memory import MemoryEngine

    if split not in SPLIT_ORDER:
        raise ValueError(f"unknown split: {split}")
    _assert_c0()
    checkpoint = _load_checkpoint(checkpoint_path, manifest)
    entries = checkpoint.setdefault("entries", {})
    valid_before = sum(
        row.get("split") == split and row.get("status") == "valid"
        for row in entries.values()
    )
    if valid_before >= NEW_TARGETS[split]:
        return {"split": split, "api_calls": 0, "valid": valid_before, "target": NEW_TARGETS[split]}

    api_key, base_url = load_volcengine_credentials(volcengine_config)
    client = VolcengineAnthropicClient(api_key, base_url, VOLCENGINE_DSV4F_MODEL)
    memory = MemoryEngine(llm_client=client)
    payloads = _export_payloads(exports)
    calls = 0
    try:
        for row in manifest["queues"][split]:
            case_id = str(row["case_id"])
            if case_id in entries:
                continue
            if max_calls is not None and calls >= max_calls:
                break
            candidate = candidate_from_record(row)
            payload = payloads.get(candidate.chat_id)
            if not payload:
                entries[case_id] = {"split": split, "status": "missing_export"}
                _write_private(checkpoint_path, checkpoint)
                continue
            calls += 1
            try:
                output = _run_c0(candidate, payload, client, memory)
                if output["generation_failed"] or not output["replies"]:
                    status = "generation_failed" if output["generation_failed"] else "silent"
                    entries[case_id] = {
                        "split": split,
                        "chat_id": candidate.chat_id,
                        "relationship": row["relationship"],
                        "status": status,
                        "duration_seconds": output["duration_seconds"],
                    }
                else:
                    reasons = detect_degenerate_case(candidate.replies, output)
                    if reasons:
                        entries[case_id] = {
                            "split": split,
                            "chat_id": candidate.chat_id,
                            "relationship": row["relationship"],
                            "status": "degenerate",
                            "reasons": reasons,
                        }
                    else:
                        entries[case_id] = {
                            "split": split,
                            "chat_id": candidate.chat_id,
                            "relationship": row["relationship"],
                            "status": "valid",
                            "replay_case": {
                                "case_id": case_id,
                                "source_chat_id": candidate.chat_id,
                                "relationship": row["relationship"],
                                "timestamp": candidate.timestamp,
                                "context_original": output["context_original"],
                                "human_reply_original": candidate.replies,
                                "ai_reply_original": output["replies"],
                                "c0_trace": {
                                    "few_shot_ids": output["few_shot_ids"],
                                    "duration_seconds": output["duration_seconds"],
                                    "input_metadata": output["input_metadata"],
                                    "generation_trace": output["trace"],
                                },
                            },
                        }
                _write_private(checkpoint_path, checkpoint)
            except Exception as exc:
                entries[case_id] = {
                    "split": split,
                    "chat_id": candidate.chat_id,
                    "relationship": row["relationship"],
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                _write_private(checkpoint_path, checkpoint)

            valid = sum(
                item.get("split") == split and item.get("status") == "valid"
                for item in entries.values()
            )
            if valid >= NEW_TARGETS[split]:
                break
    finally:
        memory.shutdown()
    _assert_c0()
    outcomes = Counter(
        item.get("status", "unknown") for item in entries.values() if item.get("split") == split
    )
    return {
        "split": split,
        "api_calls": calls,
        "valid": outcomes["valid"],
        "target": NEW_TARGETS[split],
        "outcomes": dict(outcomes),
    }


def surface_difficulty(case: dict[str, Any]) -> str:
    human = _normalized_reply(case["human_reply_original"])
    ai = _normalized_reply(case["ai_reply_original"])
    if not human or not ai:
        return "ordinary"
    ratio = len(human) / len(ai)
    if 0.5 <= ratio <= 2.0 and abs(len(human) - len(ai)) <= 16:
        return "hard_surface"
    return "ordinary"


def _split_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(cases),
        "unique_chats": len({row["source_chat_id"] for row in cases}),
        "relationships": dict(Counter(row["relationship"] for row in cases)),
        "difficulty": dict(Counter(surface_difficulty(row) for row in cases)),
    }


def freeze_benchmark(
    manifest: dict[str, Any],
    checkpoint_path: Path,
    existing_replay_path: Path,
    output_dir: Path,
    freeze_path: Path,
) -> dict[str, Any]:
    if freeze_path.exists():
        raise ValueError("benchmark is already frozen; refusing to overwrite")
    checkpoint = _load_checkpoint(checkpoint_path, manifest)
    existing_replay = json.loads(existing_replay_path.read_text(encoding="utf-8"))
    existing_chat_by_case = {
        str(row["case_id"]): str(row["chat_id"])
        for row in manifest["existing_dev"]["cases"]
    }
    existing_cases = []
    for case in existing_replay.get("cases") or []:
        source_chat_id = existing_chat_by_case.get(str(case["case_id"]))
        if source_chat_id is None:
            raise ValueError(f"existing dev case has no source chat: {case['case_id']}")
        enriched = dict(case)
        enriched["source_chat_id"] = source_chat_id
        existing_cases.append(enriched)

    generated = defaultdict(list)
    for entry in checkpoint.get("entries", {}).values():
        if entry.get("status") == "valid":
            generated[entry["split"]].append(entry["replay_case"])
    for split in SPLIT_ORDER:
        generated[split].sort(key=lambda row: _rank(f"freeze:{row['case_id']}"))
        generated[split] = generated[split][:NEW_TARGETS[split]]
        if len(generated[split]) != NEW_TARGETS[split]:
            raise ValueError(
                f"split {split} has {len(generated[split])} valid new cases; need {NEW_TARGETS[split]}"
            )
    split_cases = {
        "dev": existing_cases + generated["dev"],
        "validation": generated["validation"],
        "final": generated["final"],
    }
    # Recheck the actual selected cases, not only candidate queues.
    actual_chats = {
        split: {row["source_chat_id"] for row in cases}
        for split, cases in split_cases.items()
    }
    for index, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[index + 1:]:
            if actual_chats[left] & actual_chats[right]:
                raise ValueError(f"frozen cases leak chats across {left}/{right}")

    hashes = {}
    summaries = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, cases in split_cases.items():
        blind, answers = build_blind_packet(
            cases,
            batch_id=f"benchmark_v2_{split}",
            reference_sha256=manifest["reference"]["reference_sha256"],
        )
        replay = {
            "schema_version": 2,
            "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
            "split": split,
            "manifest_sha256": manifest["manifest_sha256"],
            "cases": cases,
        }
        paths = {
            "replay": output_dir / f"{split}_replay.json",
            "blind": output_dir / f"{split}_blind.json",
            "answers": output_dir / f"{split}_answers.json",
        }
        _write_private(paths["replay"], replay)
        _write_private(paths["blind"], blind)
        _write_private(paths["answers"], answers)
        hashes[split] = {name: stable_file_sha256(path) for name, path in paths.items()}
        summaries[split] = _split_summary(cases)

    freeze = {
        "schema_version": 2,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest["manifest_sha256"],
        "c0_generator_sha256": _assert_c0(),
        "reference_sha256": manifest["reference"]["reference_sha256"],
        "split_summaries": summaries,
        "artifact_sha256": hashes,
        "visibility": {
            "dev": "open",
            "validation": "aggregate_only",
            "final": "sealed",
        },
        "final_test": {
            "run_limit": 1,
            "runs_consumed": 0,
            "unsealed_at": None,
        },
    }
    freeze["freeze_sha256"] = _json_sha256(freeze)
    _write_private(freeze_path, freeze)
    _write_private(output_dir / "judge_reference.json", manifest["reference"])
    return freeze


def status(manifest: dict[str, Any], checkpoint_path: Path, freeze_path: Path) -> dict[str, Any]:
    checkpoint = _load_checkpoint(checkpoint_path, manifest)
    rows = checkpoint.get("entries", {}).values()
    splits = {}
    for split in SPLIT_ORDER:
        outcomes = Counter(row.get("status", "unknown") for row in rows if row.get("split") == split)
        splits[split] = {
            "new_valid": outcomes["valid"],
            "new_valid_target": NEW_TARGETS[split],
            "outcomes": dict(outcomes),
        }
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "c0_generator_sha256": _assert_c0(),
        "splits": splits,
        "frozen": freeze_path.exists(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 180-case Judge benchmark v2")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--freeze-file", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--existing-replay", type=Path, default=DEFAULT_EXISTING_REPLAY)
    parser.add_argument("--exports", type=Path, default=DEFAULT_EXPORTS)
    parser.add_argument("--wiki-dir", type=Path, default=DEFAULT_WIKI)
    parser.add_argument("--volcengine-config", type=Path, default=DEFAULT_VOLCENGINE_CONFIG)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--generate", choices=SPLIT_ORDER)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    selected_actions = sum(bool(value) for value in (args.initialize, args.generate, args.freeze))
    if selected_actions > 1:
        parser.error("choose only one of --initialize, --generate, or --freeze")

    if args.initialize:
        if args.manifest.exists():
            raise ValueError("manifest already exists; refusing to replace frozen split assignment")
        all_candidates, new_candidates, excluded_count, exposed_chat_count = _eligible_candidates(
            args.exports, args.wiki_dir, args.existing_replay.parent
        )
        replay = json.loads(args.existing_replay.read_text(encoding="utf-8"))
        manifest = build_manifest(
            all_candidates,
            new_candidates,
            replay,
            few_shot_excluded_count=excluded_count,
            previously_exposed_chat_count=exposed_chat_count,
            existing_replay_sha256=stable_file_sha256(args.existing_replay),
        )
        _write_private(args.manifest, manifest)
        _write_private(args.reference, manifest["reference"])
        print(json.dumps({
            "manifest_sha256": manifest["manifest_sha256"],
            "queue_summary": manifest["queue_summary"],
        }, ensure_ascii=False, indent=2))
        return

    if not args.manifest.exists():
        raise ValueError("manifest does not exist; run --initialize first")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if args.generate:
        result = generate_split(
            args.generate,
            manifest,
            args.checkpoint,
            args.exports,
            args.volcengine_config,
            max_calls=args.max_calls,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.freeze:
        result = freeze_benchmark(
            manifest,
            args.checkpoint,
            args.existing_replay,
            args.manifest.parent,
            args.freeze_file,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(json.dumps(status(manifest, args.checkpoint, args.freeze_file), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
