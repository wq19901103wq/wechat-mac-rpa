#!/usr/bin/env python3
"""Build the balanced 80-private + 80-group Judge development benchmark."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_human_ai_judge_benchmark_v2 import (
    _eligible_candidates,
    candidate_from_record,
    candidate_record,
    stable_file_sha256,
)
from scripts.evaluate_human_ai_judge import (
    _export_payloads,
    _json_sha256,
    _normalized_reply,
    _run_c0,
    _write_private,
    assert_reference_isolated,
    build_blind_packet,
    build_judge_reference_pack,
    detect_degenerate_case,
    reference_leakage_details,
)
from src.reply.c0_config import current_c0_baseline_id, load_current_c0
from src.reply.c0_factory import create_current_c0_clients, create_current_c0_generator

PRIVATE_DIR = PROJECT_ROOT / "data/reports/private/human_ai_judge_benchmark_v3_balanced160"
DEFAULT_MANIFEST = PRIVATE_DIR / "manifest.json"
DEFAULT_CHECKPOINT = PRIVATE_DIR / "generation_checkpoint.json"
DEFAULT_REFERENCE = PRIVATE_DIR / "judge_reference.json"
DEFAULT_FREEZE = PRIVATE_DIR / "freeze.json"
DEFAULT_EXPORTS = PROJECT_ROOT / "data/exports/b"
DEFAULT_WIKI = PROJECT_ROOT / "data/memory/wiki/users"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "data/reports/private"
DEFAULT_VOLCENGINE_CONFIG = Path.home() / ".cc-connect/config.toml"
RELATIONS = ("private", "group")
TARGETS = {"private": 80, "group": 80}
TOTAL_TARGET = 160
SEED = "human-ai-judge-benchmark-v3-balanced160-20260825"
MAX_CASES_PER_CHAT = 6


def current_c0_sha256() -> str:
    return hashlib.sha256((PROJECT_ROOT / "src/reply/generator.py").read_bytes()).hexdigest()


def _few_shot_ids_by_identity(path: Path) -> dict[tuple[Any, ...], set[str]]:
    by_identity: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = (
                row.get("relationship"),
                row.get("chat_id"),
                tuple(row.get("context") or []),
                tuple(row.get("reply") or []),
            )
            by_identity[identity].add(str(row["id"]))
    return by_identity


def _candidate_few_shot_ids(
    candidate: Any,
    by_identity: dict[tuple[Any, ...], set[str]],
) -> set[str]:
    identity = (
        candidate.relation,
        candidate.chat_id,
        tuple(candidate.context),
        tuple(candidate.replies),
    )
    return set(by_identity.get(identity, set()))


def _rank(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def relationship_of(row: Any) -> str:
    relation = row.relation if hasattr(row, "relation") else row.get("relationship")
    return "group" if relation == "group" else "private"


def build_relation_queues(candidates: list[Any]) -> dict[str, list[Any]]:
    """Round-robin chats so distinct chats appear before a second case from one chat."""
    queues: dict[str, list[Any]] = {}
    for relation in RELATIONS:
        by_chat: dict[str, list[Any]] = defaultdict(list)
        for candidate in candidates:
            if relationship_of(candidate) == relation:
                by_chat[str(candidate.chat_id)].append(candidate)
        for chat_id, rows in by_chat.items():
            rows.sort(key=lambda row: _rank(f"case:{chat_id}:{candidate_record(row)['case_id']}"))
            del rows[MAX_CASES_PER_CHAT:]
        ordered_chats = sorted(by_chat, key=lambda chat_id: _rank(f"chat:{relation}:{chat_id}"))
        queue: list[Any] = []
        for depth in range(MAX_CASES_PER_CHAT):
            queue.extend(by_chat[chat_id][depth] for chat_id in ordered_chats if depth < len(by_chat[chat_id]))
        if len(queue) < TARGETS[relation]:
            raise ValueError(
                f"{relation} has {len(queue)} eligible candidates; need exactly {TARGETS[relation]}"
            )
        queues[relation] = queue
    return queues


def build_manifest(
    all_candidates: list[Any],
    eligible_candidates: list[Any],
    *,
    few_shot_excluded_count: int,
    previously_exposed_chat_count: int,
) -> dict[str, Any]:
    queues = build_relation_queues(eligible_candidates)
    queue_records = {
        relation: [candidate_record(row) for row in rows]
        for relation, rows in queues.items()
    }
    case_ids = [str(row["case_id"]) for rows in queue_records.values() for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("candidate case appears more than once")
    test_chat_ids = {
        str(row["chat_id"]) for rows in queue_records.values() for row in rows
    }
    target_replies = {
        _normalized_reply(row["replies"]) for rows in queue_records.values() for row in rows
    }
    reference = build_judge_reference_pack(
        all_candidates,
        excluded_chat_ids=test_chat_ids,
        excluded_replies=target_replies,
    )
    c0_hash = current_c0_sha256()
    c0_baseline = current_c0_baseline_id()
    manifest = {
        "schema_version": 3,
        "benchmark_kind": "human_ai_reply_discrimination_development_balanced160",
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "protocol": {
            "purpose": "Judge development comparison; not the C0 fixed 240-case test",
            "targets": TARGETS,
            "total_cases": TOTAL_TARGET,
            "force_reply": True,
            "one_case_per_chat_first": True,
            "other_relation_may_not_fill_shortfall": True,
            "judge_calls": "one fresh API session per case; no tools; no cross-case context",
        },
        "sources": {
            "current_c0_baseline_id": c0_baseline,
            "c0_generator_sha256": c0_hash,
            "few_shot_exact_rows_excluded": few_shot_excluded_count,
            "previously_exposed_chats_excluded": previously_exposed_chat_count,
            "eligible_candidates": len(eligible_candidates),
            "candidate_pool_sha256": _json_sha256(
                [candidate_record(row) for row in eligible_candidates]
            ),
        },
        "queue_summary": {
            relation: {
                "target": TARGETS[relation],
                "queue_cases": len(rows),
                "unique_chats": len({str(row["chat_id"]) for row in rows}),
            }
            for relation, rows in queue_records.items()
        },
        "queues": queue_records,
        "reference": reference,
    }
    manifest["manifest_sha256"] = _json_sha256(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    unhashed = dict(manifest)
    expected_hash = unhashed.pop("manifest_sha256", None)
    if expected_hash != _json_sha256(unhashed):
        raise ValueError("manifest hash mismatch")
    if manifest.get("protocol", {}).get("targets") != TARGETS:
        raise ValueError("manifest must require exactly private 80 and group 80")
    if manifest.get("protocol", {}).get("force_reply") is not True:
        raise ValueError("manifest must require forced C0 replies")
    if manifest.get("sources", {}).get("c0_generator_sha256") != current_c0_sha256():
        raise ValueError("current C0 differs from the C0 frozen in this manifest")
    if manifest.get("sources", {}).get("current_c0_baseline_id") != current_c0_baseline_id():
        raise ValueError("manifest is not bound to the adopted current C0 baseline")
    seen: list[str] = []
    for relation in RELATIONS:
        rows = manifest.get("queues", {}).get(relation, [])
        if len(rows) < TARGETS[relation]:
            raise ValueError(
                f"{relation} has {len(rows)} queued candidates; need exactly {TARGETS[relation]}"
            )
        if any(relationship_of(row) != relation for row in rows):
            raise ValueError(f"{relation} queue contains a different relationship")
        seen.extend(str(row["case_id"]) for row in rows)
    if len(seen) != len(set(seen)):
        raise ValueError("candidate case appears more than once")


def _load_checkpoint(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if checkpoint.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("checkpoint belongs to a different manifest")
        if checkpoint.get("c0_generator_sha256") != current_c0_sha256():
            raise ValueError("current C0 differs from the checkpoint C0")
        if checkpoint.get("current_c0_baseline_id") != current_c0_baseline_id():
            raise ValueError("checkpoint is not bound to the adopted current C0 baseline")
        return checkpoint
    return {
        "schema_version": 3,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "manifest_sha256": manifest["manifest_sha256"],
        "current_c0_baseline_id": current_c0_baseline_id(),
        "c0_generator_sha256": current_c0_sha256(),
        "entries": {},
    }


def should_generate_case(entry: dict[str, Any] | None, *, retry_failed: bool) -> bool:
    if entry is None:
        return True
    return retry_failed and entry.get("status") in {"generation_failed", "error"}


def relation_shard_rows(rows: list[dict[str, Any]], shard_index: int, shard_count: int) -> list[dict[str, Any]]:
    if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard index must be within shard count")
    return [row for index, row in enumerate(rows) if index % shard_count == shard_index]


def merge_checkpoint_data(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    for key in ("manifest_sha256", "current_c0_baseline_id", "c0_generator_sha256"):
        if target.get(key) != source.get(key):
            raise ValueError(f"checkpoint {key} mismatch")
    merged = dict(target)
    entries = dict(target.get("entries", {}))
    for case_id, entry in source.get("entries", {}).items():
        if case_id in entries and entries[case_id] != entry:
            raise ValueError(f"checkpoint entry conflict: {case_id}")
        entries[case_id] = entry
    merged["entries"] = entries
    return merged


def generate_relation(
    relation: str,
    manifest: dict[str, Any],
    checkpoint_path: Path,
    exports: Path,
    volcengine_config: Path,
    *,
    max_calls: int | None,
    retry_failed: bool,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    from src.memory import MemoryEngine
    from src.tools.tool_registry import ToolRegistry

    if relation not in RELATIONS:
        raise ValueError(f"unknown relationship: {relation}")
    relation_shard_rows([], shard_index, shard_count)
    validate_manifest(manifest)
    checkpoint = _load_checkpoint(checkpoint_path, manifest)
    entries = checkpoint.setdefault("entries", {})
    valid_before = sum(
        row.get("relationship") == relation and row.get("status") == "valid"
        for row in entries.values()
    )
    shard_target = math.ceil(TARGETS[relation] / shard_count)
    if valid_before >= shard_target:
        return {"relationship": relation, "api_calls": 0, "valid": valid_before, "target": shard_target}

    config = load_current_c0()
    private_client, group_client = create_current_c0_clients(
        config=config,
        volcengine_config=volcengine_config,
    )
    memory = MemoryEngine(llm_client=private_client)
    generator = create_current_c0_generator(
        config=config,
        volcengine_config=volcengine_config,
        private_client=private_client,
        group_client=group_client,
        memory_engine=memory,
        tool_registry=ToolRegistry(),
        judge_worker=None,
        enable_timestamps=True,
    )
    few_shot_ids = _few_shot_ids_by_identity(Path(config["few_shot_path"]))
    payloads = _export_payloads(exports)
    calls = 0
    try:
        rows = relation_shard_rows(manifest["queues"][relation], shard_index, shard_count)
        for row in rows:
            case_id = str(row["case_id"])
            if not should_generate_case(entries.get(case_id), retry_failed=retry_failed):
                continue
            if max_calls is not None and calls >= max_calls:
                break
            candidate = candidate_from_record(row)
            payload = payloads.get(candidate.chat_id)
            if not payload:
                entries[case_id] = {"relationship": relation, "status": "missing_export"}
                _write_private(checkpoint_path, checkpoint)
                continue
            calls += 1
            try:
                output = _run_c0(
                    candidate,
                    payload,
                    force_reply=True,
                    reply_generator=generator,
                    few_shot_exclude_ids=_candidate_few_shot_ids(candidate, few_shot_ids),
                )
                if output["generation_failed"] or not output["replies"]:
                    status = "generation_failed" if output["generation_failed"] else "silent"
                    entries[case_id] = {
                        "relationship": relation,
                        "chat_id": candidate.chat_id,
                        "status": status,
                        "duration_seconds": output["duration_seconds"],
                    }
                else:
                    reasons = detect_degenerate_case(candidate.replies, output)
                    if reasons:
                        entries[case_id] = {
                            "relationship": relation,
                            "chat_id": candidate.chat_id,
                            "status": "degenerate",
                            "reasons": reasons,
                        }
                    else:
                        entries[case_id] = {
                            "relationship": relation,
                            "chat_id": candidate.chat_id,
                            "status": "valid",
                            "replay_case": {
                                "case_id": case_id,
                                "source_chat_id": candidate.chat_id,
                                "relationship": relation,
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
            except Exception as exc:
                entries[case_id] = {
                    "relationship": relation,
                    "chat_id": candidate.chat_id,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            _write_private(checkpoint_path, checkpoint)
            valid = sum(
                item.get("relationship") == relation and item.get("status") == "valid"
                for item in entries.values()
            )
            if valid >= shard_target:
                break
    finally:
        memory.shutdown()
    if current_c0_sha256() != manifest["sources"]["c0_generator_sha256"]:
        raise ValueError("C0 changed while generation was running")
    if current_c0_baseline_id() != manifest["sources"]["current_c0_baseline_id"]:
        raise ValueError("adopted C0 baseline changed while generation was running")
    outcomes = Counter(
        row.get("status", "unknown")
        for row in entries.values()
        if row.get("relationship") == relation
    )
    return {
        "relationship": relation,
        "api_calls": calls,
        "valid": outcomes["valid"],
        "target": shard_target,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "outcomes": dict(outcomes),
    }


def selected_cases(manifest: dict[str, Any], checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    entries = checkpoint.get("entries", {})
    selected: list[dict[str, Any]] = []
    for relation in RELATIONS:
        relation_cases = []
        for row in manifest["queues"][relation]:
            entry = entries.get(str(row["case_id"]), {})
            if entry.get("status") == "valid":
                relation_cases.append(entry["replay_case"])
            if len(relation_cases) == TARGETS[relation]:
                break
        if len(relation_cases) != TARGETS[relation]:
            raise ValueError(
                f"{relation} has {len(relation_cases)} valid cases; need exactly {TARGETS[relation]}"
            )
        selected.extend(relation_cases)
    return selected


def _reference_without_case_leaks(
    reference: dict[str, Any], cases: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    filtered = copy.deepcopy(reference)
    removed: list[dict[str, Any]] = []
    while leaks := reference_leakage_details(filtered, cases):
        indexes: dict[str, set[int]] = {"examples": set(), "scenario_few_shots": set()}
        for leak in leaks:
            match = re.match(
                r"reference\.(examples|scenario_few_shots)\[(\d+)\]",
                leak["reference_path"],
            )
            if not match:
                raise RuntimeError(f"cannot remove contamination outside reference examples: {leak}")
            indexes[match.group(1)].add(int(match.group(2)))
        for section, section_indexes in indexes.items():
            rows = filtered.get(section) or []
            for index in sorted(section_indexes, reverse=True):
                removed.append({"section": section, "index": index, "id": rows[index].get("id")})
                del rows[index]
    filtered.pop("reference_sha256", None)
    filtered["reference_sha256"] = _json_sha256(filtered)
    assert_reference_isolated(filtered, cases)
    return filtered, removed


def freeze_benchmark(
    manifest: dict[str, Any], checkpoint_path: Path, output_dir: Path, freeze_path: Path
) -> dict[str, Any]:
    if freeze_path.exists():
        raise ValueError("benchmark is already frozen; refusing to overwrite")
    validate_manifest(manifest)
    checkpoint = _load_checkpoint(checkpoint_path, manifest)
    cases = selected_cases(manifest, checkpoint)
    counts = Counter(row["relationship"] for row in cases)
    if counts != Counter(TARGETS):
        raise ValueError(f"frozen relationship counts are invalid: {dict(counts)}")
    blind, answers = build_blind_packet(
        cases,
        batch_id="benchmark_v3_balanced160_dev",
        reference_sha256="",
    )
    reference, removed_reference_rows = _reference_without_case_leaks(
        manifest["reference"], blind["cases"]
    )
    reference_sha256 = reference["reference_sha256"]
    blind["reference_sha256"] = reference_sha256
    answers["reference_sha256"] = reference_sha256
    answers["blind_sha256"] = _json_sha256(blind)
    assert_reference_isolated(reference, blind["cases"])
    replay = {
        "schema_version": 3,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "split": "dev",
        "manifest_sha256": manifest["manifest_sha256"],
        "cases": cases,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "replay": output_dir / "dev_replay.json",
        "blind": output_dir / "dev_blind.json",
        "answers": output_dir / "dev_answers.json",
    }
    for name, payload in (("replay", replay), ("blind", blind), ("answers", answers)):
        _write_private(paths[name], payload)
    _write_private(output_dir / "judge_reference.json", reference)
    removed_by_section = Counter(row["section"] for row in removed_reference_rows)
    freeze = {
        "schema_version": 3,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest["manifest_sha256"],
        "current_c0_baseline_id": current_c0_baseline_id(),
        "c0_generator_sha256": current_c0_sha256(),
        "reference_sha256": reference_sha256,
        "reference_filtering": {
            "removed_total": len(removed_reference_rows),
            "removed_by_section": dict(removed_by_section),
        },
        "split_summaries": {
            "dev": {
                "cases": TOTAL_TARGET,
                "relationships": dict(counts),
                "unique_chats": len({row["source_chat_id"] for row in cases}),
            }
        },
        "artifact_sha256": {
            "dev": {name: stable_file_sha256(path) for name, path in paths.items()}
        },
        "visibility": {"dev": "open"},
    }
    freeze["freeze_sha256"] = _json_sha256(freeze)
    _write_private(freeze_path, freeze)
    return freeze


def status(manifest: dict[str, Any], checkpoint_path: Path, freeze_path: Path) -> dict[str, Any]:
    checkpoint = _load_checkpoint(checkpoint_path, manifest)
    rows = checkpoint.get("entries", {}).values()
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "current_c0_baseline_id": current_c0_baseline_id(),
        "c0_generator_sha256": current_c0_sha256(),
        "relationships": {
            relation: {
                "valid": sum(
                    row.get("relationship") == relation and row.get("status") == "valid"
                    for row in rows
                ),
                "target": TARGETS[relation],
            }
            for relation in RELATIONS
        },
        "frozen": freeze_path.exists(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced 160-case Judge development set")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--freeze-file", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--exports", type=Path, default=DEFAULT_EXPORTS)
    parser.add_argument("--wiki-dir", type=Path, default=DEFAULT_WIKI)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--volcengine-config", type=Path, default=DEFAULT_VOLCENGINE_CONFIG)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--generate", choices=(*RELATIONS, "all"))
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-checkpoint", type=Path)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if sum(bool(value) for value in (
        args.initialize, args.generate, args.merge_checkpoint, args.freeze
    )) > 1:
        parser.error("choose only one operation")

    if args.initialize:
        if args.manifest.exists():
            raise ValueError("manifest already exists; refusing to replace it")
        all_candidates, eligible, excluded_count, exposed_chat_count = _eligible_candidates(
            args.exports, args.wiki_dir, args.report_dir
        )
        manifest = build_manifest(
            all_candidates,
            eligible,
            few_shot_excluded_count=excluded_count,
            previously_exposed_chat_count=exposed_chat_count,
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
    if args.merge_checkpoint:
        target = _load_checkpoint(args.checkpoint, manifest)
        source = _load_checkpoint(args.merge_checkpoint, manifest)
        merged = merge_checkpoint_data(target, source)
        _write_private(args.checkpoint, merged)
        print(json.dumps(status(manifest, args.checkpoint, args.freeze_file), ensure_ascii=False, indent=2))
        return
    if args.generate:
        relations = RELATIONS if args.generate == "all" else (args.generate,)
        results = [
            generate_relation(
                relation,
                manifest,
                args.checkpoint,
                args.exports,
                args.volcengine_config,
                max_calls=args.max_calls,
                retry_failed=args.retry_failed,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            )
            for relation in relations
        ]
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    if args.freeze:
        print(json.dumps(freeze_benchmark(
            manifest, args.checkpoint, args.manifest.parent, args.freeze_file
        ), ensure_ascii=False, indent=2))
        return
    print(json.dumps(status(manifest, args.checkpoint, args.freeze_file), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
