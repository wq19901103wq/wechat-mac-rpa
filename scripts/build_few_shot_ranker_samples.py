#!/usr/bin/env python3
"""Build leak-free, single-few-shot training samples for the C0 ranker.

The output is deliberately *unlabeled*: a later generation + formal-Judge pass
fills the label for each row.  This builder never calls an LLM.  In particular,
candidate-specific features are all computed locally so online ranking does not
need one model call for every historical example.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reply.few_shot import (  # noqa: E402
    PersonaFewShotRetriever,
    _latest_turn_form,
    _situation_tags,
)


DEFAULT_MANIFEST = ROOT / "data/reports/private/judge_feature_classifier_v1/manifest.json"
DEFAULT_POOL = ROOT / "data/reports/private/situation_few_shot_library_v1/persona_examples.jsonl"
DEFAULT_C0_CONFIG = ROOT / "config/current_c0.json"
DEFAULT_OUTPUT = ROOT / "data/reports/private/few_shot_ranker_samples_v3"
DEFAULT_CANDIDATES_PER_TARGET = 8
PRIVATE_MODE = 0o600
FEATURE_SCHEMA_VERSION = "few-shot-ranker-local-bool-plus-production-trace-v6"
RETRIEVAL_PROTOCOL = "production-retriever-v3-with-score-trace"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _recorded_path(path: Path) -> str:
    """Use portable repository paths when available; retain external inputs verbatim."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _normalized_messages(values: Any) -> tuple[str, ...]:
    if isinstance(values, str):
        values = [values]
    return tuple(
        normalized
        for value in values or []
        if (normalized := re.sub(r"\s+", "", str(value)).strip())
    )


def _contexts_overlap(
    historical_context: Any,
    historical_reply: Any,
    target_context: Any,
) -> bool:
    """Match the existing strict rule: a shared pair or a >=12-char message."""
    historical = (
        _normalized_messages(historical_context)
        + _normalized_messages(historical_reply)
    )
    target = _normalized_messages(target_context)
    historical_pairs = {
        historical[index:index + 2] for index in range(len(historical) - 1)
    }
    target_pairs = {target[index:index + 2] for index in range(len(target) - 1)}
    if historical_pairs & target_pairs:
        return True
    return bool(
        {message for message in historical if len(message) >= 12}
        & {message for message in target if len(message) >= 12}
    )


def _terms(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text.lower())
    chars = [
        char for char in compact
        if char.isalnum() or "\u4e00" <= char <= "\u9fff"
    ]
    return set(chars) | {
        "".join(chars[index:index + 2]) for index in range(max(0, len(chars) - 1))
    }


def _is_short(text: str) -> bool:
    compact = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.lower())
    return 0 < len(compact) <= 6


def _context_messages(target: dict[str, Any]) -> list[dict[str, str]]:
    supplied = target.get("context_messages") or []
    if supplied:
        return [
            {"sender": str(item.get("sender") or "").strip(), "text": str(item.get("text") or "").strip()}
            for item in supplied
            if str(item.get("text") or "").strip()
        ]
    return [
        {"sender": "", "text": str(text).strip()}
        for text in target.get("context") or []
        if str(text).strip()
    ]


def _target_view(target: dict[str, Any]) -> dict[str, Any]:
    messages = _context_messages(target)
    if not messages:
        raise ValueError(f"target {target.get('case_id')} has no context")
    text = "\n".join(message["text"] for message in messages)
    latest = messages[-1]
    others = {
        message["sender"] for message in messages
        if message["sender"] and message["sender"] != "我"
    }
    relationship = str(target.get("relationship") or "")
    if relationship not in {"private", "group"}:
        raise ValueError(f"unsupported target relationship: {relationship!r}")
    return {
        "case_id": str(target["case_id"]),
        "chat_id": str(target.get("chat_id") or ""),
        "relationship": relationship,
        "timestamp": int(target.get("timestamp") or 0),
        "context": [message["text"] for message in messages],
        "context_messages": messages,
        "text": text,
        "terms": _terms(text),
        "tags": set(_situation_tags(text)),
        "latest_form": _latest_turn_form(latest["text"]),
        "last_sender_is_self": latest["sender"] == "我",
        "self_spoke_recently": any(message["sender"] == "我" for message in messages[-3:-1]),
        "has_multiple_other_speakers": len(others) >= 2,
        "last_has_mention": "@" in latest["text"],
        # The target's true reply is intentionally kept only in memory for leak checks.
        "true_reply": _normalized_messages(target.get("replies") or []),
    }


def _candidate_view(row: dict[str, Any]) -> dict[str, Any]:
    context = [str(value).strip() for value in row.get("context") or [] if str(value).strip()]
    reply = [str(value).strip() for value in row.get("reply") or [] if str(value).strip()]
    if not context or not reply:
        raise ValueError(f"few-shot {row.get('id')} is missing context or reply")
    text = "\n".join(context)
    reply_text = "\n".join(reply)
    return {
        "id": str(row["id"]),
        "chat_id": str(row.get("chat_id") or ""),
        "relationship": str(row.get("relationship") or ""),
        "timestamp": int(row.get("timestamp") or 0),
        "context": context,
        "reply": reply,
        "terms": _terms(text),
        "reply_terms": _terms(reply_text),
        "tags": set(str(tag) for tag in row.get("situation_tags") or _situation_tags(text)),
        "latest_form": _latest_turn_form(context[-1]),
        "reply_has_first_person": bool(re.search(r"[我咱]", reply_text)),
        "reply_has_question": bool(re.search(r"[?？]|怎么|咋|什么|啥|为什么|为啥", reply_text)),
        "reply_is_short": _is_short(reply_text),
        "reply_is_multi_bubble": len(reply) > 1,
        "context_is_multi_bubble": len(context) > 1,
    }


def local_boolean_features(target: dict[str, Any], candidate: dict[str, Any]) -> dict[str, bool]:
    """Features usable for every one of ~80k candidates without an LLM call."""
    overlap = len(target["terms"] & candidate["terms"])
    union = len(target["terms"] | candidate["terms"])
    lexical_ratio = overlap / union if union else 0.0
    matching_tags = target["tags"] & candidate["tags"]
    reply_overlap = len(target["terms"] & candidate["reply_terms"])
    features = {
        "context_is_private": target["relationship"] == "private",
        "context_is_group": target["relationship"] == "group",
        "context_last_sender_is_self": target["last_sender_is_self"],
        "context_self_spoke_recently": target["self_spoke_recently"],
        "context_multiple_other_speakers": target["has_multiple_other_speakers"],
        "context_last_has_mention": target["last_has_mention"],
        "context_latest_is_question": target["latest_form"].endswith("question"),
        "candidate_same_chat": candidate["chat_id"] == target["chat_id"],
        "candidate_context_is_multi_bubble": candidate["context_is_multi_bubble"],
        "candidate_reply_is_multi_bubble": candidate["reply_is_multi_bubble"],
        "candidate_reply_is_short": candidate["reply_is_short"],
        "candidate_reply_has_question": candidate["reply_has_question"],
        "candidate_reply_has_first_person": candidate["reply_has_first_person"],
        "candidate_reply_overlap_current_ge2": reply_overlap >= 2,
        "same_latest_turn_form": (
            bool(target["latest_form"])
            and target["latest_form"] == candidate["latest_form"]
        ),
        "lexical_overlap_any": overlap > 0,
        "lexical_overlap_medium": lexical_ratio >= 0.08,
        "lexical_overlap_high": lexical_ratio >= 0.18,
        "situation_overlap_any": bool(matching_tags),
    }
    for tag in (
        "question", "invitation_request", "praise", "complaint", "teasing",
        "share", "arrival_status", "self_deprecation", "calling_presence",
    ):
        features[f"both_situation_{tag}"] = tag in matching_tags
    return features


def _candidate_exclusion_reason(target: dict[str, Any], candidate: dict[str, Any]) -> str | None:
    expected_relation = "group" if target["relationship"] == "group" else "acquaintance"
    if candidate["relationship"] != expected_relation:
        return "relationship_mismatch"
    if candidate["timestamp"] >= target["timestamp"]:
        return "candidate_not_historical"
    if candidate["chat_id"] == target["chat_id"] and _contexts_overlap(
        candidate["context"], candidate["reply"], target["context"],
    ):
        return "same_chat_context_overlap"
    if _normalized_messages(candidate["reply"]) == target["true_reply"]:
        return "direct_answer_leak"
    return None


def select_candidates(
    target: dict[str, Any],
    pool: list[dict[str, Any]],
    *,
    retriever: PersonaFewShotRetriever,
    candidates_per_target: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Return the actual production retriever's safe top-k in its native order."""
    if candidates_per_target < 1:
        raise ValueError("candidates_per_target must be positive")
    exclusions: Counter[str] = Counter()
    eligible_by_id: dict[str, dict[str, Any]] = {}
    exclude_ids: set[str] = set()
    for candidate in pool:
        reason = _candidate_exclusion_reason(target, candidate)
        if reason:
            exclusions[reason] += 1
            exclude_ids.add(candidate["id"])
            continue
        eligible_by_id[candidate["id"]] = candidate

    retrieved = retriever.retrieve(
        query=target["text"],
        chat_name="",
        chat_id=target["chat_id"],
        is_group=target["relationship"] == "group",
        limit=candidates_per_target,
        relationship="group" if target["relationship"] == "group" else "acquaintance",
        interaction_context="",
        exclude_ids=exclude_ids,
        use_situation=True,
        current_context_messages=target["context_messages"],
        include_retrieval_features=True,
    )
    selected: list[dict[str, Any]] = []
    for row in retrieved:
        candidate_id = str(row.get("id") or "")
        candidate = eligible_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(
                "production retriever returned a row outside the leak-safe eligible pool"
            )
        retrieval_features = row.get("retrieval_features")
        if not isinstance(retrieval_features, dict) or not retrieval_features:
            raise ValueError("production retriever did not return its feature trace")
        if not all(
            isinstance(name, str) and isinstance(value, (bool, int, float))
            for name, value in retrieval_features.items()
        ):
            raise ValueError("production retriever trace has a non-numeric feature")
        selected.append({
            "candidate": candidate,
            "features": local_boolean_features(target, candidate),
            "production_retrieval_features": retrieval_features,
        })
    if not selected:
        raise ValueError(
            f"target {target['case_id']} has no eligible historical candidates"
        )
    return selected, exclusions


def _load_retrieval_settings(config_path: Path) -> tuple[dict[str, bool], bytes]:
    """Read only the adopted C0 flags that construct its two retrievers."""
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    features = config.get("features") or {}
    names = (
        "enable_precise_situation_matching",
        "enable_information_gap_response_move",
        "prefer_latest_turn_form",
        "group_prefer_latest_turn_form",
    )
    settings = {name: features.get(name) for name in names}
    if not all(isinstance(value, bool) for value in settings.values()):
        raise ValueError("C0 config has incomplete few-shot retrieval flags")
    return settings, config_bytes


def _retrievers_for_settings(
    pool_path: Path, settings: dict[str, bool],
) -> dict[str, PersonaFewShotRetriever]:
    common = {
        "enable_precise_situation": settings["enable_precise_situation_matching"],
        "enable_information_gap_response_move": settings[
            "enable_information_gap_response_move"
        ],
    }
    return {
        "private": PersonaFewShotRetriever(
            pool_path,
            **common,
            prefer_latest_turn_form=settings["prefer_latest_turn_form"],
        ),
        "group": PersonaFewShotRetriever(
            pool_path,
            **common,
            prefer_latest_turn_form=settings["group_prefer_latest_turn_form"],
        ),
    }


def _public_target(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": target["case_id"],
        "chat_id": target["chat_id"],
        "relationship": target["relationship"],
        "timestamp": target["timestamp"],
        "context": target["context"],
        "context_messages": target["context_messages"],
    }


def _public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        name: candidate[name]
        for name in ("id", "chat_id", "relationship", "timestamp", "context", "reply")
    }


def _normalized_target_row(row: Mapping[str, Any]) -> dict[str, Any]:
    relationship = str(row.get("relationship") or "")
    if relationship == "acquaintance":
        relationship = "private"
    result = dict(row)
    result["case_id"] = str(row.get("case_id") or row.get("id") or "")
    result["relationship"] = relationship
    result["replies"] = row.get("replies") or row.get("reply") or []
    return result


def _select_targets(manifest: dict[str, Any], split: str, limit: int | None) -> list[dict[str, Any]]:
    if manifest.get("cases") is not None:
        rows = [_normalized_target_row(row) for row in manifest.get("cases") or []]
    else:
        rows = [
            _normalized_target_row(row)
            for row in manifest.get("splits", {}).get(split) or []
        ]
    if not rows:
        raise ValueError(f"judge manifest has no {split!r} targets")
    if limit is None or limit >= len(rows):
        return rows
    # Return a balanced stream rather than a fixed prefix.  A target can have no
    # safe historical candidate (for example, it predates the pool), and build()
    # then draws the next target without relaxing the leakage rules.
    private = [row for row in rows if row.get("relationship") == "private"]
    group = [row for row in rows if row.get("relationship") == "group"]
    selected: list[dict[str, Any]] = []
    while private or group:
        if private:
            selected.append(private.pop(0))
        if group:
            selected.append(group.pop(0))
    return selected


def _load_target_ids(path: Path) -> set[str]:
    """Load one target case id per line without copying private text into output."""
    values = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not values:
        raise ValueError(f"target exclusion file is empty: {path}")
    return values


def _load_target_ids_from_samples(path: Path) -> set[str]:
    """Reuse target ids from an earlier private ranker sample artifact."""
    values = {
        str(row.get("target", {}).get("case_id") or "").strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in (json.loads(line),)
    }
    values.discard("")
    if not values:
        raise ValueError(f"sample artifact has no target case ids: {path}")
    return values


def _write_private(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(PRIVATE_MODE)


def _load_existing(output_dir: Path, resume: bool) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    sample_path = output_dir / "samples.jsonl"
    manifest_path = output_dir / "manifest.json"
    if not sample_path.exists() and not manifest_path.exists():
        return [], None
    if not resume:
        raise ValueError(f"{output_dir} already contains samples; use --resume or another output")
    if not sample_path.exists() or not manifest_path.exists():
        raise ValueError("incomplete prior output; refuse to guess whether it is resumable")
    existing = [
        json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return existing, json.loads(manifest_path.read_text(encoding="utf-8"))


def build(
    judge_manifest_path: Path,
    pool_path: Path,
    output_dir: Path,
    *,
    c0_config_path: Path = DEFAULT_C0_CONFIG,
    split: str = "train",
    candidates_per_target: int = DEFAULT_CANDIDATES_PER_TARGET,
    minimum_candidates_per_target: int = 1,
    limit: int | None = None,
    target_limits: dict[str, int] | None = None,
    exclude_target_ids: set[str] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Build or extend unlabeled samples without invoking C0 or the Judge."""
    if not 1 <= minimum_candidates_per_target <= candidates_per_target:
        raise ValueError(
            "minimum_candidates_per_target must be between 1 and candidates_per_target"
        )
    judge_bytes = judge_manifest_path.read_bytes()
    pool_bytes = pool_path.read_bytes()
    retrieval_settings, c0_config_bytes = _load_retrieval_settings(c0_config_path)
    judge_manifest = json.loads(judge_bytes)
    raw_pool = [json.loads(line) for line in pool_bytes.decode("utf-8").splitlines() if line.strip()]
    if len({str(row.get("id") or "") for row in raw_pool}) != len(raw_pool):
        raise ValueError("few-shot pool contains duplicate or empty ids")
    pool_by_relationship: dict[str, list[dict[str, Any]]] = {"acquaintance": [], "group": []}
    for row in raw_pool:
        candidate = _candidate_view(row)
        if candidate["relationship"] in pool_by_relationship:
            pool_by_relationship[candidate["relationship"]].append(candidate)
    if target_limits is not None:
        unknown = set(target_limits) - {"private", "group"}
        if unknown or any(value < 0 for value in target_limits.values()):
            raise ValueError("target_limits must contain non-negative private/group counts only")
        if limit is not None:
            raise ValueError("use either limit or target_limits, not both")
    targets = _select_targets(judge_manifest, split, limit)
    existing, previous_manifest = _load_existing(output_dir, resume)
    exclusions_file_hash = None
    if exclude_target_ids:
        exclusions_file_hash = _sha256(
            "".join(f"{value}\n" for value in sorted(exclude_target_ids)).encode("utf-8")
        )
    source = {
        "judge_manifest": _recorded_path(judge_manifest_path),
        "judge_manifest_sha256": _sha256(judge_bytes),
        "few_shot_pool": _recorded_path(pool_path),
        "few_shot_pool_sha256": _sha256(pool_bytes),
        "c0_config": _recorded_path(c0_config_path),
        "c0_config_sha256": _sha256(c0_config_bytes),
    }
    if exclusions_file_hash:
        source["excluded_target_ids_sha256"] = exclusions_file_hash
    retrieval_contract = {
        "retrieval_protocol": RETRIEVAL_PROTOCOL,
        "retrieval_flags": retrieval_settings,
        "candidate_order": "PersonaFewShotRetriever.retrieve returned order",
        "route_text_source": "last_context_message_offline_approximation",
        "interaction_context_source": "unavailable_in_manifest; empty",
        "max_candidates_per_target": candidates_per_target,
        "minimum_candidates_per_target": minimum_candidates_per_target,
    }
    if previous_manifest:
        previous_protocol = previous_manifest.get("protocol") or {}
        if (
            previous_manifest.get("sources") != source
            or any(previous_protocol.get(name) != value for name, value in retrieval_contract.items())
        ):
            raise ValueError("cannot resume samples built with a different retrieval protocol")
    existing_targets = {
        str(row.get("target", {}).get("case_id") or ""): str(
            row.get("target", {}).get("relationship") or ""
        )
        for row in existing
    }
    existing_ids = set(existing_targets)
    existing_relationship_counts = Counter(existing_targets.values())
    samples = list(existing)
    exclusions: Counter[str] = Counter()
    retrievers = _retrievers_for_settings(pool_path, retrieval_settings)
    for row in targets:
        if limit is not None and len(existing_ids) >= limit:
            break
        if target_limits and all(
            existing_relationship_counts[relationship] >= requested
            for relationship, requested in target_limits.items()
        ):
            break
        target = _target_view(row)
        if target["case_id"] in (exclude_target_ids or set()):
            exclusions["target_explicitly_excluded"] += 1
            continue
        if target_limits and existing_relationship_counts[target["relationship"]] >= target_limits.get(
            target["relationship"], 0,
        ):
            continue
        if target["case_id"] in existing_ids:
            continue
        candidate_pool = pool_by_relationship[
            "group" if target["relationship"] == "group" else "acquaintance"
        ]
        try:
            selected, target_exclusions = select_candidates(
                target, candidate_pool, candidates_per_target=candidates_per_target,
                retriever=retrievers[target["relationship"]],
            )
        except ValueError as exc:
            if str(exc) != f"target {target['case_id']} has no eligible historical candidates":
                raise
            exclusions["target_without_safe_historical_candidate"] += 1
            continue
        if len(selected) < minimum_candidates_per_target:
            exclusions["target_with_insufficient_safe_historical_candidates"] += 1
            continue
        exclusions.update(target_exclusions)
        existing_ids.add(target["case_id"])
        existing_relationship_counts[target["relationship"]] += 1
        for rank, selected_row in enumerate(selected, start=1):
            candidate = selected_row["candidate"]
            sample_id = _sha256(f"{target['case_id']}:{candidate['id']}".encode("utf-8"))[:20]
            samples.append({
                "sample_id": sample_id,
                "target": _public_target(target),
                "candidate": _public_candidate(candidate),
                "candidate_rank": rank,
                "local_features": selected_row["features"],
                "production_retrieval_features": selected_row["production_retrieval_features"],
                "generation": {
                    "few_shot_count": 1,
                    "static_few_shot_count": 0,
                    "injected_candidate_id": candidate["id"],
                },
                "label": None,
                "label_definition": "pending_formal_judge: 1 when the C0 reply is not identified as AI; 0 otherwise",
            })
    target_ids = {str(row["target"]["case_id"]) for row in samples}
    target_sample_counts = Counter(str(row["target"]["case_id"]) for row in samples)
    if len({str(row["sample_id"]) for row in samples}) != len(samples):
        raise ValueError("sample id collision or duplicate resume rows")
    sample_content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in samples)
    manifest = {
        "schema_version": 2,
        "kind": "c0_single_few_shot_ranker_samples",
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidates_built_unlabeled",
        "sources": source,
        "protocol": {
            "target_split": split,
            "target_case_count": len(target_ids),
            "requested_target_limits": target_limits,
            **retrieval_contract,
            "minimum_candidates_per_target": minimum_candidates_per_target,
            "single_few_shot_only": True,
            "static_few_shot_count": 0,
            "target_human_reply_written": False,
            "same_chat_nonoverlap_allowed": True,
            "historical_candidate_rule": "candidate timestamp must be earlier than the target timestamp",
            "overlap_rule": "same-chat shared consecutive pair or shared message with >=12 normalized chars",
            "direct_answer_leak_rule": "exclude candidate when its full human reply equals the target human reply",
            "candidate_llm_calls": 0,
            "online_llm_contract": "one optional context-understanding call per target; never per candidate",
            "feature_schema": FEATURE_SCHEMA_VERSION,
        },
        "counts": {
            "samples": len(samples),
            "targets": len(target_ids),
            "relationship": dict(sorted(Counter(row["target"]["relationship"] for row in samples).items())),
            "samples_per_target": dict(sorted(Counter(
                str(count) for count in target_sample_counts.values()
            ).items())),
            "exclusions_for_new_targets": dict(sorted(exclusions.items())),
        },
        "files": {"samples": "samples.jsonl", "samples_sha256": _sha256(sample_content.encode("utf-8"))},
    }
    manifest["manifest_sha256"] = _sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    _write_private(output_dir / "samples.jsonl", sample_content)
    _write_private(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--c0-config", type=Path, default=DEFAULT_C0_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--candidates-per-target", type=int, default=DEFAULT_CANDIDATES_PER_TARGET)
    parser.add_argument("--minimum-candidates-per-target", type=int, default=1)
    parser.add_argument("--limit", type=int, help="balanced dry-run target count")
    parser.add_argument("--private-target-limit", type=int, help="exact private target count")
    parser.add_argument("--group-target-limit", type=int, help="exact group target count")
    exclusions_group = parser.add_mutually_exclusive_group()
    exclusions_group.add_argument(
        "--exclude-target-ids",
        type=Path,
        help="private file with one target case id per line; excludes ranking-training targets",
    )
    exclusions_group.add_argument(
        "--exclude-target-ids-from-samples",
        type=Path,
        action="append",
        help="private earlier samples.jsonl whose target ids must be excluded",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.private_target_limit is None) != (args.group_target_limit is None):
        parser.error("--private-target-limit and --group-target-limit must be used together")
    target_limits = None
    if args.private_target_limit is not None:
        target_limits = {
            "private": args.private_target_limit,
            "group": args.group_target_limit,
        }
    excluded_target_ids = None
    if args.exclude_target_ids:
        excluded_target_ids = _load_target_ids(args.exclude_target_ids)
    elif args.exclude_target_ids_from_samples:
        excluded_target_ids = set().union(*(
            _load_target_ids_from_samples(path)
            for path in args.exclude_target_ids_from_samples
        ))
    print(json.dumps(build(
        args.judge_manifest, args.pool, args.output, c0_config_path=args.c0_config, split=args.split,
        candidates_per_target=args.candidates_per_target,
        minimum_candidates_per_target=args.minimum_candidates_per_target,
        limit=args.limit, target_limits=target_limits,
        exclude_target_ids=excluded_target_ids, resume=args.resume,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
