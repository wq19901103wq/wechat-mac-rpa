#!/usr/bin/env python3
"""High-precision preview of pre-cutoff human replies from the same chat."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_persona_few_shots import Candidate, extract_candidates  # noqa: E402
from scripts.preview_full_history_style_retrieval import (  # noqa: E402
    CUTOFF,
    PRIVATE_MODE,
    PROTECTED_C0_SHA256,
    _bm25_scores,
    _candidate_id,
    _message_text,
    _serialize_message,
    _timestamp_seconds,
    _unsafe_fact_example,
)
from src.benchmarks.reply_system_replay import load_replay_cases  # noqa: E402
from src.memory.history_search import _BGEEncoder, _model_path, _try_import_encoder_deps  # noqa: E402
from src.reply.few_shot import _chat_id  # noqa: E402

MIN_ANCHOR_SIMILARITY = 0.62
MIN_CONTEXT_SIMILARITY = 0.60
MIN_JOINT_SCORE = 0.62
MIN_MARGIN = 0.04
MIN_RESPONSE_ALIGNMENT = 0.48

_NON_TARGETS = {"?", "？", "嗯", "哦", "啊", "对", "行", "好"}
_COMMITMENT_PATTERNS = (
    r"(?:给你|帮你|替你).{0,6}(?:转账|打钱|发红包)",
    r"(?:我|下次).{0,4}(?:请你|请大家|请群友)",
)


def _clean_text(text: str) -> str:
    return str(text or "").strip()


def _normalized(text: str) -> str:
    return "".join(char.lower() for char in text if char.isalnum() or "\u4e00" <= char <= "\u9fff")


def response_anchor(messages: list[str]) -> str:
    """The final substantive incoming message is the primary reply target."""
    cleaned = [_clean_text(message) for message in messages if _clean_text(message)]
    if not cleaned:
        return ""
    last = cleaned[-1]
    if last in _NON_TARGETS or re.fullmatch(r"https?://\S+", last, flags=re.IGNORECASE):
        return ""
    return last


def response_window(messages: list[str]) -> str:
    """Keep one lead-in for topic validation without merging it into the target."""
    cleaned = [_clean_text(message) for message in messages if _clean_text(message)]
    return "\n".join(cleaned[-2:])


def candidate_anchor(candidate: Candidate) -> str:
    return response_anchor(candidate.context)


def candidate_window(candidate: Candidate) -> str:
    return response_window(candidate.context)


def _character_ngrams(text: str) -> set[str]:
    compact = _normalized(text)
    return {
        compact[index:index + size]
        for size in (2, 3)
        for index in range(max(0, len(compact) - size + 1))
    }


def has_topic_evidence(current_window: str, historical_window: str) -> bool:
    return bool(_character_ngrams(current_window) & _character_ngrams(historical_window))


def commitment_risk(replies: list[str]) -> bool:
    text = " ".join(replies)
    return any(re.search(pattern, text) for pattern in _COMMITMENT_PATTERNS)


def question_intent(text: str) -> bool:
    compact = _normalized(text)
    return bool(re.search(r"[?？]", text) or any(marker in compact for marker in ("吗", "么", "什么", "啥", "怎么", "为什么", "有没有", "是否", "哪")))


def directed_to_other(text: str) -> bool:
    return bool(re.search(r"@\S+", text))


def accept_match(
    *,
    anchor_similarity: float,
    context_similarity: float,
    joint_score: float,
    margin: float,
    topic_evidence: bool,
    response_alignment: float,
    question_intent_compatible: bool,
    target_directed_to_other: bool,
) -> tuple[bool, list[str]]:
    failures = []
    if anchor_similarity < MIN_ANCHOR_SIMILARITY:
        failures.append("reply_target_similarity_below_threshold")
    if context_similarity < MIN_CONTEXT_SIMILARITY:
        failures.append("short_context_similarity_below_threshold")
    if joint_score < MIN_JOINT_SCORE:
        failures.append("joint_score_below_threshold")
    if margin < MIN_MARGIN:
        failures.append("leader_margin_below_threshold")
    if not topic_evidence:
        failures.append("no_shared_topic_evidence")
    if response_alignment < MIN_RESPONSE_ALIGNMENT:
        failures.append("historical_reply_not_aligned_with_incoming")
    if not question_intent_compatible:
        failures.append("question_vs_statement_intent_mismatch")
    if target_directed_to_other:
        failures.append("target_explicitly_directed_to_other_person")
    return not failures, failures


def _encode(encoder: _BGEEncoder, texts: list[str]) -> np.ndarray:
    batches = [encoder.encode(texts[start:start + 256]) for start in range(0, len(texts), 256)]
    return np.concatenate(batches, axis=0) if batches else np.zeros((0, 512), dtype=np.float32)


def _write_private(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(PRIVATE_MODE)
    temporary.replace(path)
    path.chmod(PRIVATE_MODE)


def main() -> None:
    parser = argparse.ArgumentParser(description="严格同聊天真人原话匹配预览，不调用生成模型")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data/cases.db")
    parser.add_argument("--scenarios", type=Path, default=PROJECT_ROOT / "data/reports/private/reply_system_scenarios.json")
    parser.add_argument("--selection", type=Path, default=PROJECT_ROOT / "data/reports/private/reply_quality_gold_cases.json")
    parser.add_argument("--exports", type=Path, default=PROJECT_ROOT / "data/exports/b")
    parser.add_argument("--wiki-dir", type=Path, default=PROJECT_ROOT / "data/memory/wiki/users")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/reports/private/strict_same_chat_style_matching_preview.json")
    parser.add_argument("--inspect-top-k", type=int, default=3)
    args = parser.parse_args()

    generator_sha = hashlib.sha256((PROJECT_ROOT / "src/reply/generator.py").read_bytes()).hexdigest()
    if generator_sha != PROTECTED_C0_SHA256:
        raise ValueError("current generator no longer matches protected C0")

    backend = _try_import_encoder_deps()
    if not backend or not _model_path().exists():
        raise RuntimeError("local BGE encoder is unavailable")
    encoder = _BGEEncoder(_model_path(), backend)

    source_fingerprint, cases = load_replay_cases(args.db, args.scenarios, args.selection)
    relevant_chat_ids = {
        _chat_id(case["all_messages"][0].chat_name)
        for case in cases
        if case["all_messages"] and case["all_messages"][0].chat_name
    }
    candidates = extract_candidates(args.exports, args.wiki_dir, human_before=CUTOFF)
    candidates = [
        candidate for candidate in candidates
        if candidate.chat_id in relevant_chat_ids
        and 0 < _timestamp_seconds(candidate.timestamp) < CUTOFF
        and candidate_anchor(candidate)
        and not commitment_risk(candidate.replies)
    ]
    eligible_pre_cutoff_candidate_count = len(candidates)
    raw_by_chat: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        raw_by_chat[candidate.chat_id].append(candidate)
    raw_chat_counts = {chat_id: len(rows) for chat_id, rows in raw_by_chat.items()}

    selected_candidate_ids: set[str] = set()
    for case in cases:
        chat_name = case["all_messages"][0].chat_name if case["all_messages"] else ""
        chat_id = _chat_id(chat_name) if chat_name else ""
        target_messages = [str(message.text or "") for message in case["unreplied"]]
        target_window = response_window(target_messages) if response_anchor(target_messages) else ""
        pool = raw_by_chat.get(chat_id, [])
        if not target_window or not pool:
            continue
        scores = _bm25_scores(target_window, [candidate_window(row) for row in pool])
        indices = sorted(range(len(pool)), key=lambda index: (-float(scores[index]), _candidate_id(pool[index])))
        selected_candidate_ids.update(
            _candidate_id(pool[index]) for index in indices[:256] if scores[index] > 0
        )

    candidates = [candidate for candidate in candidates if _candidate_id(candidate) in selected_candidate_ids]
    by_chat: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        by_chat[candidate.chat_id].append(index)

    anchor_vectors = _encode(encoder, ["需要回应的消息：" + candidate_anchor(row) for row in candidates])
    window_vectors = _encode(encoder, ["相邻话题：" + candidate_window(row) for row in candidates])
    reply_vectors = _encode(encoder, ["真人回应：" + " ".join(row.replies) for row in candidates])

    output_cases = []
    for case in cases:
        chat_name = case["all_messages"][0].chat_name if case["all_messages"] else ""
        chat_id = _chat_id(chat_name) if chat_name else ""
        target_messages = [str(message.text or "") for message in case["unreplied"]]
        target_anchor = response_anchor(target_messages)
        target_window = response_window(target_messages) if target_anchor else ""
        recent_context = [_message_text(message) for message in case["all_messages"][-8:]]
        recent_compact = _normalized("\n".join(recent_context))
        pool_indices = by_chat.get(chat_id, [])
        ranked: list[dict[str, Any]] = []

        if target_anchor and pool_indices:
            query_anchor = encoder.encode(["需要回应的消息：" + target_anchor])[0]
            query_window = encoder.encode(["相邻话题：" + target_window])[0]
            for index in pool_indices:
                candidate = candidates[index]
                if _unsafe_fact_example(target_anchor, candidate):
                    continue
                human_reply_compact = _normalized(" ".join(candidate.replies))
                if human_reply_compact and human_reply_compact in recent_compact:
                    continue
                anchor_similarity = float(anchor_vectors[index] @ query_anchor)
                context_similarity = float(window_vectors[index] @ query_window)
                response_alignment = float(anchor_vectors[index] @ reply_vectors[index])
                joint_score = anchor_similarity * 0.65 + context_similarity * 0.35
                ranked.append({
                    "candidate_id": _candidate_id(candidate),
                    "reply_target_similarity": round(anchor_similarity, 6),
                    "short_context_similarity": round(context_similarity, 6),
                    "joint_score": round(joint_score, 6),
                    "historical_response_alignment": round(response_alignment, 6),
                    "shared_topic_evidence": has_topic_evidence(target_window, candidate_window(candidate)),
                    "question_intent_compatible": question_intent(target_anchor) == question_intent(candidate_anchor(candidate)),
                    "target_directed_to_other": directed_to_other(target_anchor),
                    "historical_context": candidate.context,
                    "historical_reply_target": candidate_anchor(candidate),
                    "human_reply": candidate.replies,
                    "timestamp": candidate.timestamp,
                    "source_provenance": candidate.source_provenance,
                })
            ranked.sort(key=lambda row: (-row["joint_score"], row["candidate_id"]))

        leader = ranked[0] if ranked else None
        runner_up_score = ranked[1]["joint_score"] if len(ranked) > 1 else 0.0
        margin = (leader["joint_score"] - runner_up_score) if leader else 0.0
        if leader:
            accepted, reasons = accept_match(
                anchor_similarity=leader["reply_target_similarity"],
                context_similarity=leader["short_context_similarity"],
                joint_score=leader["joint_score"],
                margin=margin,
                topic_evidence=leader["shared_topic_evidence"],
                response_alignment=leader["historical_response_alignment"],
                question_intent_compatible=leader["question_intent_compatible"],
                target_directed_to_other=leader["target_directed_to_other"],
            )
        else:
            accepted, reasons = False, ["no_eligible_same_chat_candidate"]

        output_cases.append({
            "case_id": case["case_id"],
            "chat_id": chat_id,
            "context_original": [_serialize_message(message) for message in case["all_messages"]],
            "target_original": [_serialize_message(message) for message in case["unreplied"]],
            "matching_query": {"reply_target": target_anchor, "short_context": target_window},
            "same_chat_candidate_count": raw_chat_counts.get(chat_id, 0),
            "lexical_compute_prefilter_count": len(pool_indices),
            "accepted": accepted,
            "rejection_reasons": reasons,
            "leader_margin": round(margin, 6),
            "matched_human_example": leader if accepted else None,
            "inspection_candidates": ranked[: max(0, args.inspect_top_k)],
        })

    packet = {
        "schema_version": 1,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "purpose": "high-precision matching inspection only; no reply generation and no production integration",
        "cutoff_epoch": CUTOFF,
        "source_fingerprint": source_fingerprint,
        "generator_sha256": generator_sha,
        "thresholds": {
            "minimum_reply_target_similarity": MIN_ANCHOR_SIMILARITY,
            "minimum_short_context_similarity": MIN_CONTEXT_SIMILARITY,
            "minimum_joint_score": MIN_JOINT_SCORE,
            "minimum_leader_margin": MIN_MARGIN,
            "minimum_historical_response_alignment": MIN_RESPONSE_ALIGNMENT,
            "shared_topic_evidence_required": True,
            "question_statement_intent_must_match": True,
            "explicit_other_person_mention_rejected": True,
        },
        "matching_policy": "strict same chat; final substantive incoming message is the reply target; one lead-in is context only; both similarities, leader margin, topic evidence, question/statement intent, addressee, and historical incoming/reply alignment must pass; uncertain means no match",
        "eligible_pre_cutoff_candidate_count": eligible_pre_cutoff_candidate_count,
        "compute_prefilter_candidate_count": len(candidates),
        "compute_prefilter": "same-chat character BM25 union for the fixed scenarios; no conversation-move keyword classifier",
        "model_generation_calls": 0,
        "bot_constructed": False,
        "cases": output_cases,
    }
    _write_private(args.output, packet)
    print(json.dumps({
        "output": str(args.output),
        "case_count": len(output_cases),
        "accepted_match_count": sum(case["accepted"] for case in output_cases),
        "model_generation_calls": 0,
        "bot_constructed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
