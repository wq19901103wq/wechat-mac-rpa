#!/usr/bin/env python3
"""Build and inspect an offline human-style action index; never call a chat model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
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
    _candidate_id,
    _current_response_anchor,
    _message_text,
    _serialize_message,
    _timestamp_seconds,
)
from src.benchmarks.reply_system_replay import load_replay_cases  # noqa: E402
from src.memory.history_search import _BGEEncoder, _model_path, _try_import_encoder_deps  # noqa: E402
from src.reply.few_shot import _chat_id  # noqa: E402


ACTION_DESCRIPTIONS = {
    "playful_counter": "被点名、被调侃、被损或被夸张评价时，用反转、拆台或荒诞类比接梗",
    "arrival_emotion": "对方说到了、快到家或正在路上，表达惊讶、羡慕、关心并自然追问",
    "brief_presence": "对方催人、叫人或问人在哪，用很短的状态回应",
    "light_refusal": "对方邀请、要求一起做事或劝参与时，简短自然地拒绝",
    "shared_feeling": "对方表达羡慕、失落或工作情绪时，先接住感受再顺势回应",
    "practical_discussion": "围绕股票、工作或现实事项继续追问、澄清或补充自己的情况",
}

# These IDs were directly reviewed against raw pre-cutoff context and human replies.
# The manifest stores no chat text; the private preview materializes it from local exports.
CURATED_RAW_PROFILES: dict[str, dict[str, Any]] = {
    "human_37e39a728895": {"action": "playful_counter", "response_move": "absurd_analogy_denial"},
    "human_59931cfd7788": {"action": "brief_presence", "response_move": "brief_status"},
    "human_7df9b6c48e6f": {"action": "light_refusal", "response_move": "polite_short_refusal"},
    "human_c4e53665cc0c": {"action": "shared_feeling", "response_move": "empathy_then_next_step"},
    "human_7be27a193311": {"action": "practical_discussion", "response_move": "clarify_then_self_disclose", "fact_risk": True},
    "human_3e346d275db9": {"action": "practical_discussion", "response_move": "counter_question_then_explain", "fact_risk": True},
    "human_3ec3b101bc90": {"action": "arrival_emotion", "response_move": "surprise_then_affection"},
    "human_d11c8134a379": {"action": "arrival_emotion", "response_move": "surprise_then_follow_up"},
}


def _normalized(text: str) -> str:
    return "".join(char.lower() for char in text if char.isalnum() or "\u4e00" <= char <= "\u9fff")


def _stable_profile_id(row: dict[str, Any]) -> str:
    value = "\n".join([*(row.get("context") or []), "=>", *(row.get("reply") or [])])
    return "profile_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _fact_sensitive(text: str) -> bool:
    compact = _normalized(text)
    fact_markers = ("请回答", "是否", "是不是", "有没有", "领结婚证", "属什么")
    return any(marker in compact for marker in fact_markers)


def _commitment_risk(replies: list[str]) -> bool:
    compact = _normalized(" ".join(replies))
    return any(marker in compact for marker in ("发红包", "转账给你", "打钱给你", "请你喝"))


def build_curated_rows(
    raw_candidates: list[Candidate],
    profiled_path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    raw_by_id = {_candidate_id(candidate): candidate for candidate in raw_candidates}
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for candidate_id, profile in CURATED_RAW_PROFILES.items():
        candidate = raw_by_id.get(candidate_id)
        if candidate is None:
            missing.append(candidate_id)
            continue
        rows.append({
            "id": candidate_id,
            "relationship": candidate.relation,
            "chat_id": candidate.chat_id,
            "context": candidate.context,
            "reply": candidate.replies,
            "timestamp": candidate.timestamp,
            "source_provenance": candidate.source_provenance,
            "source_kind": "raw_pre_cutoff_direct_review",
            **profile,
            "commitment_risk": _commitment_risk(candidate.replies),
        })

    for line in profiled_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        profile = source.get("semantic_profile") or {}
        if source.get("source_provenance") != "before_automation_cutoff" or not profile:
            continue
        rows.append({
            "id": _stable_profile_id(source),
            "relationship": source.get("relationship") or "group",
            "chat_id": source.get("chat_id") or "",
            "context": source["context"],
            "reply": source["reply"],
            "timestamp": source.get("timestamp") or 0,
            "source_provenance": source["source_provenance"],
            "source_kind": "pre_cutoff_human_profile",
            "action": "playful_counter",
            "response_move": profile.get("response_move") or "playful_counter",
            "tone": profile.get("tone") or "playful",
            "fact_risk": False,
            "commitment_risk": _commitment_risk(source["reply"]),
        })
    return rows, missing


def _encode(encoder: _BGEEncoder, texts: list[str]) -> np.ndarray:
    batches = [encoder.encode(texts[start:start + 128]) for start in range(0, len(texts), 128)]
    return np.concatenate(batches, axis=0) if batches else np.zeros((0, 512), dtype=np.float32)


def select_action(
    query_embedding: np.ndarray,
    action_embeddings: dict[str, np.ndarray],
    *,
    minimum: float = 0.48,
    minimum_margin: float = 0.025,
) -> tuple[str | None, dict[str, float]]:
    scores = {
        action: float(np.max(vectors @ query_embedding))
        for action, vectors in action_embeddings.items()
        if len(vectors)
    }
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if not ranked or ranked[0][1] < minimum:
        return None, scores
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < minimum_margin:
        return None, scores
    return ranked[0][0], scores


def rank_action_rows(
    query_embedding: np.ndarray,
    selected_action: str | None,
    rows: list[dict[str, Any]],
    row_embeddings: np.ndarray,
    *,
    chat_id: str,
    relationship: str,
    query_text: str,
    recent_text: str,
    limit: int,
) -> list[dict[str, Any]]:
    if selected_action is None:
        return []
    fact_sensitive = _fact_sensitive(query_text)
    normalized_recent = _normalized(recent_text)
    ranked = []
    for index, row in enumerate(rows):
        if row["action"] != selected_action or row.get("commitment_risk"):
            continue
        if fact_sensitive and row.get("fact_risk"):
            continue
        reply_text = _normalized(" ".join(row["reply"]))
        if reply_text and reply_text in normalized_recent:
            continue
        semantic = float(row_embeddings[index] @ query_embedding)
        same_chat = bool(chat_id and row.get("chat_id") == chat_id)
        same_relationship = row.get("relationship") == relationship
        score = semantic + (0.08 if same_chat else 0.0) + (0.02 if same_relationship else 0.0)
        ranked.append((score, semantic, row))
    ranked.sort(key=lambda item: (-item[0], item[2]["id"]))
    return [
        {
            **row,
            "semantic_similarity": round(semantic, 6),
            "retrieval_score": round(score, 6),
            "same_chat": bool(chat_id and row.get("chat_id") == chat_id),
        }
        for score, semantic, row in ranked[: max(0, limit)]
    ]


def _write_private(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(PRIVATE_MODE)
    temporary.replace(path)
    path.chmod(PRIVATE_MODE)


def main() -> None:
    parser = argparse.ArgumentParser(description="离线对话动作与真人风格索引预览")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data/cases.db")
    parser.add_argument("--scenarios", type=Path, default=PROJECT_ROOT / "data/reports/private/reply_system_scenarios.json")
    parser.add_argument("--selection", type=Path, default=PROJECT_ROOT / "data/reports/private/reply_quality_gold_cases.json")
    parser.add_argument("--exports", type=Path, default=PROJECT_ROOT / "data/exports/b")
    parser.add_argument("--wiki-dir", type=Path, default=PROJECT_ROOT / "data/memory/wiki/users")
    parser.add_argument("--profiled", type=Path, default=PROJECT_ROOT / "data/few_shot_v5/verified_human_humor.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/reports/private/offline_style_action_index_preview.json")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    generator_sha = hashlib.sha256((PROJECT_ROOT / "src/reply/generator.py").read_bytes()).hexdigest()
    if generator_sha != PROTECTED_C0_SHA256:
        raise ValueError("current generator no longer matches protected C0")
    source_fingerprint, cases = load_replay_cases(args.db, args.scenarios, args.selection)
    raw_candidates = extract_candidates(args.exports, args.wiki_dir, human_before=CUTOFF)
    raw_candidates = [row for row in raw_candidates if 0 < _timestamp_seconds(row.timestamp) < CUTOFF]
    rows, missing = build_curated_rows(raw_candidates, args.profiled)
    if missing:
        raise ValueError(f"curated raw examples not found: {missing}")

    backend = _try_import_encoder_deps()
    if not backend or not _model_path().exists():
        raise RuntimeError("local BGE encoder is unavailable")
    encoder = _BGEEncoder(_model_path(), backend)
    row_texts = ["对话输入：" + "\n".join(row["context"]) for row in rows]
    row_embeddings = _encode(encoder, row_texts)
    action_embeddings: dict[str, np.ndarray] = {}
    for action, description in ACTION_DESCRIPTIONS.items():
        prototypes = ["互动动作：" + description]
        prototypes.extend(row_texts[index] for index, row in enumerate(rows) if row["action"] == action)
        action_embeddings[action] = _encode(encoder, prototypes)

    output_cases = []
    for case in cases:
        chat_name = case["all_messages"][0].chat_name if case["all_messages"] else ""
        current_chat_id = _chat_id(chat_name) if chat_name else ""
        target_messages = [str(message.text or "") for message in case["unreplied"]]
        target = _current_response_anchor(target_messages)
        full_target = "\n".join(part for part in (_message_text(message) for message in case["unreplied"]) if part)
        target_ids = {message.local_id for message in case["unreplied"] if message.local_id is not None}
        prior = [message for message in case["all_messages"] if message.local_id not in target_ids]
        recent_context = [_message_text(message) for message in prior[-6:]]
        query_text = "\n".join(part for part in ("最近上下文：" + "\n".join(recent_context), "当前消息：" + full_target) if part)
        query_embedding = encoder.encode([query_text])[0]
        action, action_scores = select_action(query_embedding, action_embeddings)
        relationship = "group" if case["is_group"] else "acquaintance"
        selected = rank_action_rows(
            query_embedding,
            action,
            rows,
            row_embeddings,
            chat_id=current_chat_id,
            relationship=relationship,
            query_text=target,
            recent_text="\n".join(recent_context + [full_target]),
            limit=args.top_k,
        )
        output_cases.append({
            "case_id": case["case_id"],
            "context_original": [_serialize_message(message) for message in case["all_messages"]],
            "target_original": [_serialize_message(message) for message in case["unreplied"]],
            "retrieval_query": {"target_anchor": target, "full_target": full_target, "recent_context": recent_context},
            "selected_action": action,
            "action_scores": {key: round(value, 6) for key, value in sorted(action_scores.items())},
            "top_candidates": selected,
        })

    packet = {
        "schema_version": 1,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "purpose": "offline action/style matching only; no generation and no production integration",
        "source_fingerprint": source_fingerprint,
        "generator_sha256": generator_sha,
        "index_row_count": len(rows),
        "action_counts": dict(sorted(Counter(row["action"] for row in rows).items())),
        "excluded_commitment_risk_count": sum(bool(row.get("commitment_risk")) for row in rows),
        "model_generation_calls": 0,
        "bot_constructed": False,
        "cases": output_cases,
    }
    _write_private(args.output, packet)
    print(json.dumps({
        "output": str(args.output),
        "case_count": len(output_cases),
        "index_row_count": len(rows),
        "cases_with_candidate": sum(bool(case["top_candidates"]) for case in output_cases),
        "selected_action_counts": dict(sorted(Counter(case["selected_action"] or "none" for case in output_cases).items())),
        "model_generation_calls": 0,
        "bot_constructed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
