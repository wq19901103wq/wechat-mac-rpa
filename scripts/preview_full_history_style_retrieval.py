#!/usr/bin/env python3
"""Preview same-chat pre-cutoff human-style retrieval without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_persona_few_shots import Candidate, extract_candidates  # noqa: E402
from src.benchmarks.reply_system_replay import load_replay_cases  # noqa: E402
from src.memory.history_search import _BGEEncoder, _model_path, _try_import_encoder_deps  # noqa: E402
from src.reply.few_shot import _chat_id  # noqa: E402

CUTOFF = 1774972800.0
PRIVATE_MODE = 0o600
PROTECTED_C0_SHA256 = "625b47803fff55a2e18914c5e734fcb2f01ae7c16e5180fdebdc817badd2656b"

_MOVE_TERMS = {
    "arrival": ("到了", "到家", "马上到", "出发", "坐地铁"),
    "work": ("加班", "离职", "上班", "下班", "同事", "领导", "日报"),
    "market": ("股票", "基金", "开盘", "市盈率", "财报", "踏空", "加仓", "抄底", "涨", "跌"),
    "affection": ("谢谢", "开心", "宝贝", "心理医生", "感动", "辛苦"),
    "technical": ("chatgpt", "助手", "任务", "简历", "存储空间", "vpn", "软件", "bug", "ai"),
    "banter": ("首富", "有钱", "欠", "猪", "垃圾", "装逼", "污蔑", "撇清", "复读", "发财", "房贷", "分手", "财产"),
    "social_request": ("奶茶", "请客", "请群友", "红包", "转账"),
    "system_state": ("没关", "关了", "没开", "开着", "关闭", "打开"),
}
_STRICT_MOVES = {"arrival", "affection", "social_request", "system_state"}
_FACT_SENSITIVE_TERMS = ("请回答", "领结婚证", "你属什么", "是否", "是不是", "有没有")
_GENERIC_FACT_REPLIES = {"是", "不是", "有", "没有", "领了", "没领", "对", "不对", "嗯", "没"}


def _normalized(text: str) -> str:
    return "".join(char.lower() for char in text if char.isalnum() or "\u4e00" <= char <= "\u9fff")


def _terms(text: str) -> list[str]:
    normalized = _normalized(text)
    terms = [normalized[index:index + size] for size in (2, 3) for index in range(max(0, len(normalized) - size + 1))]
    terms.extend(re.findall(r"[a-z0-9]{2,}", text.lower()))
    return terms


def _bm25_scores(query: str, documents: list[str]) -> np.ndarray:
    """Small dependency-free BM25, with IDF learned inside one chat."""
    if not documents:
        return np.zeros(0, dtype=np.float32)
    query_terms = Counter(_terms(query))
    document_terms = [Counter(_terms(document)) for document in documents]
    document_frequency = Counter(term for counts in document_terms for term in counts)
    average_length = sum(sum(counts.values()) for counts in document_terms) / max(1, len(document_terms))
    scores = np.zeros(len(documents), dtype=np.float32)
    for index, counts in enumerate(document_terms):
        length = sum(counts.values())
        for term, query_count in query_terms.items():
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            idf = math.log(1.0 + (len(documents) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            denominator = frequency + 1.2 * (0.25 + 0.75 * length / max(1.0, average_length))
            scores[index] += query_count * idf * (frequency * 2.2 / denominator)
    return scores


def _response_anchor(candidate: Candidate) -> str:
    """Use the incoming turn most likely answered by the human reply."""
    messages = [message.strip() for message in candidate.context if message.strip()]
    if not messages:
        return ""
    if len(messages) > 1 and len(_normalized(messages[-1])) <= 4:
        return "\n".join(messages[-2:])
    return messages[-1]


def _current_response_anchor(messages: list[str]) -> str:
    """Focus retrieval on the latest current text while keeping a short lead-in."""
    messages = [message.strip() for message in messages if message.strip()]
    if not messages:
        return ""
    last = messages[-1]
    if len(messages) > 1 and (len(_normalized(last)) <= 10 or last.startswith(("http://", "https://"))):
        return "\n".join(messages[-2:])
    return last


def _conversation_moves(text: str) -> set[str]:
    lowered = text.lower()
    return {move for move, terms in _MOVE_TERMS.items() if any(term in lowered for term in terms)}


def _conversation_move_compatible(query: str, historical_incoming: str) -> bool:
    query_moves = _conversation_moves(query)
    strict_moves = query_moves & _STRICT_MOVES
    if strict_moves:
        return bool(strict_moves & _conversation_moves(historical_incoming))
    return not query_moves or bool(query_moves & _conversation_moves(historical_incoming))


def _generic_fact_answer(replies: list[str]) -> bool:
    normalized_replies = [_normalized(reply) for reply in replies if _normalized(reply)]
    return bool(normalized_replies) and all(reply in _GENERIC_FACT_REPLIES for reply in normalized_replies)


def _unsafe_fact_example(query: str, candidate: Candidate) -> bool:
    return any(term in query.lower() for term in _FACT_SENSITIVE_TERMS) and _generic_fact_answer(candidate.replies)


def shortlist_same_chat(query: str, candidates: list[Candidate], limit: int = 128) -> list[Candidate]:
    documents = [_response_anchor(candidate) for candidate in candidates]
    scores = _bm25_scores(query, documents)
    ranked_indices = sorted(range(len(candidates)), key=lambda index: (-float(scores[index]), _candidate_id(candidates[index])))
    return [
        candidates[index]
        for index in ranked_indices
        if scores[index] > 0
        and _conversation_move_compatible(query, documents[index])
        and not _unsafe_fact_example(query, candidates[index])
    ][: max(0, limit)]


def _message_text(message: Any) -> str:
    parts = [str(getattr(message, "text", "") or "")]
    image_description = str(getattr(message, "image_description", "") or "")
    image_text = str(getattr(message, "image_text", "") or "")
    quoted_text = str(getattr(message, "quoted_text", "") or "")
    parts.extend(part for part in (image_description, image_text, quoted_text) if part)
    return " ".join(part.strip() for part in parts if part.strip())


def _serialize_message(message: Any) -> dict[str, Any]:
    return {
        "sender": message.sender,
        "sender_type": str(message.sender_type.value),
        "text": message.text,
        "message_type": message.message_type,
        "image_description": message.image_description,
        "image_text": message.image_text,
        "quoted_text": message.quoted_text,
        "timestamp": message.timestamp,
    }


def _candidate_id(candidate: Candidate) -> str:
    payload = "\n".join([candidate.chat_id, *candidate.context, "=>", *candidate.replies])
    return "human_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _timestamp_seconds(value: int) -> float:
    return value / 1000 if value > 10_000_000_000 else float(value)


def _encode_documents(encoder: _BGEEncoder, documents: list[str]) -> np.ndarray:
    batches = []
    for start in range(0, len(documents), 256):
        batches.append(encoder.encode(documents[start:start + 256]))
    return np.concatenate(batches, axis=0) if batches else np.zeros((0, 512), dtype=np.float32)


def rank_same_chat(
    query: str,
    candidates: list[Candidate],
    *,
    dense_scores: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    documents = [_response_anchor(candidate) for candidate in candidates]
    bm25 = _bm25_scores(query, documents)
    if dense_scores is None:
        dense_scores = np.zeros(len(candidates), dtype=np.float32)
    positive_bm25 = bm25[bm25 > 0]
    bm25_scale = float(np.percentile(positive_bm25, 90)) if len(positive_bm25) else 1.0
    normalized_query = _normalized(query)
    ranked = []
    for index, candidate in enumerate(candidates):
        lexical = min(1.0, float(bm25[index]) / max(0.001, bm25_scale))
        semantic = float(dense_scores[index])
        combined = semantic * 0.72 + lexical * 0.28
        historical_context = "\n".join(candidate.context)
        response_anchor = documents[index]
        ranked.append({
            "candidate_id": _candidate_id(candidate),
            "combined_score": round(combined, 6),
            "semantic_similarity": round(semantic, 6),
            "bm25_score": round(float(bm25[index]), 6),
            "historical_context": candidate.context,
            "likely_replied_to": response_anchor,
            "human_reply": candidate.replies,
            "timestamp": candidate.timestamp,
            "source_provenance": candidate.source_provenance,
            "exact_target_reuse_risk": bool(
                normalized_query
                and (
                    normalized_query in _normalized(historical_context)
                    or _normalized(historical_context) in normalized_query
                )
            ),
        })
    return sorted(ranked, key=lambda row: (-row["combined_score"], -row["semantic_similarity"], row["candidate_id"]))


def _write_private(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.chmod(PRIVATE_MODE)
    temporary.replace(path)
    path.chmod(PRIVATE_MODE)


def main() -> None:
    parser = argparse.ArgumentParser(description="预览全量同聊天真人回复召回，不调用生成模型")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data/cases.db")
    parser.add_argument("--scenarios", type=Path, default=PROJECT_ROOT / "data/reports/private/reply_system_scenarios.json")
    parser.add_argument("--selection", type=Path, default=PROJECT_ROOT / "data/reports/private/reply_quality_gold_cases.json")
    parser.add_argument("--exports", type=Path, default=PROJECT_ROOT / "data/exports/b")
    parser.add_argument("--wiki-dir", type=Path, default=PROJECT_ROOT / "data/memory/wiki/users")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/reports/private/full_history_style_retrieval_preview.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-dense", action="store_true")
    args = parser.parse_args()

    generator_sha = hashlib.sha256((PROJECT_ROOT / "src/reply/generator.py").read_bytes()).hexdigest()
    if generator_sha != PROTECTED_C0_SHA256:
        raise ValueError("current generator no longer matches protected C0")
    source_fingerprint, cases = load_replay_cases(args.db, args.scenarios, args.selection)
    candidates = extract_candidates(args.exports, args.wiki_dir, human_before=CUTOFF)
    candidates = [candidate for candidate in candidates if 0 < _timestamp_seconds(candidate.timestamp) < CUTOFF]
    by_chat: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_chat[candidate.chat_id].append(candidate)

    encoder = None
    if not args.no_dense:
        backend = _try_import_encoder_deps()
        if not backend or not _model_path().exists():
            raise RuntimeError("local BGE encoder is unavailable")
        encoder = _BGEEncoder(_model_path(), backend)

    output_cases = []
    for case in cases:
        chat_name = case["all_messages"][0].chat_name if case["all_messages"] else ""
        chat_id = _chat_id(chat_name) if chat_name else ""
        target_parts = [_message_text(message) for message in case["unreplied"]]
        target_full = "\n".join(part for part in target_parts if part)
        target_text_messages = [str(message.text or "") for message in case["unreplied"]]
        target = _current_response_anchor(target_text_messages)
        target_has_text = bool(target)
        target_local_ids = {message.local_id for message in case["unreplied"] if message.local_id is not None}
        prior_messages = [message for message in case["all_messages"] if message.local_id not in target_local_ids]
        recent_context = [_message_text(message) for message in prior_messages[-6:]]
        semantic_query = "当前消息：" + target + "\n最近上下文：" + "\n".join(recent_context)
        pool = by_chat.get(chat_id, [])
        shortlist = shortlist_same_chat(target, pool) if target_has_text else []
        documents = [_response_anchor(candidate) for candidate in shortlist]
        dense = None
        if encoder is not None and documents:
            dense = _encode_documents(encoder, documents) @ encoder.encode([semantic_query])[0]
        ranked = rank_same_chat(target, shortlist, dense_scores=dense)[: max(0, args.top_k)]
        output_cases.append({
            "case_id": case["case_id"],
            "chat_id": chat_id,
            "is_group": case["is_group"],
            "evaluation_time": case["evaluation_time"].isoformat(),
            "context_original": [_serialize_message(message) for message in case["all_messages"]],
            "target_original": [_serialize_message(message) for message in case["unreplied"]],
            "retrieval_query": {"target": target, "full_target": target_full, "recent_context": recent_context},
            "same_chat_candidate_count": len(pool),
            "lexical_shortlist_count": len(shortlist),
            "top_candidates": ranked,
        })

    packet = {
        "schema_version": 1,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "purpose": "retrieval quality inspection only; no model generation and no production integration",
        "cutoff_epoch": CUTOFF,
        "source_fingerprint": source_fingerprint,
        "generator_sha256": generator_sha,
        "raw_pre_cutoff_candidate_count": len(candidates),
        "unique_chat_count": len(by_chat),
        "ranking": "strict same-chat; likely replied-to incoming turn (last message plus a short trailing vocative); high-precision conversation-move compatibility; fact-sensitive generic-answer block; image-only targets allow zero; local BGE plus same-chat BM25; preview still requires manual rejection",
        "model_generation_calls": 0,
        "bot_constructed": False,
        "cases": output_cases,
    }
    _write_private(args.output, packet)
    print(json.dumps({
        "output": str(args.output),
        "case_count": len(output_cases),
        "raw_pre_cutoff_candidate_count": len(candidates),
        "cases_with_same_chat_candidates": sum(bool(case["same_chat_candidate_count"]) for case in output_cases),
        "model_generation_calls": 0,
        "bot_constructed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
