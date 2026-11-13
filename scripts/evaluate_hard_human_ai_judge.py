#!/usr/bin/env python3
"""Build a harder human-vs-style-imitation reply discrimination benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.build_persona_few_shots import Candidate, extract_candidates
    from scripts.evaluate_human_ai_judge import (
        CUTOFF,
        DEFAULT_VOLCENGINE_CONFIG,
        PROJECT_ROOT,
        VOLCENGINE_DSV4F_MODEL,
        VolcengineAnthropicClient,
        _candidate_identity,
        _case_id,
        _chat_names,
        _few_shot_identities,
        _write_private,
        load_volcengine_credentials,
        score_predictions,
        write_benchmark_artifacts,
    )
except ModuleNotFoundError:
    from build_persona_few_shots import Candidate, extract_candidates
    from evaluate_human_ai_judge import (
    CUTOFF,
    DEFAULT_VOLCENGINE_CONFIG,
    PROJECT_ROOT,
    VOLCENGINE_DSV4F_MODEL,
    VolcengineAnthropicClient,
    _candidate_identity,
    _case_id,
    _chat_names,
    _few_shot_identities,
    _write_private,
    load_volcengine_credentials,
    score_predictions,
        write_benchmark_artifacts,
    )


def _seconds(value: int) -> float:
    return value / (1000 if value > 10_000_000_000 else 1)


def select_hard_holdouts(
    candidates: list[Candidate],
    excluded: set[tuple[Any, ...]],
    per_relation: int,
    minimum_prior_references: int = 6,
    excluded_case_ids: set[str] | None = None,
    minimum_context_messages: int = 3,
) -> list[Candidate]:
    excluded_case_ids = excluded_case_ids or set()
    by_chat: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.timestamp and _seconds(candidate.timestamp) < CUTOFF:
            by_chat[candidate.chat_id].append(candidate)
    eligible = []
    for candidate in candidates:
        if (
            _candidate_identity(candidate) in excluded
            or _case_id(candidate) in excluded_case_ids
            or not candidate.timestamp
            or _seconds(candidate.timestamp) >= CUTOFF
            or len(candidate.context) < minimum_context_messages
            or not 2 <= len("".join(candidate.replies)) <= 60
            or not 2 <= len("".join(candidate.context)) <= 220
        ):
            continue
        prior = [row for row in by_chat[candidate.chat_id] if row.timestamp < candidate.timestamp]
        if len(prior) >= minimum_prior_references:
            eligible.append(candidate)
    eligible.sort(key=lambda row: hashlib.sha256(_case_id(row).encode()).hexdigest())
    selected: list[Candidate] = []
    chats: set[str] = set()
    counts: Counter[str] = Counter()
    for candidate in eligible:
        relation = "group" if candidate.relation == "group" else "private"
        if candidate.chat_id in chats or counts[relation] >= per_relation:
            continue
        selected.append(candidate)
        chats.add(candidate.chat_id)
        counts[relation] += 1
        if counts["private"] >= per_relation and counts["group"] >= per_relation:
            break
    return selected


def style_references(
    target: Candidate,
    candidates: list[Candidate],
    all_target_identities: set[tuple[Any, ...]],
    limit: int = 6,
) -> list[Candidate]:
    target_replies = set(target.replies)
    rows = [
        row
        for row in candidates
        if row.chat_id == target.chat_id
        and row.timestamp < target.timestamp
        and _candidate_identity(row) not in all_target_identities
        and not target_replies.intersection(row.replies)
        and 2 <= len("".join(row.replies)) <= 60
    ]
    rows.sort(key=lambda row: row.timestamp, reverse=True)
    return rows[:limit]


def build_imitation_prompt(target: Candidate, references: list[Candidate]) -> list[dict[str, str]]:
    examples = [
        {"context": row.context, "reply": row.replies}
        for row in references
    ]
    system = (
        "你在做离线风格模仿测试。根据同一个人在同一聊天中的历史例子，回复当前聊天。"
        "学习短句节奏、分条习惯、口语感和自然幽默，但不要照搬例子中的事实、人名、承诺或原句。"
        "事实只能来自当前上下文；不得承诺转账、发红包、赴约等现实行动。"
        "不要解释，不要写得完整工整，不要刻意造金句。不要给多个备选反应，也不要逐点回答。"
        "通常只回1条，确实符合参考习惯时才拆成2条；每条尽量短。"
        "只输出JSON：{\"replies\":[\"...\"]}，最多2条。"
    )
    user = json.dumps(
        {"same_chat_style_examples": examples, "current_context": target.context},
        ensure_ascii=False,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_replies(raw: str) -> list[str]:
    text = raw.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return []
    replies = payload.get("replies") if isinstance(payload, dict) else None
    if not isinstance(replies, list):
        return []
    return [str(row).strip() for row in replies if isinstance(row, str) and row.strip()][:3]


def main() -> None:
    parser = argparse.ArgumentParser(description="高难度真人回复与风格模仿回复匿名二选一")
    parser.add_argument("--per-relation", type=int, default=6)
    parser.add_argument("--max-valid", type=int, default=12)
    parser.add_argument("--min-valid", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--score", type=Path)
    parser.add_argument("--exports", type=Path, default=PROJECT_ROOT / "data/exports/b")
    parser.add_argument("--wiki-dir", type=Path, default=PROJECT_ROOT / "data/memory/wiki/users")
    parser.add_argument("--volcengine-config", type=Path, default=DEFAULT_VOLCENGINE_CONFIG)
    private = PROJECT_ROOT / "data/reports/private"
    parser.add_argument("--basic-blind", type=Path, default=private / "human_ai_judge_blind.json")
    parser.add_argument("--leaked-blind", type=Path, default=private / "human_ai_judge_hard_leaked_blind.json")
    parser.add_argument("--replay", type=Path, default=private / "human_ai_judge_hard_replay.json")
    parser.add_argument("--blind", type=Path, default=private / "human_ai_judge_hard_blind.json")
    parser.add_argument("--answers", type=Path, default=private / "human_ai_judge_hard_answers.json")
    parser.add_argument("--result", type=Path, default=private / "human_ai_judge_hard_result.json")
    args = parser.parse_args()

    if args.score:
        score = score_predictions(
            json.loads(args.answers.read_text(encoding="utf-8")),
            json.loads(args.score.read_text(encoding="utf-8")),
        )
        _write_private(args.result, score)
        print(json.dumps({key: score[key] for key in ("status", "answered", "total", "correct", "accuracy")}))
        return

    candidates = extract_candidates(args.exports, args.wiki_dir, human_before=CUTOFF)
    few_shot_excluded = _few_shot_identities(PROJECT_ROOT / "data")
    previously_seen_case_ids = {
        row["case_id"]
        for path in (args.basic_blind, args.leaked_blind)
        if path.exists()
        for row in json.loads(path.read_text(encoding="utf-8")).get("cases", [])
    }
    selected = select_hard_holdouts(
        candidates,
        few_shot_excluded,
        args.per_relation,
        excluded_case_ids=previously_seen_case_ids,
    )
    target_identities = {_candidate_identity(row) for row in selected}
    references = {row.chat_id: style_references(row, candidates, target_identities) for row in selected}
    validation = {
        "raw_candidate_count": len(candidates),
        "selected_holdout_count": len(selected),
        "selected_relationships": dict(Counter("group" if row.relation == "group" else "private" for row in selected)),
        "all_targets_excluded_from_style_references": all(
            _candidate_identity(ref) not in target_identities
            for rows in references.values()
            for ref in rows
        ),
        "target_replies_absent_from_style_references": all(
            not set(target.replies).intersection(reply for ref in references[target.chat_id] for reply in ref.replies)
            for target in selected
        ),
        "minimum_style_references": min((len(rows) for rows in references.values()), default=0),
        "minimum_context_messages": min((len(row.context) for row in selected), default=0),
        "target_reply_passed_to_model": False,
        "previously_seen_case_count": len(previously_seen_case_ids),
        "previously_seen_targets_excluded": all(
            _case_id(row) not in previously_seen_case_ids for row in selected
        ),
        "bot_constructed": False,
        "production_generator_used": False,
    }
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return
    if not validation["all_targets_excluded_from_style_references"] or not validation["target_replies_absent_from_style_references"]:
        raise ValueError("target leakage detected")

    api_key, base_url = load_volcengine_credentials(args.volcengine_config)
    client = VolcengineAnthropicClient(api_key, base_url, VOLCENGINE_DSV4F_MODEL)
    chat_names = _chat_names(args.exports)
    cases = []
    for target in selected:
        refs = references[target.chat_id]
        raw = client.chat(
            messages=build_imitation_prompt(target, refs),
            temperature=0.9,
            max_tokens=300,
            timeout=60,
            response_format={"type": "json_object"},
        )
        replies = parse_replies(raw)
        if not replies or replies == target.replies or any(reply in {r for ref in refs for r in ref.replies} for reply in replies):
            continue
        cases.append({
            "case_id": _case_id(target),
            "relationship": "group" if target.relation == "group" else "private",
            "chat_name_resolved": bool(chat_names.get(target.chat_id)),
            "context_original": target.context,
            "human_reply_original": target.replies,
            "ai_reply_original": replies,
            "style_reference_count": len(refs),
            "style_reference_ids": [_case_id(ref) for ref in refs],
        })
        if len(cases) >= args.max_valid:
            break
    report = {
        "schema_version": 1,
        "privacy": "LOCAL PRIVATE: contains real chat text; never publish or commit",
        "benchmark_kind": "hard_human_ai_reply_discrimination_replay",
        "candidate_label": "Volcengine DeepSeek V4 Flash adversarial same-chat style imitation; not current C0 output",
        "backend": {"requested_model": client.model, "resolved_model": client.resolved_model},
        "validation": validation,
        "cases": cases,
    }
    write_benchmark_artifacts(
        cases,
        report,
        minimum_valid=args.min_valid,
        replay_path=args.replay,
        blind_path=args.blind,
        answers_path=args.answers,
    )
    print(json.dumps({"valid_cases": len(cases), **validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
