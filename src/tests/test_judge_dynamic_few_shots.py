import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.build_judge_dynamic_few_shots import (
    _load_pool,
    admitted_examples,
    build_matcher_prompt,
    leakage_exclusions,
    parse_option_divergence_response,
    parse_matcher_response,
    prepare_from_retrieval_audit,
    retrieve_case_candidates,
    run_codex_matcher,
)
from scripts.evaluate_human_ai_judge import _json_sha256, run_codex_judge_independently


class FakeRetriever:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def retrieve(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        excluded = set(kwargs["exclude_ids"])
        return [row for row in self.rows if row["id"] not in excluded][: kwargs["limit"]]


class ShortSituationRetriever(FakeRetriever):
    def retrieve(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        excluded = set(kwargs["exclude_ids"])
        rows = [row for row in self.rows if row["id"] not in excluded]
        if kwargs["use_situation"]:
            rows = rows[:2]
        return rows[: kwargs["limit"]]


def _row(row_id, *, chat_id, relationship="acquaintance", context=None, reply=None):
    return {
        "id": row_id,
        "chat_id": chat_id,
        "relationship": relationship,
        "context": context or [f"上下文-{row_id}"],
        "reply": reply or [f"真人回复-{row_id}"],
        "timestamp": f"2026-08-26T00:00:{row_id[-2:]}Z",
    }


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _match_row(
    candidate_id,
    *,
    situation_match=2,
    event_stage_match=2,
    addressee_match=2,
    historical_action="answer_now",
    is_generic_formula=0,
):
    return {
        "candidate_id": candidate_id,
        "situation_match": situation_match,
        "event_stage_match": event_stage_match,
        "addressee_match": addressee_match,
        "historical_action": historical_action,
        "is_generic_formula": is_generic_formula,
        "reason": "具体处境、阶段、回复对象和历史真人行为均已比较",
    }


def _decision_row(candidate_id, **kwargs):
    row = _match_row(candidate_id, **kwargs)
    row["supported_option"] = (
        "A" if row["historical_action"] == "answer_now"
        else "B" if row["historical_action"] == "defer"
        else "none"
    )
    row["admitted"] = (
        row["situation_match"] == 2
        and row["supported_option"] != "none"
        and row["is_generic_formula"] == 0
    )
    return row


def _fake_codex(prompts):
    def fake_run(command, input, **kwargs):
        prompts.append(input)
        output_path = command[command.index("--output-last-message") + 1]
        Path(output_path).write_text(
            '{"human_option":"A","confidence":0.8,"reason":"更自然"}',
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return fake_run


def test_pool_rejects_empty_and_duplicate_ids(tmp_path):
    pool = tmp_path / "pool.jsonl"
    pool.write_text('{"id":""}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate or empty"):
        _load_pool(pool)

    pool.write_text('{"id":"same"}\n{"id":"same"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate or empty"):
        _load_pool(pool)


def test_retrieval_allows_nonoverlapping_target_chat_and_excludes_answers():
    pool = [
        _row("same-chat", chat_id="target-chat"),
        _row(
            "same-chat-overlap",
            chat_id="target-chat",
            context=["前文", "当前问题"],
        ),
        _row("same-answer", chat_id="other-1", reply=["真人答案"]),
        _row(
            "same-context-answer",
            chat_id="other-2",
            context=["前文", "当前问题"],
            reply=["AI答案"],
        ),
        _row("safe-1", chat_id="safe-chat-1"),
        _row("safe-2", chat_id="safe-chat-2"),
        _row("safe-3", chat_id="safe-chat-3"),
    ]
    replay = {
        "case_id": "source-case",
        "source_chat_id": "target-chat",
        "relationship": "private",
        "context_original": ["前文", "当前问题"],
    }
    base = {
        "case_id": "blind-case",
        "relationship": "private",
        "context_original": [
            '<conversation_history><message role="other">甲（10:00）：前文</message></conversation_history>',
            '<messages_to_reply><message role="other">甲（10:01）：当前问题</message></messages_to_reply>',
        ],
        "option_A": ["真人答案"],
        "option_B": ["AI答案"],
    }
    retriever = FakeRetriever(pool)

    examples_a, _ = retrieve_case_candidates(retriever, pool, base, replay, 3)
    swapped = {**base, "option_A": base["option_B"], "option_B": base["option_A"]}
    examples_b, _ = retrieve_case_candidates(retriever, pool, swapped, replay, 3)

    assert [row["id"] for row in examples_a] == ["same-chat", "safe-1", "safe-2"]
    assert [row["id"] for row in examples_b] == ["same-chat", "safe-1", "safe-2"]
    assert retriever.calls[0]["exclude_ids"] == retriever.calls[1]["exclude_ids"]
    assert retriever.calls[0]["relationship"] == "acquaintance"
    assert retriever.calls[0]["is_group"] is False
    assert "真人答案" not in retriever.calls[0]["query"]
    assert "AI答案" not in retriever.calls[0]["query"]

    excluded, reasons = leakage_exclusions(
        pool,
        source_chat_id="target-chat",
        target_contexts=[["前文", "当前问题"]],
        options=[["真人答案"], ["AI答案"]],
    )
    assert excluded == {"same-chat-overlap", "same-answer", "same-context-answer"}
    assert reasons["overlapping_source_context"] == 1
    assert reasons["exact_option_reply"] == 2


def test_context_overlap_uses_pairs_or_long_messages_only_within_source_chat():
    long_message = "这是长度超过十二个字符的完整消息"
    pool = [
        _row("same-pair", chat_id="target", context=["第一句", "第二句"]),
        _row("same-long", chat_id="target", context=[long_message]),
        _row("same-short", chat_id="target", context=["哈哈"]),
        _row("other-pair", chat_id="other", context=["第一句", "第二句"]),
        _row("other-long", chat_id="other", context=[long_message]),
    ]

    excluded, reasons = leakage_exclusions(
        pool,
        source_chat_id="target",
        target_contexts=[["哈哈", "第一句", "第二句", long_message]],
        options=[["候选甲"], ["候选乙"]],
    )

    assert excluded == {"same-pair", "same-long"}
    assert reasons == {"overlapping_source_context": 2}


def test_context_overlap_includes_historical_reply_as_next_message():
    pool = [
        _row(
            "same-context-reply-pair",
            chat_id="target",
            context=["好 刚看到"],
            reply=["10min到"],
        ),
        _row(
            "other-chat-context-reply-pair",
            chat_id="other",
            context=["好 刚看到"],
            reply=["10min到"],
        ),
    ]

    excluded, reasons = leakage_exclusions(
        pool,
        source_chat_id="target",
        target_contexts=[["前文", "好 刚看到", "10min到", "后文"]],
        options=[["候选甲"], ["候选乙"]],
    )

    assert excluded == {"same-context-reply-pair"}
    assert reasons == {"overlapping_source_context": 1}


def test_expanded_retrieval_accepts_available_candidates_above_frozen_minimum():
    pool = [
        _row(f"safe-{index}", chat_id=f"safe-chat-{index}")
        for index in range(5)
    ]
    retriever = FakeRetriever(pool)
    blind = {
        "case_id": "blind-case",
        "relationship": "private",
        "context_original": [
            '<messages_to_reply><message role="other">甲（10:01）：当前问题</message></messages_to_reply>'
        ],
        "option_A": ["候选甲"],
        "option_B": ["候选乙"],
    }
    replay = {
        "case_id": "source-case",
        "source_chat_id": "target-chat",
        "relationship": "private",
        "context_original": ["当前问题"],
    }

    examples, audit = retrieve_case_candidates(
        retriever,
        pool,
        blind,
        replay,
        10,
        diversity_prefix_size=3,
        minimum_candidates_per_case=3,
    )

    assert len(examples) == 5
    assert audit["requested_candidate_limit"] == 10
    assert audit["retrieved_candidate_count"] == 5


def test_retrieval_supplements_short_situation_bucket_without_duplicates():
    pool = [
        _row(f"safe-{index}", chat_id=f"safe-chat-{index}")
        for index in range(5)
    ]
    retriever = ShortSituationRetriever(pool)
    blind = {
        "case_id": "blind-case",
        "relationship": "private",
        "context_original": [
            '<messages_to_reply><message role="other">甲（10:01）：当前问题</message></messages_to_reply>'
        ],
        "option_A": ["候选甲"],
        "option_B": ["候选乙"],
    }
    replay = {
        "case_id": "source-case",
        "source_chat_id": "target-chat",
        "relationship": "private",
        "context_original": ["当前问题"],
    }

    examples, audit = retrieve_case_candidates(retriever, pool, blind, replay, 5)

    assert [row["id"] for row in examples] == [
        "safe-0",
        "safe-1",
        "safe-2",
        "safe-3",
        "safe-4",
    ]
    assert audit["situation_candidate_count"] == 2
    assert audit["supplemental_candidate_count"] == 3
    assert retriever.calls[0]["use_situation"] is True
    assert retriever.calls[1]["use_situation"] is False
    assert retriever.calls[1]["exclude_ids"] == {"safe-0", "safe-1"}


def test_group_retrieval_stays_in_group_pool():
    pool = [
        _row(f"group-{index}", chat_id=f"chat-{index}", relationship="group")
        for index in range(3)
    ]
    retriever = FakeRetriever(pool)
    blind = {
        "case_id": "group-case",
        "relationship": "group",
        "context_original": [
            '<messages_to_reply><message role="other">甲（10:01）：跟不跟</message></messages_to_reply>'
        ],
        "option_A": ["跟"],
        "option_B": ["不跟"],
    }
    replay = {
        "case_id": "source-group",
        "source_chat_id": "target-group",
        "relationship": "group",
        "context_original": ["跟不跟"],
    }

    examples, _ = retrieve_case_candidates(retriever, pool, blind, replay, 3)

    assert len(examples) == 3
    assert {row["relationship"] for row in examples} == {"group"}
    assert retriever.calls[0]["relationship"] == "group"
    assert retriever.calls[0]["is_group"] is True


def test_prepare_from_retrieval_audit_reuses_frozen_candidates_and_rejects_drift():
    blind = {
        "reference_sha256": "ref-sha",
        "cases": [{
            "case_id": "blind-one",
            "relationship": "private",
            "context_original": [
                '<messages_to_reply><message role="other">甲（10:01）：当前问题</message></messages_to_reply>'
            ],
        }],
    }
    answers = {
        "cases": [{"case_id": "blind-one", "source_case_id": "source-one"}],
    }
    replay = {
        "cases": [{
            "case_id": "source-one",
            "source_chat_id": "target-chat",
            "relationship": "private",
        }],
    }
    reference = {"reference_sha256": "ref-sha"}
    selected = [
        {
            "id": f"frozen-{index}",
            "pool_relationship": "acquaintance",
            "chat_id": "target-chat" if index == 0 else f"other-chat-{index}",
            "context": [f"历史上下文-{index}"],
            "human_reply": [f"真人回复-{index}"],
        }
        for index in range(66)
    ]
    audit = {
        "kind": "judge_dynamic_few_shot_retrieval_audit",
        "blind_sha256": _json_sha256(blind),
        "reference_sha256": "ref-sha",
        "pool_sha256": "pool-sha",
        "configuration": {"retrieval_candidates_per_case": 100},
        "cases": [{
            "case_id": "blind-one",
            "source_case_id": "source-one",
            "source_chat_id": "target-chat",
            "relationship": "private",
            "query_messages": [{"sender": "甲", "role": "other", "text": "当前问题"}],
            "selected": selected,
        }],
    }

    prepared = prepare_from_retrieval_audit(
        audit,
        blind=blind,
        answers=answers,
        replay=replay,
        reference=reference,
        pool_sha256="pool-sha",
    )

    assert len(prepared["blind-one"]["candidates"]) == 66
    assert prepared["blind-one"]["candidates"][0]["id"] == "frozen-0"
    assert prepared["blind-one"]["candidates"][-1]["id"] == "frozen-65"
    with pytest.raises(ValueError, match="few-shot pool"):
        prepare_from_retrieval_audit(
            audit,
            blind=blind,
            answers=answers,
            replay=replay,
            reference=reference,
            pool_sha256="different-pool-sha",
        )

    leaked = json.loads(json.dumps(audit))
    leaked["cases"][0]["selected"][0]["context"] = ["当前问题", "补充一句"]
    blind["cases"][0]["context_original"].append(
        '<conversation_history><message role="other">甲（10:02）：补充一句</message></conversation_history>'
    )
    audit["blind_sha256"] = _json_sha256(blind)
    leaked["blind_sha256"] = _json_sha256(blind)
    leaked["cases"][0]["query_messages"].append(
        {"sender": "甲", "role": "other", "text": "补充一句"}
    )
    with pytest.raises(ValueError, match="overlapping context"):
        prepare_from_retrieval_audit(
            leaked,
            blind=blind,
            answers=answers,
            replay=replay,
            reference=reference,
            pool_sha256="pool-sha",
        )


def obsolete_v6_matcher_applies_every_strict_admission_threshold():
    rows = [
        _match_row("valid"),
        *[
            _match_row(
                f"context-{index}",
                context_scores=tuple(1 if field == index else 2 for field in range(4)),
            )
            for index in range(4)
        ],
        _match_row("slot-mismatch", historical_unresolved_information_slot="date"),
        _match_row(
            "slot-other",
            current_unresolved_information_slot="other",
            historical_unresolved_information_slot="other",
        ),
        _match_row("no-prior-context", requires_prior_context=0),
        _match_row("generic", is_generic_formula=1),
        _match_row(
            "high-side-seven",
            option_a_scores=(2, 2, 1, 1, 1),
            option_b_scores=(0, 0, 0, 0, 0),
        ),
        _match_row(
            "margin-three",
            option_a_scores=(2, 2, 2, 1, 1),
            option_b_scores=(1, 1, 1, 1, 1),
        ),
    ]
    candidates = [
        {
            "id": row["candidate_id"],
            "relationship": "group",
            "context": [f"历史上下文-{row['candidate_id']}"],
            "human_reply": [f"历史真人回复-{row['candidate_id']}"],
        }
        for row in rows
    ]
    prompt = build_matcher_prompt(
        "group",
        [{"role": "other", "speaker": "甲", "text": "当前问题"}],
        ["继续追问"],
        ["结束话题"],
        candidates,
    )
    for forbidden in (
        '"source_case_id"', '"source_chat_id"', '"actual"', '"correct"'
    ):
        assert forbidden not in prompt
    assert '"anonymous_options":{"A":["继续追问"],"B":["结束话题"]}' in prompt
    decisions = parse_matcher_response(
        json.dumps({"candidates": rows}),
        [row["id"] for row in candidates],
    )

    admitted_by_id = {
        row["candidate_id"]: row["admitted"]
        for row in decisions["candidates"]
    }
    assert admitted_by_id["valid"] is True
    assert all(
        admitted is False
        for candidate_id, admitted in admitted_by_id.items()
        if candidate_id != "valid"
    )


def test_matcher_allows_reordered_ids_but_rejects_missing_or_duplicate_ids():
    def row(candidate_id):
        return _match_row(candidate_id)

    reordered = json.dumps({"candidates": [row("candidate-1"), row("candidate-0")]})
    parsed = parse_matcher_response(reordered, ["candidate-0", "candidate-1"])
    assert [item["candidate_id"] for item in parsed["candidates"]] == [
        "candidate-1", "candidate-0"
    ]

    missing = json.dumps({"candidates": [row("candidate-0")]})
    with pytest.raises(ValueError, match="candidate ids changed"):
        parse_matcher_response(missing, ["candidate-0", "candidate-1"])

    duplicate = json.dumps({"candidates": [row("candidate-0"), row("candidate-0")]})
    with pytest.raises(ValueError, match="candidate ids changed"):
        parse_matcher_response(duplicate, ["candidate-0", "candidate-1"])


def test_matcher_repairs_only_unique_short_terminal_id_changes():
    truncated = json.dumps({"candidates": [_match_row("human_a9742aae3c")]})
    parsed = parse_matcher_response(truncated, ["human_a9742aae3c80"])
    assert parsed["candidates"][0]["candidate_id"] == "human_a9742aae3c80"

    appended = json.dumps({"candidates": [_match_row("human_2d23138acc05e")]})
    parsed = parse_matcher_response(appended, ["human_2d23138acc05"])
    assert parsed["candidates"][0]["candidate_id"] == "human_2d23138acc05"

    changed = json.dumps({"candidates": [_match_row("human_2d23138acc06")]})
    with pytest.raises(ValueError, match="candidate ids changed"):
        parse_matcher_response(changed, ["human_2d23138acc05"])


def obsolete_v6_matcher_rejects_only_candidate_with_unsupported_category():
    invalid = _match_row("invalid", response_goal_label="answer_question")
    valid = _match_row("valid")

    parsed = parse_matcher_response(
        json.dumps({"candidates": [invalid, valid]}),
        ["invalid", "valid"],
    )

    assert parsed["candidates"][0]["response_goal_label"] == "other"
    assert parsed["candidates"][0]["admitted"] is False
    assert "answer_question" in parsed["candidates"][0]["reason"]
    assert parsed["candidates"][1]["admitted"] is True


def obsolete_v6_matcher_batches_all_candidates_and_restores_input_order(monkeypatch):
    calls = []

    def fake_run(command, input, **kwargs):
        packet = json.loads(input.split("输入：\n", 1)[1])
        candidate_ids = [
            row["candidate_id"] for row in packet["historical_candidates"]
        ]
        calls.append(candidate_ids)
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(
            json.dumps({
                "candidates": [
                    _match_row(candidate_id)
                    for candidate_id in reversed(candidate_ids)
                ]
            }),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "scripts.build_judge_dynamic_few_shots.subprocess.run", fake_run
    )
    candidates = [
        {
            "id": f"candidate-{index:02d}",
            "relationship": "private",
            "context": [f"上下文-{index}"],
            "human_reply": [f"回复-{index}"],
        }
        for index in range(45)
    ]

    result = run_codex_matcher(
        "private",
        [{"role": "other", "speaker": "甲", "text": "现在怎么样"}],
        ["候选甲"],
        ["候选乙"],
        candidates,
        180,
        batch_size=20,
        batch_workers=3,
    )

    assert sorted(len(batch) for batch in calls) == [5, 20, 20]
    assert {candidate_id for batch in calls for candidate_id in batch} == {
        row["id"] for row in candidates
    }
    assert [
        row["candidate_id"] for row in result["decision"]["candidates"]
    ] == [row["id"] for row in candidates]
    assert result["matcher_batch_size"] == 20
    assert result["matcher_batch_workers"] == 3
    assert result["batch_count"] == 3
    assert result["elapsed_ms"] == sum(
        row["elapsed_ms"] for row in result["batch_audit"]
    )
    assert result["tool_call_count"] == 0
    assert result["event_parse_errors"] == []


def obsolete_v6_matcher_discards_whole_case_when_any_batch_fails(monkeypatch):
    call_count = 0

    def fake_run(command, input, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return SimpleNamespace(returncode=1, stdout="", stderr="batch failed")
        packet = json.loads(input.split("输入：\n", 1)[1])
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(
            json.dumps({
                "candidates": [
                    _match_row(row["candidate_id"])
                    for row in packet["historical_candidates"]
                ]
            }),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "scripts.build_judge_dynamic_few_shots.subprocess.run", fake_run
    )
    candidates = [
        {
            "id": f"candidate-{index:02d}",
            "relationship": "private",
            "context": [f"上下文-{index}"],
            "human_reply": [f"回复-{index}"],
        }
        for index in range(45)
    ]

    with pytest.raises(RuntimeError, match="batch failed"):
        run_codex_matcher(
            "private", [], ["候选甲"], ["候选乙"], candidates, 180,
            batch_size=20,
        )
    assert call_count == 2


def obsolete_v6_admitted_examples_deduplicates_events_and_clears_same_cluster_conflict():
    candidates = [
        {
            "id": "slice-weaker",
            "historical_event_id": "event-one",
            "relationship": "private",
            "context": ["同一事件切片一"],
            "human_reply": ["回复一"],
            "same_source_chat": False,
            "semantic_rank": 1,
        },
        {
            "id": "slice-stronger",
            "historical_event_id": "event-one",
            "relationship": "private",
            "context": ["同一事件切片二"],
            "human_reply": ["回复二"],
            "same_source_chat": False,
            "semantic_rank": 2,
        },
        {
            "id": "independent-event",
            "historical_event_id": "event-two",
            "relationship": "private",
            "context": ["独立事件"],
            "human_reply": ["回复三"],
            "same_source_chat": False,
            "semantic_rank": 3,
        },
    ]
    raw_rows = [
        _match_row(
            "slice-weaker",
            option_a_scores=(2, 2, 2, 2, 1),
            option_b_scores=(0, 0, 0, 0, 0),
        ),
        _match_row("slice-stronger"),
        _match_row("independent-event"),
    ]
    decisions = parse_matcher_response(
        json.dumps({"candidates": raw_rows}),
        [row["id"] for row in candidates],
    )

    assert admitted_examples(
        candidates[:2],
        {"decision": {"candidates": decisions["candidates"][:2]}},
    ) == []
    assert [
        row["id"] for row in admitted_examples(candidates, {"decision": decisions})
    ] == ["slice-stronger"]

    conflicting_rows = raw_rows + [
        _match_row(
            "opposite",
            option_a_scores=(0, 0, 0, 0, 0),
            option_b_scores=(2, 2, 2, 2, 2),
        )
    ]
    conflicting_candidates = candidates + [{
        "id": "opposite",
        "historical_event_id": "event-three",
        "relationship": "private",
        "context": ["反向证据"],
        "human_reply": ["反向回复"],
        "same_source_chat": False,
        "semantic_rank": 4,
    }]
    conflicting_decisions = parse_matcher_response(
        json.dumps({"candidates": conflicting_rows}),
        [row["id"] for row in conflicting_candidates],
    )

    assert admitted_examples(
        conflicting_candidates, {"decision": conflicting_decisions}
    ) == []


def obsolete_v6_event_fallback_uses_unique_candidate_id_not_chat_timestamp():
    candidates = [
        {
            "id": f"candidate-{index}",
            "chat_id": "same-chat",
            "timestamp": "same-timestamp",
            "relationship": "group",
            "context": [f"上下文-{index}"],
            "human_reply": [f"回复-{index}"],
            "same_source_chat": True,
            "semantic_rank": index + 1,
        }
        for index in range(2)
    ]
    decisions = parse_matcher_response(
        json.dumps({
            "candidates": [_match_row(row["id"]) for row in candidates]
        }),
        [row["id"] for row in candidates],
    )

    assert len(admitted_examples(candidates, {"decision": decisions})) == 1


def obsolete_v6_admitted_examples_clears_cross_cluster_conflict():
    candidates = []
    raw_rows = []
    for prefix, goal, supports_a in (
        ("answer", "answer", True),
        ("clarify", "clarify", False),
    ):
        for index in range(2):
            candidate_id = f"{prefix}-{index}"
            candidates.append({
                "id": candidate_id,
                "historical_event_id": candidate_id,
                "relationship": "private",
                "context": [f"上下文-{candidate_id}"],
                "human_reply": [f"回复-{candidate_id}"],
                "same_source_chat": False,
                "semantic_rank": len(candidates),
            })
            raw_rows.append(_match_row(
                candidate_id,
                response_goal_label=goal,
                option_a_scores=(2, 2, 2, 2, 2) if supports_a else (0, 0, 0, 0, 0),
                option_b_scores=(0, 0, 0, 0, 0) if supports_a else (2, 2, 2, 2, 2),
            ))
    decisions = parse_matcher_response(
        json.dumps({"candidates": raw_rows}),
        [row["id"] for row in candidates],
    )

    assert admitted_examples(candidates, {"decision": decisions}) == []


def obsolete_v6_admitted_examples_uses_six_rank_keys_and_stops_at_ten():
    candidates = []
    raw_rows = []

    def add_cluster(
        prefix,
        label,
        event_count,
        option_a_scores,
        option_b_scores,
        *,
        same_source_chat=False,
        semantic_rank=100,
    ):
        for index in range(event_count):
            candidate_id = f"{prefix}-{index}"
            candidates.append({
                "id": candidate_id,
                "historical_event_id": f"{prefix}-event-{index}",
                "relationship": "group",
                "context": [f"上下文-{candidate_id}"],
                "human_reply": [f"回复-{candidate_id}"],
                "same_source_chat": same_source_chat and index == 0,
                "semantic_rank": semantic_rank + index,
            })
            raw_rows.append(_match_row(
                candidate_id,
                specific_response_point_label=label,
                option_a_scores=option_a_scores,
                option_b_scores=option_b_scores,
            ))

    score_10 = (2, 2, 2, 2, 2)
    score_9 = (2, 2, 2, 2, 1)
    score_8 = (2, 2, 2, 1, 1)
    score_4 = (1, 1, 1, 1, 0)
    score_2 = (1, 1, 0, 0, 0)
    score_1 = (1, 0, 0, 0, 0)
    score_0 = (0, 0, 0, 0, 0)
    add_cluster("three-margin10", "greeting_presence", 3, score_10, score_0)
    add_cluster("four-margin8", "identity_person", 4, score_8, score_0)
    add_cluster("two-margin10", "date_time", 2, score_10, score_0)
    add_cluster("two-high10", "location", 2, score_10, score_2)
    add_cluster("two-high9", "method_process", 2, score_9, score_1)
    add_cluster(
        "two-same", "reason_explanation", 2, score_8, score_0,
        same_source_chat=True,
    )
    add_cluster(
        "two-sem1", "quantity_price", 2, score_8, score_0,
        semantic_rank=1,
    )
    add_cluster(
        "id-a", "status_progress", 2, score_8, score_0,
        semantic_rank=5,
    )
    add_cluster(
        "id-b", "opinion_evaluation", 2, score_8, score_0,
        semantic_rank=5,
    )
    add_cluster("lower-margin7", "decision_choice", 2, score_8, score_1)
    add_cluster("lower-margin6", "invitation_request", 2, score_8, score_2)
    add_cluster("lower-margin4", "factual_claim", 2, score_8, score_4)
    decisions = parse_matcher_response(
        json.dumps({"candidates": raw_rows}),
        [row["id"] for row in candidates],
    )

    selected = admitted_examples(candidates, {"decision": decisions})

    assert [row["id"] for row in selected] == [
        "three-margin10-0",
        "four-margin8-0",
        "two-margin10-0",
        "two-high10-0",
        "two-high9-0",
        "two-same-0",
        "two-sem1-0",
        "id-a-0",
        "id-b-0",
        "lower-margin7-0",
    ]
    assert all(
        set(row) == {"id", "relationship", "context", "human_reply"}
        for row in selected
    )


def test_v7_matcher_classifies_option_behavior_once_and_reuses_fixed_labels(
    monkeypatch,
):
    calls = []

    def fake_run(command, input, **kwargs):
        packet = json.loads(input.split("输入：\n", 1)[1])
        output_path = Path(command[command.index("--output-last-message") + 1])
        if "anonymous_options" in packet:
            calls.append(("option", packet))
            payload = {
                "option_a_action": "answer_now",
                "option_b_action": "defer",
                "has_behavior_divergence": True,
                "reason": "A 立即回答，B 延后回答",
            }
        else:
            calls.append(("history", packet))
            payload = {
                "candidates": [
                    _match_row(row["candidate_id"])
                    for row in reversed(packet["historical_candidates"])
                ]
            }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "scripts.build_judge_dynamic_few_shots.subprocess.run", fake_run
    )
    candidates = [
        {
            "id": f"candidate-{index:02d}",
            "relationship": "private",
            "context": [f"上下文-{index}"],
            "human_reply": [f"回复-{index}"],
        }
        for index in range(45)
    ]

    result = run_codex_matcher(
        "private",
        [{"role": "other", "sender": "甲", "text": "现在怎么样"}],
        ["现在就说"],
        ["晚点再说"],
        candidates,
        180,
        batch_size=20,
        batch_workers=3,
    )

    assert [kind for kind, _ in calls].count("option") == 1
    history_packets = [packet for kind, packet in calls if kind == "history"]
    assert sorted(len(packet["historical_candidates"]) for packet in history_packets) == [5, 20, 20]
    assert all(
        packet["fixed_option_actions"] == {"A": "answer_now", "B": "defer"}
        for packet in history_packets
    )
    assert [
        row["candidate_id"] for row in result["decision"]["candidates"]
    ] == [row["id"] for row in candidates]
    assert all(
        row["supported_option"] == "A" and row["admitted"]
        for row in result["decision"]["candidates"]
    )
    assert result["decision"]["option_a_action"] == "answer_now"
    assert result["decision"]["option_b_action"] == "defer"
    assert result["batch_count"] == 3


@pytest.mark.parametrize(
    ("option_a_action", "option_b_action"),
    [("confirm", "confirm"), ("other", "defer")],
)
def test_v7_matcher_skips_history_when_option_behavior_is_not_usable(
    monkeypatch, option_a_action, option_b_action
):
    calls = []

    def fake_run(command, input, **kwargs):
        calls.append(input)
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(json.dumps({
            "option_a_action": option_a_action,
            "option_b_action": option_b_action,
            "has_behavior_divergence": False,
            "reason": "没有可用的固定行为分歧",
        }), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "scripts.build_judge_dynamic_few_shots.subprocess.run", fake_run
    )
    candidates = [{
        "id": "candidate",
        "relationship": "group",
        "context": ["历史上下文"],
        "human_reply": ["历史回复"],
    }]

    result = run_codex_matcher(
        "group", [], ["候选甲"], ["候选乙"], candidates, 180
    )

    assert len(calls) == 1
    assert result["batch_count"] == 0
    assert result["decision"]["candidates"][0]["supported_option"] == "none"
    assert result["decision"]["candidates"][0]["admitted"] is False


def test_v7_matcher_applies_all_four_admission_conditions(monkeypatch):
    def fake_run(command, input, **kwargs):
        packet = json.loads(input.split("输入：\n", 1)[1])
        output_path = Path(command[command.index("--output-last-message") + 1])
        if "anonymous_options" in packet:
            payload = {
                "option_a_action": "answer_now",
                "option_b_action": "defer",
                "has_behavior_divergence": True,
                "reason": "立即回答与延后回答",
            }
        else:
            rows = {
                "valid": _match_row("valid"),
                "weak-situation": _match_row("weak-situation", situation_match=1),
                "unrelated-action": _match_row("unrelated-action", historical_action="accept"),
                "generic": _match_row("generic", is_generic_formula=1),
            }
            payload = {
                "candidates": [
                    rows[row["candidate_id"]]
                    for row in packet["historical_candidates"]
                ]
            }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "scripts.build_judge_dynamic_few_shots.subprocess.run", fake_run
    )
    candidates = [{
        "id": candidate_id,
        "relationship": "private",
        "context": [candidate_id],
        "human_reply": [candidate_id],
    } for candidate_id in ("valid", "weak-situation", "unrelated-action", "generic")]

    result = run_codex_matcher(
        "private", [], ["立即回答"], ["以后回答"], candidates, 180
    )
    admitted_by_id = {
        row["candidate_id"]: row["admitted"]
        for row in result["decision"]["candidates"]
    }

    assert admitted_by_id == {
        "valid": True,
        "weak-situation": False,
        "unrelated-action": False,
        "generic": False,
    }


def test_v7_unknown_historical_action_is_other_and_rejected():
    parsed = parse_matcher_response(
        json.dumps({"candidates": [
            _match_row("unknown", historical_action="invented_action")
        ]}),
        ["unknown"],
    )

    assert parsed["candidates"][0]["historical_action"] == "other"
    assert "invented_action" in parsed["candidates"][0]["reason"]


def test_v7_admission_needs_two_independent_events_and_deduplicates_slices():
    candidates = [
        {
            "id": "event-one-weak",
            "historical_event_id": "event-one",
            "relationship": "private",
            "context": ["事件一较弱切片"],
            "human_reply": ["回复一"],
            "same_source_chat": True,
            "semantic_rank": 1,
        },
        {
            "id": "event-one-strong",
            "historical_event_id": "event-one",
            "relationship": "private",
            "context": ["事件一较强切片"],
            "human_reply": ["回复二"],
            "same_source_chat": False,
            "semantic_rank": 2,
        },
        {
            "id": "event-two",
            "historical_event_id": "event-two",
            "relationship": "private",
            "context": ["事件二"],
            "human_reply": ["回复三"],
            "same_source_chat": False,
            "semantic_rank": 3,
        },
    ]
    decisions = [
        _decision_row("event-one-weak", event_stage_match=1),
        _decision_row("event-one-strong", event_stage_match=2),
        _decision_row("event-two"),
    ]

    assert admitted_examples(
        candidates[:2], {"decision": {"candidates": decisions[:2]}}
    ) == []
    assert [row["id"] for row in admitted_examples(
        candidates, {"decision": {"candidates": decisions}}
    )] == ["event-one-strong", "event-two"]


def test_v7_one_qualified_opposite_event_clears_whole_case():
    candidates = [{
        "id": candidate_id,
        "historical_event_id": candidate_id,
        "relationship": "group",
        "context": [candidate_id],
        "human_reply": [candidate_id],
        "same_source_chat": False,
        "semantic_rank": index,
    } for index, candidate_id in enumerate(("a-one", "a-two", "b-one"), start=1)]
    decisions = [
        _decision_row("a-one"),
        _decision_row("a-two"),
        _decision_row("b-one", historical_action="defer"),
    ]

    assert admitted_examples(
        candidates, {"decision": {"candidates": decisions}}
    ) == []


def test_v7_ranking_uses_stage_then_addressee_then_chat_then_semantics_and_caps_ten():
    specs = [
        ("stage-first", 2, 0, False, 100),
        ("addressee-second", 1, 2, False, 100),
        ("chat-third", 1, 1, True, 100),
        ("semantic-fourth", 1, 1, False, 1),
        *[(f"tail-{index:02d}", 0, 0, False, 20 + index) for index in range(9)],
    ]
    candidates = [{
        "id": candidate_id,
        "historical_event_id": candidate_id,
        "relationship": "group",
        "context": [candidate_id],
        "human_reply": [candidate_id],
        "same_source_chat": same_chat,
        "semantic_rank": semantic_rank,
    } for candidate_id, _, _, same_chat, semantic_rank in specs]
    decisions = [
        _decision_row(
            candidate_id,
            event_stage_match=stage,
            addressee_match=addressee,
        )
        for candidate_id, stage, addressee, _, _ in specs
    ]

    selected = admitted_examples(
        candidates, {"decision": {"candidates": decisions}}
    )

    assert len(selected) == 10
    assert [row["id"] for row in selected[:4]] == [
        "stage-first", "addressee-second", "chat-third", "semantic-fourth"
    ]
    assert all(
        set(row) == {"id", "relationship", "context", "human_reply"}
        for row in selected
    )


def test_v7_matcher_discards_whole_case_when_any_history_batch_fails(monkeypatch):
    history_calls = 0

    def fake_run(command, input, **kwargs):
        nonlocal history_calls
        packet = json.loads(input.split("输入：\n", 1)[1])
        output_path = Path(command[command.index("--output-last-message") + 1])
        if "anonymous_options" in packet:
            payload = {
                "option_a_action": "answer_now",
                "option_b_action": "defer",
                "has_behavior_divergence": True,
                "reason": "立即回答与延后回答",
            }
        else:
            history_calls += 1
            if history_calls == 2:
                return SimpleNamespace(returncode=1, stdout="", stderr="batch failed")
            payload = {"candidates": [
                _match_row(row["candidate_id"])
                for row in packet["historical_candidates"]
            ]}
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "scripts.build_judge_dynamic_few_shots.subprocess.run", fake_run
    )
    candidates = [{
        "id": f"candidate-{index:02d}",
        "relationship": "private",
        "context": [str(index)],
        "human_reply": [str(index)],
    } for index in range(45)]

    with pytest.raises(RuntimeError, match="batch failed"):
        run_codex_matcher(
            "private", [], ["候选甲"], ["候选乙"], candidates, 180,
            batch_size=20,
        )
    assert history_calls == 2


def test_v7_matcher_prompt_contains_fixed_actions_without_answer_metadata():
    prompt = build_matcher_prompt(
        "group",
        [{"role": "other", "sender": "甲", "text": "当前问题"}],
        "accept",
        "refuse",
        [{
            "id": "history-one",
            "relationship": "group",
            "context": ["历史上下文"],
            "human_reply": ["历史真人回复"],
        }],
    )

    assert '"fixed_option_actions":{"A":"accept","B":"refuse"}' in prompt
    for forbidden in (
        '"source_case_id"', '"source_chat_id"', '"actual"', '"correct"'
    ):
        assert forbidden not in prompt


def test_v7_option_divergence_parser_rejects_self_contradiction():
    with pytest.raises(ValueError, match="contradicts"):
        parse_option_divergence_response(json.dumps({
            "option_a_action": "accept",
            "option_b_action": "refuse",
            "has_behavior_divergence": False,
            "reason": "标签不同却声称没有分歧",
        }))


def test_dynamic_manifest_reaches_prompt_without_answer_metadata_and_locks_resume(
    tmp_path, monkeypatch
):
    blind = {
        "batch_id": "balanced-160",
        "reference_sha256": "ref",
        "cases": [{
            "case_id": "one",
            "relationship": "private",
            "context_original": ["当前上下文"],
            "option_A": ["候选甲"],
            "option_B": ["候选乙"],
        }],
    }
    reference = {"reference_sha256": "ref", "facts": [], "examples": []}
    manifest = {
        "schema_version": 2,
        "kind": "judge_dynamic_few_shot_manifest",
        "blind_sha256": _json_sha256(blind),
        "reference_sha256": "ref",
        "pool_sha256": "pool",
        "cases": [{
            "case_id": "one",
            "relationship": "private",
            "examples": [
                {
                    "id": "example-0",
                    "relationship": "private",
                    "context": ["历史上下文-0"],
                    "human_reply": ["历史真人回复-0"],
                },
                {
                    "id": "example-1",
                    "relationship": "private",
                    "context": ["历史上下文-1"],
                    "human_reply": ["历史真人回复-1"],
                },
            ],
        }],
    }
    blind_path = tmp_path / "blind.json"
    reference_path = tmp_path / "reference.json"
    manifest_path = tmp_path / "manifest.json"
    predictions_path = tmp_path / "predictions.json"
    audit_path = tmp_path / "audit.json"
    _write_json(blind_path, blind)
    _write_json(reference_path, reference)
    _write_json(manifest_path, manifest)
    prompts = []

    monkeypatch.setattr(
        "scripts.evaluate_human_ai_judge.subprocess.run", _fake_codex(prompts)
    )
    progress = run_codex_judge_independently(
        blind_path,
        predictions_path,
        audit_path,
        reference_path,
        dynamic_few_shot_manifest_path=manifest_path,
    )

    assert progress == {"completed": 1, "total": 1}
    assert all(
        value in prompts[0]
        for value in ("历史上下文-0", "历史真人回复-0", "历史上下文-1", "历史真人回复-1")
    )
    for forbidden in ('"source_case_id"', '"source_chat_id"', '"actual"', '"correct"'):
        assert forbidden not in prompts[0]
    assert not any(
        marker in prompts[0]
        for marker in (
            '"human_option":"A"', '"human_option": "A"',
            '"human_option":"B"', '"human_option": "B"',
        )
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["calls"][0]["dynamic_few_shot_example_ids"] == ["example-0", "example-1"]
    assert audit["calls"][0]["answer_fields_in_request"] == []

    manifest["cases"][0]["examples"][0]["id"] = "different-example"
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="does not match"):
        run_codex_judge_independently(
            blind_path,
            predictions_path,
            audit_path,
            reference_path,
            dynamic_few_shot_manifest_path=manifest_path,
        )


def test_empty_dynamic_manifest_case_does_not_add_a_prompt_example(tmp_path, monkeypatch):
    blind = {
        "batch_id": "balanced-160",
        "reference_sha256": "ref",
        "cases": [{
            "case_id": "one",
            "relationship": "private",
            "context_original": ["当前上下文"],
            "option_A": ["候选甲"],
            "option_B": ["候选乙"],
        }],
    }
    reference = {"reference_sha256": "ref", "facts": [], "examples": []}
    manifest = {
        "schema_version": 2,
        "kind": "judge_dynamic_few_shot_manifest",
        "blind_sha256": _json_sha256(blind),
        "reference_sha256": "ref",
        "pool_sha256": "pool",
        "cases": [{"case_id": "one", "relationship": "private", "examples": []}],
    }
    blind_path = tmp_path / "blind.json"
    reference_path = tmp_path / "reference.json"
    manifest_path = tmp_path / "manifest.json"
    predictions_path = tmp_path / "predictions.json"
    audit_path = tmp_path / "audit.json"
    _write_json(blind_path, blind)
    _write_json(reference_path, reference)
    _write_json(manifest_path, manifest)
    prompts = []

    monkeypatch.setattr(
        "scripts.evaluate_human_ai_judge.subprocess.run", _fake_codex(prompts)
    )
    progress = run_codex_judge_independently(
        blind_path,
        predictions_path,
        audit_path,
        reference_path,
        dynamic_few_shot_manifest_path=manifest_path,
    )

    assert progress == {"completed": 1, "total": 1}
    assert "历史上下文" not in prompts[0]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert "dynamic_few_shot_example_ids" not in audit["calls"][0]


def test_replace_mode_removes_static_examples_for_matched_and_unmatched_cases(
    tmp_path, monkeypatch
):
    blind = {
        "batch_id": "balanced-160",
        "reference_sha256": "ref",
        "cases": [
            {
                "case_id": "matched",
                "relationship": "private",
                "context_original": ["命中题"],
                "option_A": ["候选甲"],
                "option_B": ["候选乙"],
            },
            {
                "case_id": "unmatched",
                "relationship": "group",
                "context_original": ["未命中题"],
                "option_A": ["候选丙"],
                "option_B": ["候选丁"],
            },
        ],
    }
    reference = {
        "reference_sha256": "ref",
        "facts": [],
        "examples": [{
            "relationship": "private",
            "context": ["静态案例上下文"],
            "human_reply": ["静态案例回复"],
        }],
    }
    manifest = {
        "schema_version": 2,
        "kind": "judge_dynamic_few_shot_manifest",
        "blind_sha256": _json_sha256(blind),
        "reference_sha256": "ref",
        "pool_sha256": "pool",
        "cases": [
            {
                "case_id": "matched",
                "relationship": "private",
                "examples": [{
                    "id": "dynamic-example",
                    "relationship": "private",
                    "context": ["动态案例上下文"],
                    "human_reply": ["动态案例回复"],
                }],
            },
            {"case_id": "unmatched", "relationship": "group", "examples": []},
        ],
    }
    paths = {
        "blind": tmp_path / "blind.json",
        "reference": tmp_path / "reference.json",
        "manifest": tmp_path / "manifest.json",
        "predictions": tmp_path / "predictions.json",
        "audit": tmp_path / "audit.json",
    }
    for key, payload in (
        ("blind", blind),
        ("reference", reference),
        ("manifest", manifest),
    ):
        _write_json(paths[key], payload)
    prompts = []
    monkeypatch.setattr(
        "scripts.evaluate_human_ai_judge.subprocess.run", _fake_codex(prompts)
    )

    progress = run_codex_judge_independently(
        paths["blind"],
        paths["predictions"],
        paths["audit"],
        paths["reference"],
        dynamic_few_shot_manifest_path=paths["manifest"],
        dynamic_few_shot_mode="replace",
    )

    assert progress == {"completed": 2, "total": 2}
    assert all("静态案例上下文" not in prompt for prompt in prompts)
    assert "动态案例上下文" in prompts[0]
    assert "动态案例上下文" not in prompts[1]
    predictions = json.loads(paths["predictions"].read_text(encoding="utf-8"))
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    assert predictions["dynamic_few_shot_mode"] == "replace"
    assert audit["dynamic_few_shot_mode"] == "replace"


def test_unmatched_cases_reuse_baseline_and_only_matched_cases_run(tmp_path, monkeypatch):
    blind = {
        "batch_id": "balanced-160",
        "reference_sha256": "ref",
        "cases": [
            {
                "case_id": "unmatched",
                "relationship": "private",
                "context_original": ["未命中上下文"],
                "option_A": ["候选甲"],
                "option_B": ["候选乙"],
            },
            {
                "case_id": "matched",
                "relationship": "group",
                "context_original": ["命中上下文"],
                "option_A": ["候选丙"],
                "option_B": ["候选丁"],
            },
        ],
    }
    reference = {
        "reference_sha256": "ref",
        "facts": [],
        "examples": [{
            "relationship": "group",
            "context": ["静态案例上下文"],
            "human_reply": ["静态案例回复"],
        }],
    }
    manifest = {
        "schema_version": 2,
        "kind": "judge_dynamic_few_shot_manifest",
        "blind_sha256": _json_sha256(blind),
        "reference_sha256": "ref",
        "pool_sha256": "pool",
        "cases": [
            {"case_id": "unmatched", "relationship": "private", "examples": []},
            {
                "case_id": "matched",
                "relationship": "group",
                "examples": [{
                    "id": "group-example",
                    "relationship": "group",
                    "context": ["历史群聊上下文"],
                    "human_reply": ["历史群聊真人回复"],
                }],
            },
        ],
    }
    baseline_predictions = {
        "schema_version": 3,
        "benchmark_kind": "independent_codex_human_ai_reply_discrimination_predictions",
        "invocation_id": "baseline-run",
        "batch_id": "balanced-160",
        "blind_sha256": _json_sha256(blind),
        "reference_sha256": "ref",
        "judge_profile_name": "builtin_default",
        "judge_profile_sha256": None,
        "cases": [
            {"case_id": "unmatched", "human_option": "B", "confidence": 0.7, "reason": "基线"},
            {"case_id": "matched", "human_option": "A", "confidence": 0.7, "reason": "基线"},
        ],
    }
    baseline_audit = {
        "schema_version": 1,
        "invocation_id": "baseline-run",
        "blind_sha256": _json_sha256(blind),
        "reference_sha256": "ref",
        "judge_profile_name": "builtin_default",
        "judge_profile_sha256": None,
        "calls": [
            {"index": 1, "case_id": "unmatched", "parsed_prediction": baseline_predictions["cases"][0]},
            {"index": 2, "case_id": "matched", "parsed_prediction": baseline_predictions["cases"][1]},
        ],
    }
    paths = {
        "blind": tmp_path / "blind.json",
        "reference": tmp_path / "reference.json",
        "manifest": tmp_path / "manifest.json",
        "baseline_predictions": tmp_path / "baseline_predictions.json",
        "baseline_audit": tmp_path / "baseline_audit.json",
        "predictions": tmp_path / "predictions.json",
        "audit": tmp_path / "audit.json",
    }
    for key, payload in (
        ("blind", blind),
        ("reference", reference),
        ("manifest", manifest),
        ("baseline_predictions", baseline_predictions),
        ("baseline_audit", baseline_audit),
    ):
        _write_json(paths[key], payload)
    prompts = []

    monkeypatch.setattr(
        "scripts.evaluate_human_ai_judge.subprocess.run", _fake_codex(prompts)
    )
    progress = run_codex_judge_independently(
        paths["blind"],
        paths["predictions"],
        paths["audit"],
        paths["reference"],
        dynamic_few_shot_manifest_path=paths["manifest"],
        baseline_predictions_path=paths["baseline_predictions"],
        baseline_audit_path=paths["baseline_audit"],
        dynamic_few_shot_mode="replace_on_match",
    )

    assert progress == {"completed": 2, "total": 2}
    assert len(prompts) == 1
    assert "命中上下文" in prompts[0]
    assert "历史群聊真人回复" in prompts[0]
    assert "静态案例上下文" not in prompts[0]
    predictions = json.loads(paths["predictions"].read_text(encoding="utf-8"))
    assert [row["case_id"] for row in predictions["cases"]] == ["unmatched", "matched"]
    assert predictions["cases"][0] == baseline_predictions["cases"][0]
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    assert audit["calls"][0]["reused_from_baseline"] is True
    assert audit["calls"][0]["baseline_invocation_id"] == "baseline-run"
    assert audit["calls"][1]["dynamic_few_shot_example_ids"] == ["group-example"]


def test_replace_on_match_reuses_unchanged_matched_case_from_replace_candidate(
    tmp_path, monkeypatch
):
    blind = {
        "batch_id": "balanced-160",
        "reference_sha256": "ref",
        "cases": [
            {
                "case_id": case_id,
                "relationship": "private",
                "context_original": [case_id],
                "option_A": ["候选甲"],
                "option_B": ["候选乙"],
            }
            for case_id in ("unmatched", "unchanged", "changed")
        ],
    }
    reference = {"reference_sha256": "ref", "facts": [], "examples": []}

    def example(example_id):
        return {
            "id": example_id,
            "relationship": "private",
            "context": [f"上下文-{example_id}"],
            "human_reply": [f"回复-{example_id}"],
        }

    prior_manifest = {
        "schema_version": 2,
        "kind": "judge_dynamic_few_shot_manifest",
        "blind_sha256": _json_sha256(blind),
        "reference_sha256": "ref",
        "pool_sha256": "pool",
        "cases": [
            {"case_id": "unmatched", "relationship": "private", "examples": []},
            {"case_id": "unchanged", "relationship": "private", "examples": [example("same")]},
            {"case_id": "changed", "relationship": "private", "examples": [example("old")]},
        ],
    }
    manifest = json.loads(json.dumps(prior_manifest))
    manifest["cases"][2]["examples"] = [example("new")]
    prior_manifest_path = tmp_path / "prior_manifest.json"
    _write_json(prior_manifest_path, prior_manifest)
    prior_manifest_sha = hashlib.sha256(prior_manifest_path.read_bytes()).hexdigest()
    prior_predictions = {
        "schema_version": 3,
        "benchmark_kind": "independent_codex_human_ai_reply_discrimination_predictions",
        "invocation_id": "prior-run",
        "batch_id": "balanced-160",
        "blind_sha256": _json_sha256(blind),
        "reference_sha256": "ref",
        "judge_profile_name": "builtin_default",
        "judge_profile_sha256": None,
        "dynamic_few_shot_manifest_sha256": prior_manifest_sha,
        "dynamic_few_shot_mode": "replace",
        "cases": [
            {"case_id": case_id, "human_option": "B", "confidence": 0.7, "reason": "旧结果"}
            for case_id in ("unmatched", "unchanged", "changed")
        ],
    }
    prior_audit = {
        "schema_version": 1,
        "invocation_id": "prior-run",
        "blind_sha256": _json_sha256(blind),
        "reference_sha256": "ref",
        "judge_profile_name": "builtin_default",
        "judge_profile_sha256": None,
        "dynamic_few_shot_manifest_sha256": prior_manifest_sha,
        "dynamic_few_shot_mode": "replace",
        "calls": [
            {"index": index, "case_id": case_id, "parsed_prediction": prior_predictions["cases"][index - 1]}
            for index, case_id in enumerate(("unmatched", "unchanged", "changed"), start=1)
        ],
    }
    baseline_predictions = {
        **{key: value for key, value in prior_predictions.items() if not key.startswith("dynamic_few_shot_")},
        "invocation_id": "baseline-run",
        "cases": [
            {"case_id": case_id, "human_option": "A", "confidence": 0.6, "reason": "基线"}
            for case_id in ("unmatched", "unchanged", "changed")
        ],
    }
    baseline_audit = {
        "schema_version": 1,
        "invocation_id": "baseline-run",
        "blind_sha256": _json_sha256(blind),
        "reference_sha256": "ref",
        "judge_profile_name": "builtin_default",
        "judge_profile_sha256": None,
        "calls": [
            {"index": index, "case_id": case_id, "parsed_prediction": baseline_predictions["cases"][index - 1]}
            for index, case_id in enumerate(("unmatched", "unchanged", "changed"), start=1)
        ],
    }
    paths = {
        "blind": tmp_path / "blind.json",
        "reference": tmp_path / "reference.json",
        "manifest": tmp_path / "manifest.json",
        "prior_predictions": tmp_path / "prior_predictions.json",
        "prior_audit": tmp_path / "prior_audit.json",
        "baseline_predictions": tmp_path / "baseline_predictions.json",
        "baseline_audit": tmp_path / "baseline_audit.json",
        "predictions": tmp_path / "predictions.json",
        "audit": tmp_path / "audit.json",
    }
    for key, payload in (
        ("blind", blind),
        ("reference", reference),
        ("manifest", manifest),
        ("prior_predictions", prior_predictions),
        ("prior_audit", prior_audit),
        ("baseline_predictions", baseline_predictions),
        ("baseline_audit", baseline_audit),
    ):
        _write_json(paths[key], payload)
    prompts = []
    monkeypatch.setattr(
        "scripts.evaluate_human_ai_judge.subprocess.run", _fake_codex(prompts)
    )

    progress = run_codex_judge_independently(
        paths["blind"],
        paths["predictions"],
        paths["audit"],
        paths["reference"],
        dynamic_few_shot_manifest_path=paths["manifest"],
        baseline_predictions_path=paths["baseline_predictions"],
        baseline_audit_path=paths["baseline_audit"],
        dynamic_few_shot_mode="replace_on_match",
        prior_dynamic_few_shot_manifest_path=prior_manifest_path,
        prior_predictions_path=paths["prior_predictions"],
        prior_audit_path=paths["prior_audit"],
    )

    assert progress == {"completed": 3, "total": 3}
    assert len(prompts) == 1
    assert "上下文-new" in prompts[0]
    predictions = json.loads(paths["predictions"].read_text(encoding="utf-8"))
    assert predictions["cases"][0] == baseline_predictions["cases"][0]
    assert predictions["cases"][1] == prior_predictions["cases"][1]
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    assert audit["calls"][0]["reused_from_baseline"] is True
    assert audit["calls"][1]["reused_from_prior_candidate"] is True
    assert audit["calls"][1]["prior_invocation_id"] == "prior-run"
