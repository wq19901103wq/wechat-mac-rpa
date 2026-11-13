import json

import scripts.build_few_shot_ranker_samples as builder
from scripts.build_few_shot_ranker_samples import build


def _target(case_id, relationship, chat_id, context, reply):
    return {
        "case_id": case_id,
        "relationship": relationship,
        "chat_id": chat_id,
        "timestamp": 1_700_000_000,
        "context": context,
        "context_messages": [{"sender": "甲", "text": text} for text in context],
        "replies": reply,
    }


def _pool_row(row_id, relationship, chat_id, context, reply, timestamp=1_600_000_000):
    return {
        "id": row_id,
        "relationship": relationship,
        "chat_id": chat_id,
        "timestamp": timestamp,
        "context": context,
        "reply": reply,
        "situation_tags": ["question"] if "？" in "".join(context) else [],
        "source_provenance": "before_automation_cutoff",
    }


def test_builds_single_shot_samples_and_keeps_safe_same_chat_history(tmp_path):
    manifest = {"splits": {"train": [
        _target("private-target", "private", "same-chat", ["现在这件事到底怎么处理？"], ["真人答案"]),
        _target("group-target", "group", "group-target", ["群里有人吗？"], ["真人群答案"]),
    ]}}
    pool = [
        _pool_row("same-chat-safe", "acquaintance", "same-chat", ["早些时候的独立问题？"], ["安全历史回复"]),
        _pool_row("same-chat-overlap", "acquaintance", "same-chat", ["现在这件事到底怎么处理？"], ["不应出现"]),
        _pool_row("same-chat-future", "acquaintance", "same-chat", ["之后的独立问题？"], ["不应出现"], 1_800_000_000),
        _pool_row("answer-leak", "acquaintance", "another-chat", ["完全不同的上下文"], ["真人答案"]),
        _pool_row("private-other", "acquaintance", "other-chat", ["另外一个问题？"], ["另一个回复"]),
        _pool_row("group-one", "group", "group-one", ["群里有人吗？"], ["我在"]),
        _pool_row("group-two", "group", "group-two", ["另一个群聊问题？"], ["群聊回复"]),
    ]
    manifest_path = tmp_path / "judge.json"
    pool_path = tmp_path / "pool.jsonl"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    pool_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pool), encoding="utf-8")

    report = build(manifest_path, pool_path, tmp_path / "out", candidates_per_target=2)
    samples = [json.loads(line) for line in (tmp_path / "out" / "samples.jsonl").read_text(encoding="utf-8").splitlines()]

    assert report["protocol"]["single_few_shot_only"] is True
    assert report["protocol"]["candidate_llm_calls"] == 0
    assert len(samples) == 4
    private_rows = [row for row in samples if row["target"]["case_id"] == "private-target"]
    assert {row["candidate"]["id"] for row in private_rows} == {"same-chat-safe", "private-other"}
    assert all(row["generation"]["few_shot_count"] == 1 for row in samples)
    assert all(row["generation"]["static_few_shot_count"] == 0 for row in samples)
    assert all(row["label"] is None for row in samples)
    assert all(isinstance(value, bool) for row in samples for value in row["local_features"].values())
    assert "真人答案" not in (tmp_path / "out" / "samples.jsonl").read_text(encoding="utf-8")


def test_resume_adds_new_targets_without_rewriting_previous_samples(tmp_path):
    manifest = {"splits": {"train": [
        _target("private", "private", "private-chat", ["私聊问题？"], ["真人私聊答案"]),
        _target("group", "group", "group-chat", ["群聊问题？"], ["真人群聊答案"]),
    ]}}
    pool = [
        _pool_row("p1", "acquaintance", "p1", ["私聊问题？x"], ["回复一"]),
        _pool_row("p2", "acquaintance", "p2", ["别的私聊？"], ["回复二"]),
        _pool_row("g1", "group", "g1", ["群聊问题？x"], ["回复三"]),
        _pool_row("g2", "group", "g2", ["别的群聊？"], ["回复四"]),
    ]
    manifest_path = tmp_path / "judge.json"
    pool_path = tmp_path / "pool.jsonl"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    pool_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pool), encoding="utf-8")

    build(manifest_path, pool_path, tmp_path / "out", candidates_per_target=1, limit=1)
    first = (tmp_path / "out" / "samples.jsonl").read_text(encoding="utf-8")
    report = build(
        manifest_path, pool_path, tmp_path / "out", candidates_per_target=1, limit=2, resume=True,
    )
    final = (tmp_path / "out" / "samples.jsonl").read_text(encoding="utf-8")

    assert final.startswith(first)
    assert report["counts"]["targets"] == 2


def test_does_not_cap_safe_same_chat_history(tmp_path):
    manifest = {"splits": {"train": [
        _target("target", "private", "same-chat", ["现在怎么办？"], ["真人答案"]),
    ]}}
    pool = [
        _pool_row(f"same-{index}", "acquaintance", "same-chat", [f"早些时候的问题{index}？"], [f"历史回复{index}"])
        for index in range(3)
    ]
    manifest_path = tmp_path / "judge.json"
    pool_path = tmp_path / "pool.jsonl"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    pool_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pool), encoding="utf-8")

    build(manifest_path, pool_path, tmp_path / "out", candidates_per_target=3)
    samples = [json.loads(line) for line in (tmp_path / "out" / "samples.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len(samples) == 3
    assert all(row["local_features"]["candidate_same_chat"] for row in samples)


def test_candidate_reply_first_person_is_a_local_boolean_feature():
    target = builder._target_view(_target(
        "target", "group", "target-chat", ["这个事情怎么处理？"], ["真人答案"],
    ))
    first_person = builder._candidate_view(_pool_row(
        "first-person", "group", "history-chat", ["你来看看这个"], ["我回一下"],
    ))
    no_first_person = builder._candidate_view(_pool_row(
        "no-first-person", "group", "history-chat", ["你来看看这个"], ["帮忙回一下"],
    ))

    assert builder.local_boolean_features(target, first_person)[
        "candidate_reply_has_first_person"
    ] is True
    assert builder.local_boolean_features(target, no_first_person)[
        "candidate_reply_has_first_person"
    ] is False


def test_candidate_reply_overlap_with_current_context_is_a_local_boolean_feature():
    target = builder._target_view(_target(
        "target", "group", "target-chat", ["这个事情怎么处理？"], ["真人答案"],
    ))
    matching_reply = builder._candidate_view(_pool_row(
        "matching-reply", "group", "history-chat", ["之前的话题"], ["这个怎么处理"],
    ))
    one_term_reply = builder._candidate_view(_pool_row(
        "one-term-reply", "group", "history-chat", ["之前的话题"], ["处"],
    ))

    assert builder.local_boolean_features(target, matching_reply)[
        "candidate_reply_overlap_current_ge2"
    ] is True
    assert builder.local_boolean_features(target, one_term_reply)[
        "candidate_reply_overlap_current_ge2"
    ] is False


def test_keeps_fewer_than_requested_candidates_when_only_safe_history_is_available(tmp_path):
    manifest = {"splits": {"train": [
        _target("target", "private", "target-chat", ["现在怎么办？"], ["真人答案"]),
    ]}}
    pool = [
        _pool_row(f"candidate-{index}", "acquaintance", f"chat-{index}", [f"早些时候的问题{index}？"], [f"历史回复{index}"])
        for index in range(2)
    ]
    manifest_path = tmp_path / "judge.json"
    pool_path = tmp_path / "pool.jsonl"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    pool_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pool), encoding="utf-8")

    report = build(manifest_path, pool_path, tmp_path / "out", candidates_per_target=3)

    assert report["counts"]["samples"] == 2
    assert report["counts"]["samples_per_target"] == {"2": 1}


def test_replaces_targets_without_safe_history_to_keep_requested_target_count(tmp_path):
    too_early = _target("too-early", "private", "too-early", ["没有更早的历史？"], ["真人答案"])
    too_early["timestamp"] = 1_500_000_000
    manifest = {"splits": {"train": [
        too_early,
        _target("safe-private", "private", "safe-private", ["私聊问题？"], ["真人私聊答案"]),
        _target("safe-group", "group", "safe-group", ["群聊问题？"], ["真人群聊答案"]),
    ]}}
    pool = [
        _pool_row("private", "acquaintance", "private", ["早些时候私聊？"], ["回复"], 1_600_000_000),
        _pool_row("group", "group", "group", ["早些时候群聊？"], ["回复"], 1_600_000_000),
    ]
    manifest_path = tmp_path / "judge.json"
    pool_path = tmp_path / "pool.jsonl"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    pool_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pool), encoding="utf-8")

    report = build(manifest_path, pool_path, tmp_path / "out", candidates_per_target=1, limit=2)

    assert report["counts"]["targets"] == 2
    assert report["counts"]["exclusions_for_new_targets"]["target_without_safe_historical_candidate"] == 1


def test_exact_relationship_limits_exclude_prior_ranking_targets(tmp_path):
    manifest = {"splits": {"train": [
        _target("private-trained", "private", "p0", ["私聊0？"], ["真人答案"]),
        _target("private-keep", "private", "p1", ["私聊1？"], ["真人答案"]),
        _target("private-extra", "private", "p2", ["私聊2？"], ["真人答案"]),
        _target("group-trained", "group", "g0", ["群聊0？"], ["真人答案"]),
        _target("group-keep", "group", "g1", ["群聊1？"], ["真人答案"]),
        _target("group-extra", "group", "g2", ["群聊2？"], ["真人答案"]),
    ]}}
    pool = [
        _pool_row("private-history", "acquaintance", "hp", ["历史私聊？"], ["回复"]),
        _pool_row("group-history", "group", "hg", ["历史群聊？"], ["回复"]),
    ]
    manifest_path = tmp_path / "judge.json"
    pool_path = tmp_path / "pool.jsonl"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    pool_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pool), encoding="utf-8")

    report = build(
        manifest_path, pool_path, tmp_path / "out", candidates_per_target=1,
        target_limits={"private": 1, "group": 2},
        exclude_target_ids={"private-trained", "group-trained"},
    )
    samples = [json.loads(line) for line in (tmp_path / "out" / "samples.jsonl").read_text(encoding="utf-8").splitlines()]

    assert report["protocol"]["requested_target_limits"] == {"private": 1, "group": 2}
    assert report["counts"]["relationship"] == {"group": 2, "private": 1}
    assert {row["target"]["case_id"] for row in samples}.isdisjoint({"private-trained", "group-trained"})


def test_uses_production_retriever_order_and_c0_turn_form_flags(tmp_path, monkeypatch):
    manifest = {"splits": {"train": [
        _target("private", "private", "private-chat", ["私聊问题？"], ["真人私聊答案"]),
        _target("group", "group", "group-chat", ["群聊问题？"], ["真人群聊答案"]),
    ]}}
    pool = [
        _pool_row("private-first", "acquaintance", "p1", ["私聊问题？x"], ["回复一"]),
        _pool_row("private-second", "acquaintance", "p2", ["另一个私聊？"], ["回复二"]),
        _pool_row("group-first", "group", "g1", ["群聊问题？x"], ["回复三"]),
        _pool_row("group-second", "group", "g2", ["另一个群聊？"], ["回复四"]),
    ]
    manifest_path = tmp_path / "judge.json"
    pool_path = tmp_path / "pool.jsonl"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    pool_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pool), encoding="utf-8")

    calls = []

    class SpyRetriever:
        def __init__(self, path, **kwargs):
            self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.kwargs = kwargs

        def retrieve(self, **kwargs):
            calls.append({"init": self.kwargs, "retrieve": kwargs})
            relationship = kwargs["relationship"]
            rows = [row for row in self.rows if row["relationship"] == relationship]
            excluded = kwargs.get("exclude_ids") or set()
            return [
                {
                    **row,
                    "retrieval_features": {"test_retrieval_score": 1.0},
                }
                for row in reversed(rows)
                if row["id"] not in excluded
            ][:kwargs["limit"]]

    monkeypatch.setattr(builder, "PersonaFewShotRetriever", SpyRetriever)
    build(manifest_path, pool_path, tmp_path / "out", candidates_per_target=2)
    samples = [json.loads(line) for line in (tmp_path / "out" / "samples.jsonl").read_text(encoding="utf-8").splitlines()]

    assert [row["candidate"]["id"] for row in samples if row["target"]["case_id"] == "private"] == [
        "private-second", "private-first",
    ]
    assert [row["candidate"]["id"] for row in samples if row["target"]["case_id"] == "group"] == [
        "group-second", "group-first",
    ]
    assert calls[0]["init"]["prefer_latest_turn_form"] is True
    assert calls[1]["init"]["prefer_latest_turn_form"] is False
