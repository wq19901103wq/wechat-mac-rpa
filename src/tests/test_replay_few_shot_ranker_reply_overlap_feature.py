import json

from scripts.build_few_shot_ranker_samples import build
from scripts.replay_few_shot_ranker_reply_overlap_feature import replay


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


def _pool_row(row_id, relationship, chat_id, context, reply):
    return {
        "id": row_id,
        "relationship": relationship,
        "chat_id": chat_id,
        "timestamp": 1_600_000_000,
        "context": context,
        "reply": reply,
        "situation_tags": ["question"],
        "source_provenance": "before_automation_cutoff",
    }


def test_replay_preserves_v3_features_and_allows_private_rows_without_trace(tmp_path):
    manifest_path = tmp_path / "judge.json"
    pool_path = tmp_path / "pool.jsonl"
    config_path = tmp_path / "c0.json"
    manifest_path.write_text(json.dumps({"splits": {"train": [
        _target("private", "private", "p-target", ["私聊问题？"], ["真人私聊答案"]),
        _target("group", "group", "g-target", ["群聊问题？"], ["真人群聊答案"]),
    ]}}, ensure_ascii=False), encoding="utf-8")
    pool_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in [
        _pool_row("p1", "acquaintance", "p1", ["早些时候私聊问题？"], ["回复一"]),
        _pool_row("g1", "group", "g1", ["早些时候群聊问题？"], ["回复二"]),
    ]), encoding="utf-8")
    config_path.write_text(json.dumps({"features": {
        "enable_precise_situation_matching": True,
        "enable_information_gap_response_move": True,
        "prefer_latest_turn_form": True,
        "group_prefer_latest_turn_form": False,
    }}), encoding="utf-8")
    built_dir = tmp_path / "built"
    build(manifest_path, pool_path, built_dir, c0_config_path=config_path, candidates_per_target=1)
    source_rows = [
        json.loads(line)
        for line in (built_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in source_rows:
        row["local_features"].pop("candidate_reply_has_first_person")
        row["local_features"].pop("candidate_reply_overlap_current_ge2")
        if row["target"]["relationship"] == "private":
            row.pop("production_retrieval_features")
    source_path = tmp_path / "v3-source.jsonl"
    source_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_rows),
        encoding="utf-8",
    )

    report = replay(source_path, tmp_path / "replayed")
    replayed_rows = [
        json.loads(line)
        for line in (tmp_path / "replayed" / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert report["counts"]["samples"] == 2
    assert all(
        "candidate_reply_overlap_current_ge2" in row["local_features"]
        and "candidate_reply_has_first_person" not in row["local_features"]
        for row in replayed_rows
    )
    assert any(
        "production_retrieval_features" not in row
        for row in replayed_rows
        if row["target"]["relationship"] == "private"
    )
    assert all(
        "production_retrieval_features" in row
        for row in replayed_rows
        if row["target"]["relationship"] == "group"
    )
