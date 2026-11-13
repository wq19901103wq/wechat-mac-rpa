import json

from scripts.build_few_shot_ranker_samples import build
from scripts.replay_few_shot_ranker_group_trace import replay


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


def test_replay_adds_group_trace_without_requiring_private_trace(tmp_path):
    manifest_path = tmp_path / "judge.json"
    pool_path = tmp_path / "pool.jsonl"
    config_path = tmp_path / "c0.json"
    manifest_path.write_text(json.dumps({"splits": {"train": [
        _target("private", "private", "p-target", ["私聊问题？"], ["真人私聊答案"]),
        _target("group", "group", "g-target", ["群聊问题？"], ["真人群聊答案"]),
    ]}}, ensure_ascii=False), encoding="utf-8")
    pool_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in [
        _pool_row("p1", "acquaintance", "p1", ["早些时候私聊问题？"], ["回复一"]),
        _pool_row("p2", "acquaintance", "p2", ["另一个私聊？"], ["回复二"]),
        _pool_row("g1", "group", "g1", ["早些时候群聊问题？"], ["回复三"]),
        _pool_row("g2", "group", "g2", ["另一个群聊？"], ["回复四"]),
    ]), encoding="utf-8")
    config_path.write_text(json.dumps({"features": {
        "enable_precise_situation_matching": True,
        "enable_information_gap_response_move": True,
        "prefer_latest_turn_form": True,
        "group_prefer_latest_turn_form": False,
    }}), encoding="utf-8")
    built_dir = tmp_path / "built"
    build(manifest_path, pool_path, built_dir, c0_config_path=config_path, candidates_per_target=2)
    source_rows = [
        json.loads(line)
        for line in (built_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in source_rows:
        row.pop("production_retrieval_features")
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_rows),
        encoding="utf-8",
    )

    report = replay(source_path, pool_path, tmp_path / "replayed", c0_config_path=config_path)
    replayed_rows = [
        json.loads(line)
        for line in (tmp_path / "replayed" / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert report["counts"] == {
        "source_samples": 4,
        "group_targets_replayed": 1,
        "group_samples_with_trace": 2,
        "private_samples_copied_unchanged": 2,
    }
    assert all(
        "production_retrieval_features" in row
        for row in replayed_rows
        if row["target"]["relationship"] == "group"
    )
    assert all(
        "production_retrieval_features" not in row
        for row in replayed_rows
        if row["target"]["relationship"] == "private"
    )


def test_replay_adds_private_trace_without_requiring_group_trace(tmp_path):
    manifest_path = tmp_path / "judge.json"
    pool_path = tmp_path / "pool.jsonl"
    config_path = tmp_path / "c0.json"
    manifest_path.write_text(json.dumps({"splits": {"train": [
        _target("private", "private", "p-target", ["私聊问题？"], ["真人私聊答案"]),
        _target("group", "group", "g-target", ["群聊问题？"], ["真人群聊答案"]),
    ]}}, ensure_ascii=False), encoding="utf-8")
    pool_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in [
        _pool_row("p1", "acquaintance", "p1", ["早些时候私聊问题？"], ["回复一"]),
        _pool_row("p2", "acquaintance", "p2", ["另一个私聊？"], ["回复二"]),
        _pool_row("g1", "group", "g1", ["早些时候群聊问题？"], ["回复三"]),
        _pool_row("g2", "group", "g2", ["另一个群聊？"], ["回复四"]),
    ]), encoding="utf-8")
    config_path.write_text(json.dumps({"features": {
        "enable_precise_situation_matching": True,
        "enable_information_gap_response_move": True,
        "prefer_latest_turn_form": True,
        "group_prefer_latest_turn_form": False,
    }}), encoding="utf-8")
    built_dir = tmp_path / "built"
    build(manifest_path, pool_path, built_dir, c0_config_path=config_path, candidates_per_target=2)
    source_rows = [
        json.loads(line)
        for line in (built_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in source_rows:
        row.pop("production_retrieval_features")
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_rows),
        encoding="utf-8",
    )

    report = replay(
        source_path,
        pool_path,
        tmp_path / "replayed",
        c0_config_path=config_path,
        relationship="private",
    )
    replayed_rows = [
        json.loads(line)
        for line in (tmp_path / "replayed" / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert report["counts"] == {
        "source_samples": 4,
        "private_targets_replayed": 1,
        "private_samples_with_trace": 2,
        "group_samples_copied_unchanged": 2,
    }
    assert all(
        "production_retrieval_features" in row
        for row in replayed_rows
        if row["target"]["relationship"] == "private"
    )
    assert all(
        "production_retrieval_features" not in row
        for row in replayed_rows
        if row["target"]["relationship"] == "group"
    )
