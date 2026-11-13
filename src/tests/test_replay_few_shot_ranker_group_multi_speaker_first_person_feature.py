import json

from scripts.build_few_shot_ranker_samples import build
from scripts.replay_few_shot_ranker_group_multi_speaker_first_person_feature import replay


def _target(case_id, relationship, chat_id, context, reply, context_messages):
    return {
        "case_id": case_id,
        "relationship": relationship,
        "chat_id": chat_id,
        "timestamp": 1_700_000_000,
        "context": context,
        "context_messages": context_messages,
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


def test_replay_adds_only_multi_speaker_first_person_cross(tmp_path):
    manifest_path = tmp_path / "judge.json"
    pool_path = tmp_path / "pool.jsonl"
    config_path = tmp_path / "c0.json"
    manifest_path.write_text(json.dumps({"splits": {"train": [
        _target(
            "group", "group", "g-target", ["甲问？", "乙也问"], ["真人群聊答案"],
            [{"sender": "甲", "text": "甲问？"}, {"sender": "乙", "text": "乙也问"}],
        ),
    ]}}, ensure_ascii=False), encoding="utf-8")
    pool_path.write_text(json.dumps(
        _pool_row("g1", "group", "g1", ["早些时候群聊问题？"], ["我来看看"]),
        ensure_ascii=False,
    ) + "\n", encoding="utf-8")
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
    # Match the frozen v5 schema: the newer overlap feature was intentionally
    # absent, but the two facts needed by this replay were already present.
    for row in source_rows:
        row["local_features"].pop("candidate_reply_overlap_current_ge2")
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_rows),
        encoding="utf-8",
    )

    report = replay(source_path, tmp_path / "replayed")
    replayed = json.loads((tmp_path / "replayed" / "samples.jsonl").read_text(encoding="utf-8"))

    assert report["counts"]["samples"] == 1
    assert report["counts"]["context_multiple_other_speakers__candidate_reply_has_first_person_true"] == {"group": 1}
    assert replayed["local_features"]["context_multiple_other_speakers"] is True
    assert replayed["local_features"]["candidate_reply_has_first_person"] is True
    assert replayed["local_features"]["context_multiple_other_speakers__candidate_reply_has_first_person"] is True
