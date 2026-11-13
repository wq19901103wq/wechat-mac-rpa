import json

from scripts.build_few_shot_ranker_samples import build
from scripts.replay_few_shot_ranker_group_question_turn_match_feature import (
    FEATURE_NAME,
    replay,
)


def test_replay_adds_only_question_turn_compatibility_cross(tmp_path):
    manifest_path = tmp_path / "judge.json"
    pool_path = tmp_path / "pool.jsonl"
    config_path = tmp_path / "c0.json"
    manifest_path.write_text(json.dumps({"splits": {"train": [{
        "case_id": "group",
        "relationship": "group",
        "chat_id": "g-target",
        "timestamp": 1_700_000_000,
        "context": ["是不是现在处理？"],
        "context_messages": [{"sender": "甲", "text": "是不是现在处理？"}],
        "replies": ["真人群聊答案"],
    }]}}, ensure_ascii=False), encoding="utf-8")
    pool_path.write_text(json.dumps({
        "id": "g1",
        "relationship": "group",
        "chat_id": "g1",
        "timestamp": 1_600_000_000,
        "context": ["之前为什么这样处理？"],
        "reply": ["我来看看"],
        "situation_tags": ["question"],
        "source_provenance": "before_automation_cutoff",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
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
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_rows),
        encoding="utf-8",
    )

    report = replay(source_path, tmp_path / "replayed")
    replayed = json.loads((tmp_path / "replayed" / "samples.jsonl").read_text(encoding="utf-8"))

    assert report["counts"][f"{FEATURE_NAME}_true"] == {"group": 1}
    assert replayed["local_features"][FEATURE_NAME] is True
    assert replayed["local_features"]["context_latest_is_question"] is True
    assert replayed["local_features"]["same_latest_turn_form"] is False
