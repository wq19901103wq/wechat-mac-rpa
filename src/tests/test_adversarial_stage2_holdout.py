import hashlib
import json
import stat
from collections import Counter

from scripts import build_adversarial_stage2_holdout as stage2


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_stage2_holdout_meets_frozen_coverage_and_is_private():
    manifest = _load(stage2.MANIFEST_PATH)
    public = _load(stage2.PUBLIC_SUMMARY_PATH)
    cases = manifest["cases"]
    coverage = manifest["coverage"]

    assert manifest["status"] == "sealed_pending_user_confirmation"
    assert coverage == {
        "total": 240,
        "group_chat": 160,
        "private_chat": 80,
        "complex_large_group": 102,
        "distinct_chats": 178,
        "distinct_event_clusters": 240,
        "correct_common_prosperity_with_luxixi": 20,
        "youzi_group": 20,
    }
    assert len(cases) == 240
    assert len({case["internal_id"] for case in cases}) == 240
    assert len({case["event_cluster"] for case in cases}) == 240
    assert all(case["identity_evidence"] and case["context"] for case in cases)
    assert all(
        case["context_messages"]
        and [row["content"] for row in case["context_messages"]] == case["context"]
        and all(row["speaker"] == "self" or row["speaker"].startswith("participant_") for row in case["context_messages"])
        for case in cases
    )

    ordinary_counts = Counter(
        case["chat_id"] for case in cases if not case.get("specified_group")
    )
    assert max(ordinary_counts.values()) <= stage2.ORDINARY_CHAT_CAP
    assert stat.S_IMODE(stage2.MANIFEST_PATH.stat().st_mode) == 0o600
    assert stat.S_IMODE(stage2.PUBLIC_SUMMARY_PATH.stat().st_mode) == 0o600
    assert hashlib.sha256(stage2.MANIFEST_PATH.read_bytes()).hexdigest() == public[
        "manifest_sha256"
    ]


def test_stage2_holdout_does_not_reuse_stage1_events_or_replies():
    stage1_manifest = _load(stage2.STAGE1_MANIFEST)
    stage2_manifest = _load(stage2.MANIFEST_PATH)
    stage1_events = {case["event_cluster"] for case in stage1_manifest["cases"]}
    stage2_events = {case["event_cluster"] for case in stage2_manifest["cases"]}
    stage1_replies = [case["reply"] for case in stage1_manifest["cases"]]

    assert stage1_events.isdisjoint(stage2_events)
    assert not any(
        stage2._near_duplicate(case["reply"], old_reply)
        for case in stage2_manifest["cases"]
        for old_reply in stage1_replies
    )


def test_stage2_public_summary_contains_no_hidden_material():
    manifest = _load(stage2.MANIFEST_PATH)
    public_text = stage2.PUBLIC_SUMMARY_PATH.read_text(encoding="utf-8")

    for case in manifest["cases"]:
        assert case["internal_id"] not in public_text
        assert case["event_cluster"] not in public_text
        assert case["chat_id"] not in public_text
        assert case["reply"] not in public_text
        assert not any(
            message in public_text
            for message in case["context"]
            if len(stage2._normalize(message)) >= 4
        )


def test_target_resolution_work_files_do_not_self_pollute_candidates(tmp_path, monkeypatch):
    work_dir = tmp_path / "target_resolution"
    manifest_path = work_dir / "manifest.json"
    labels_path = work_dir / "strata_labels.json"
    audit_path = work_dir / "strata_curation_audit.json"
    stage2_path = tmp_path / "stage2.json"
    stage2_path.write_text('{"cases": []}', encoding="utf-8")
    candidate = {
        "internal_id": "candidate-1",
        "event_cluster": "event-1",
        "chat_id": "chat-1",
        "chat_type": "group",
        "complex_large_group": True,
        "context_messages": [
            {"speaker": "participant_1", "content": "第一句"},
            {"speaker": "participant_2", "content": "第二句"},
        ],
        "reply": "候选真人回复",
    }

    monkeypatch.setattr(stage2, "EXPOSURE_ROOTS", (work_dir,))
    monkeypatch.setattr(stage2, "ROOT", tmp_path)
    monkeypatch.setattr(stage2, "TARGET_RESOLUTION_DIR", work_dir)
    monkeypatch.setattr(stage2, "TARGET_RESOLUTION_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(stage2, "MANIFEST_PATH", stage2_path)
    monkeypatch.setattr(
        stage2,
        "IGNORED_EXPOSURE_PATHS",
        {manifest_path.resolve(), labels_path.resolve(), audit_path.resolve()},
    )
    monkeypatch.setattr(stage2, "_stage1_replies", lambda: [])
    monkeypatch.setattr(
        stage2,
        "_ordinary_candidates",
        lambda exposed: []
        if stage2._fingerprint(candidate["reply"]) in exposed
        else [candidate],
    )
    monkeypatch.setattr(
        stage2,
        "_filter_semantic",
        lambda rows, _old, deduplicate_within=False: rows,
    )

    first = stage2.build_target_resolution_candidates()
    first_bytes = manifest_path.read_bytes()
    labels_path.write_text(json.dumps({"reply": candidate["reply"]}), encoding="utf-8")
    audit_path.write_text(json.dumps({"reply": candidate["reply"]}), encoding="utf-8")
    second = stage2.build_target_resolution_candidates()

    assert first["candidates"] == second["candidates"] == 1
    assert manifest_path.read_bytes() == first_bytes
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600


def test_all_target_resolution_work_dirs_are_ignored(tmp_path):
    work_file = (
        tmp_path
        / "adversarial_stage2_target_resolution_candidates_v2"
        / "manifest.json"
    )
    assert stage2._is_target_resolution_work_file(work_file)
    assert not stage2._is_target_resolution_work_file(tmp_path / "unrelated" / "manifest.json")


def test_target_resolution_candidates_exclude_stage2_events(tmp_path, monkeypatch):
    work_dir = tmp_path / "target_resolution"
    manifest_path = work_dir / "manifest.json"
    stage2_path = tmp_path / "stage2.json"
    candidate = {
        "internal_id": "candidate-1",
        "event_cluster": "shared-event",
        "chat_id": "chat-1",
        "chat_type": "group",
        "complex_large_group": True,
        "context_messages": [
            {"speaker": "participant_1", "content": "第一句"},
            {"speaker": "participant_2", "content": "第二句"},
        ],
        "reply": "另一条真人回复",
    }
    stage2_path.write_text(
        json.dumps({"cases": [{**candidate, "internal_id": "holdout-1", "reply": "终验回复"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(stage2, "TARGET_RESOLUTION_DIR", work_dir)
    monkeypatch.setattr(stage2, "TARGET_RESOLUTION_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(stage2, "MANIFEST_PATH", stage2_path)
    monkeypatch.setattr(stage2, "ROOT", tmp_path)
    monkeypatch.setattr(stage2, "collect_exposed_fingerprints", lambda: (set(), 0))
    monkeypatch.setattr(stage2, "_stage1_replies", lambda: [])
    monkeypatch.setattr(stage2, "_ordinary_candidates", lambda _exposed: [candidate])
    monkeypatch.setattr(
        stage2,
        "_filter_semantic",
        lambda rows, _old, deduplicate_within=False: rows,
    )

    result = stage2.build_target_resolution_candidates()

    assert result["candidates"] == 0
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["cases"] == []


def test_target_resolution_candidates_keep_one_most_informative_case_per_event(
    tmp_path, monkeypatch
):
    manifest_path = tmp_path / "candidates.json"
    stage2_path = tmp_path / "stage2.json"
    stage2_path.write_text(json.dumps({"cases": []}), encoding="utf-8")
    base = {
        "chat_id": "group",
        "chat_type": "group",
        "complex_large_group": True,
        "event_cluster": "same-event",
        "reply": "reply",
    }
    less = {**base, "internal_id": "less", "context_messages": [
        {"speaker": "participant_1", "content": "a"},
        {"speaker": "participant_2", "content": "b"},
    ]}
    more = {**base, "internal_id": "more", "context_messages": [
        {"speaker": "participant_1", "content": "a"},
        {"speaker": "participant_2", "content": "b"},
        {"speaker": "participant_3", "content": "c"},
    ]}
    monkeypatch.setattr(stage2, "MANIFEST_PATH", stage2_path)
    monkeypatch.setattr(stage2, "ROOT", tmp_path)
    monkeypatch.setattr(stage2, "collect_exposed_fingerprints", lambda: (set(), 0))
    monkeypatch.setattr(stage2, "_stage1_replies", lambda: [])
    monkeypatch.setattr(stage2, "_ordinary_candidates", lambda _exposed: [less, more])
    monkeypatch.setattr(
        stage2,
        "_filter_semantic",
        lambda rows, _old, deduplicate_within=False: rows,
    )
    monkeypatch.setattr(stage2, "_near_duplicate", lambda *_args: False)

    stage2.build_target_resolution_candidates(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert [row["internal_id"] for row in payload["cases"]] == ["more"]
    assert payload["selection_policy"]["one_case_per_event"] is True
