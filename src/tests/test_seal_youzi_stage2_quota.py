import hashlib
import json
import stat
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts import seal_youzi_stage2_quota as seal


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_sealed_youzi_quota_is_complete_private_and_reproducible():
    manifest = _load(seal.MANIFEST_PATH)
    public = _load(seal.PUBLIC_SUMMARY_PATH)
    cases = manifest["cases"]

    assert len(cases) == 20
    assert len({case["event_hash"] for case in cases}) == 20
    assert {case["anchor_server_id"] for case in cases} == set(seal.SELECTED_ANCHORS)
    assert not set(seal.EXCLUDED_EXPOSED_ANCHORS) & {
        case["anchor_server_id"] for case in cases
    }
    assert all(case["context"] for case in cases)
    assert all(
        case["human_reply"]["is_send"]
        and case["human_reply"]["sender"] == seal.SELF_SENDER
        and case["human_reply"]["local_type"] == 1
        and case["timestamp"] < seal.CUTOFF_EXCLUSIVE
        for case in cases
    )
    assert max(
        datetime.fromtimestamp(case["timestamp"], ZoneInfo("Asia/Shanghai")).date()
        for case in cases
    ).isoformat() == public["coverage"]["latest_selected_date"]

    assert stat.S_IMODE(seal.MANIFEST_PATH.stat().st_mode) == 0o600
    assert stat.S_IMODE(seal.PUBLIC_SUMMARY_PATH.stat().st_mode) == 0o600
    assert hashlib.sha256(seal.MANIFEST_PATH.read_bytes()).hexdigest() == public[
        "manifest_sha256"
    ]


def test_public_youzi_summary_contains_no_hidden_material():
    manifest = _load(seal.MANIFEST_PATH)
    public_text = seal.PUBLIC_SUMMARY_PATH.read_text(encoding="utf-8")
    public = json.loads(public_text)

    assert public["coverage"]["selected_events"] == 20
    assert public["coverage"]["quota_gap"] == 0
    assert public["coverage"]["source_snapshot_latest_date"] == "2026-07-08"
    assert public["stage_2_overall_ready"] is False
    for case in manifest["cases"]:
        assert case["anchor_server_id"] not in public_text
        assert case["event_hash"] not in public_text
        assert case["human_reply"]["content"] not in public_text
    assert seal.TARGET_CHAT not in public_text
