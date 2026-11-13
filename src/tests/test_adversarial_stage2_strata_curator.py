import fcntl
import json
from pathlib import Path

import pytest

from scripts import curate_adversarial_stage2_strata as curator


def _case():
    return {
        "internal_id": "hidden-case-id",
        "chat_id": "hidden-chat-id",
        "event_cluster": "hidden-event-id",
        "source_path": "hidden/source.json",
        "context_messages": [
            {"speaker": "participant_1", "content": "你到哪了"},
            {"speaker": "participant_2", "content": "我先过去"},
        ],
        "reply": "马上",
    }


def test_curator_request_contains_only_anonymous_single_case_packet():
    case = _case()
    messages = curator.build_messages(case)
    body = {"messages": messages, "temperature": 0}
    curator.validate_request_body(body, case)
    serialized = json.dumps(messages, ensure_ascii=False)
    assert case["internal_id"] not in serialized
    assert case["chat_id"] not in serialized
    assert case["event_cluster"] not in serialized
    assert case["source_path"] not in serialized

    system, anthropic_messages = curator.VolcengineAnthropicClient._messages(messages)
    anthropic_body = {"system": system, "messages": anthropic_messages, "temperature": 0}
    curator.validate_request_body(anthropic_body, case)


def test_curator_request_rejects_tools_and_identity_metadata():
    case = _case()
    body = {"messages": curator.build_messages(case), "temperature": 0, "tools": [{"name": "x"}]}
    with pytest.raises(ValueError, match="tools"):
        curator.validate_request_body(body, case)

    body = {"messages": curator.build_messages(case), "temperature": 0, "leak": case["chat_id"]}
    with pytest.raises(ValueError, match="identity"):
        curator.validate_request_body(body, case)


def test_curator_response_requires_all_six_boolean_labels_and_evidence():
    payload = {
        "labels": {name: False for name in curator.STRATA},
        "evidence": {name: "不满足必要条件" for name in curator.STRATA},
    }
    assert curator.parse_response(json.dumps(payload, ensure_ascii=False)) == payload

    payload["labels"][curator.STRATA[0]] = "false"
    with pytest.raises(ValueError, match="booleans"):
        curator.parse_response(json.dumps(payload, ensure_ascii=False))


def test_aggregate_allows_overlapping_strata():
    rows = [
        {"labels": {name: name in curator.STRATA[:2] for name in curator.STRATA}},
        {"labels": {name: name == curator.STRATA[0] for name in curator.STRATA}},
    ]
    result = curator.aggregate(rows)
    assert result[curator.STRATA[0]] == 2
    assert result[curator.STRATA[1]] == 1


def test_curate_case_uses_a_fresh_client_without_tools(monkeypatch):
    clients = []

    class FakeClient:
        def __init__(self, *_args):
            clients.append(self)
            self.last_request_body = {}
            self.last_response_payload = {"model": curator.VOLCENGINE_DSV4F_MODEL}
            self.last_status_code = 200
            self.resolved_model = curator.VOLCENGINE_DSV4F_MODEL

        def chat(self, *, messages, temperature, **_kwargs):
            system = next(row["content"] for row in messages if row["role"] == "system")
            user = next(row["content"] for row in messages if row["role"] == "user")
            self.last_request_body = {
                "model": curator.VOLCENGINE_DSV4F_MODEL,
                "system": system,
                "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
                "temperature": temperature,
            }
            return json.dumps({
                "labels": {name: False for name in curator.STRATA},
                "evidence": {name: "不满足必要条件" for name in curator.STRATA},
            }, ensure_ascii=False)

    monkeypatch.setattr(curator, "VolcengineAnthropicClient", FakeClient)
    call_1, parsed_1 = curator._curate_case("key", "url", _case())
    call_2, parsed_2 = curator._curate_case("key", "url", _case())
    assert len(clients) == 2 and clients[0] is not clients[1]
    assert not call_1["tools_sent"] and not call_2["tools_sent"]
    assert parsed_1 is not None and parsed_2 is not None


def test_run_reports_every_attempted_case_with_workers(monkeypatch, tmp_path: Path):
    cases = [{**_case(), "internal_id": f"case-{index}"} for index in range(240)]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    monkeypatch.setattr(curator, "load_volcengine_credentials", lambda _path: ("key", "url"))
    parsed = {
        "labels": {name: False for name in curator.STRATA},
        "evidence": {name: "不满足必要条件" for name in curator.STRATA},
    }
    monkeypatch.setattr(
        curator,
        "_curate_case",
        lambda _key, _url, case: ({"internal_id": case["internal_id"], "error": None}, parsed),
    )
    checkpoints = []
    monkeypatch.setattr(
        curator,
        "_write_private",
        lambda path, payload: checkpoints.append((path.name, len(payload.get("cases", [])))),
    )

    result = curator.run(
        tmp_path / "config.json",
        max_cases=2,
        workers=2,
        manifest_path=manifest_path,
        output_dir=tmp_path,
    )

    assert result["completed"] == 2
    assert result["attempted_this_run"] == 2
    assert checkpoints == [
        ("strata_labels.json", 1),
        ("strata_curation_audit.json", 0),
        ("strata_labels.json", 2),
        ("strata_curation_audit.json", 0),
    ]


def test_run_rejects_a_second_writer(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"cases": [_case()]}), encoding="utf-8")
    lock_handle = (tmp_path / ".curation.lock").open("a+", encoding="utf-8")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(RuntimeError, match="another curation process"):
            curator.run(
                tmp_path / "config.json",
                manifest_path=manifest_path,
                output_dir=tmp_path,
            )
    finally:
        lock_handle.close()


def test_run_refuses_to_resume_when_source_manifest_changed(tmp_path: Path):
    manifest_path = tmp_path / "source" / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps({"cases": [_case()]}), encoding="utf-8")
    output_dir = tmp_path / "run"
    curator._load_input_snapshot(manifest_path, output_dir)
    manifest_path.write_text(json.dumps({"cases": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="changed since"):
        curator.run(
            tmp_path / "config.json",
            manifest_path=manifest_path,
            output_dir=output_dir,
        )


def test_verify_input_unchanged_detects_mid_run_overwrite(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("before", encoding="utf-8")
    expected = curator._file_sha256(manifest_path)
    manifest_path.write_text("after", encoding="utf-8")

    with pytest.raises(ValueError, match="changed during"):
        curator._verify_input_unchanged(manifest_path, expected)
