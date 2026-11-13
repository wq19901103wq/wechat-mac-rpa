import json
import sqlite3

from src.benchmarks.reply_system_replay import (
    load_replay_cases,
    private_source_snapshot,
    public_replay_report,
    replay_cases,
)


class FakeGenerator:
    calls = 0

    def generate(self, unreplied, all_messages, **kwargs):
        self.__class__.calls += 1
        assert unreplied[-1].text == "现在说"
        assert all_messages[0].sender == "真人"
        assert kwargs["evaluation_time"].isoformat() == "2026-06-17T07:10:02"
        self.last_generation_failed = False
        self.last_active_llm = "fake"
        self.last_loaded_skills = ["casual_chat"]
        self.last_self_refine_applied = False
        self.last_feedback_decision = ""
        self.last_feedback_issues = []
        self.last_iterate_count = 0
        self.last_llm_calls = [{}]
        self.last_tool_calls = []
        return ["当前回复"]


def _fixtures(tmp_path):
    db = tmp_path / "cases.db"
    connection = sqlite3.connect(db)
    connection.execute(
        """CREATE TABLE tick_log (
        id INTEGER, tick_id INTEGER, created_at TEXT, chat_name TEXT, is_group INTEGER,
        session_input_messages_json TEXT, session_output_unreplied_json TEXT)"""
    )
    previous = {"sender": "真人", "sender_type": "other", "text": "前文"}
    target = {"sender": "真人", "sender_type": "other", "text": "现在说"}
    connection.execute(
        "INSERT INTO tick_log VALUES (?, ?, ?, ?, ?, ?, ?)",
        (7, 9, "2026-06-17 07:10:02", "私聊名", 0,
         json.dumps([previous, target]), json.dumps([target])),
    )
    connection.commit()
    connection.close()
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text(json.dumps({
        "source_fingerprint": "source",
        "cases": [{"case_id": "one", "source_db_id": 7}],
    }))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "source_fingerprint": "source",
        "cases": [{"case_id": "one"}],
    }))
    return db, scenarios, selection


def test_loads_original_tick_and_replays_without_bot(tmp_path):
    db, scenarios, selection = _fixtures(tmp_path)
    fingerprint, cases = load_replay_cases(db, scenarios, selection)
    output = tmp_path / "private.json"
    FakeGenerator.calls = 0

    packet = replay_cases(
        cases, FakeGenerator, output, fingerprint, {"model": "fake"}
    )

    assert FakeGenerator.calls == 1
    assert packet["cases"][0]["replies"] == ["当前回复"]
    assert output.stat().st_mode & 0o777 == 0o600
    assert public_replay_report(packet)["quality_status"] == "ai_provisional_review_pending"
    assert "当前回复" not in json.dumps(public_replay_report(packet), ensure_ascii=False)


def test_private_source_snapshot_keeps_exact_replay_source_out_of_public_report(tmp_path):
    (tmp_path / "src/reply").mkdir(parents=True)
    (tmp_path / "src/reply/generator.py").write_text("exact source", encoding="utf-8")
    snapshot = private_source_snapshot(tmp_path)

    assert snapshot == {"src/reply/generator.py": "exact source"}
    packet = {
        "status": "completed",
        "source_fingerprint": "source",
        "system": {"model": "fake"},
        "source_snapshot": snapshot,
        "cases": [],
    }
    assert "exact source" not in json.dumps(public_replay_report(packet))


def test_resume_does_not_repeat_completed_case(tmp_path):
    db, scenarios, selection = _fixtures(tmp_path)
    fingerprint, cases = load_replay_cases(db, scenarios, selection)
    output = tmp_path / "private.json"
    FakeGenerator.calls = 0
    replay_cases(cases, FakeGenerator, output, fingerprint, {"model": "fake"})
    replay_cases(cases, FakeGenerator, output, fingerprint, {"model": "fake"})
    assert FakeGenerator.calls == 1


def test_parallel_replay_keeps_all_cases(tmp_path):
    db, scenarios, selection = _fixtures(tmp_path)
    scenario_data = json.loads(scenarios.read_text())
    selection_data = json.loads(selection.read_text())
    scenario_data["cases"].append({"case_id": "two", "source_db_id": 7})
    selection_data["cases"].append({"case_id": "two"})
    scenarios.write_text(json.dumps(scenario_data))
    selection.write_text(json.dumps(selection_data))
    fingerprint, cases = load_replay_cases(db, scenarios, selection)

    packet = replay_cases(
        cases,
        FakeGenerator,
        tmp_path / "private.json",
        fingerprint,
        {"model": "fake"},
        workers=2,
    )
    assert {item["case_id"] for item in packet["cases"]} == {"one", "two"}


def test_failure_is_checkpointed_and_counted(tmp_path):
    db, scenarios, selection = _fixtures(tmp_path)
    fingerprint, cases = load_replay_cases(db, scenarios, selection)

    def broken():
        raise RuntimeError("no model")

    packet = replay_cases(
        cases, broken, tmp_path / "private.json", fingerprint, {"model": "fake"}
    )
    report = public_replay_report(packet)
    assert report["failed_cases"] == 1
    assert packet["cases"][0]["error"] == "RuntimeError: no model"


def test_retry_generation_failures_replaces_failed_checkpoint(tmp_path):
    db, scenarios, selection = _fixtures(tmp_path)
    fingerprint, cases = load_replay_cases(db, scenarios, selection)
    output = tmp_path / "private.json"

    def broken():
        raise RuntimeError("no model")

    replay_cases(cases, broken, output, fingerprint, {"model": "fake"})
    FakeGenerator.calls = 0
    packet = replay_cases(
        cases,
        FakeGenerator,
        output,
        fingerprint,
        {"model": "fake"},
        retry_generation_failures=True,
    )

    assert FakeGenerator.calls == 1
    assert len(packet["cases"]) == 1
    assert packet["cases"][0]["status"] == "completed"
