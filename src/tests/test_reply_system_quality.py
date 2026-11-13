import json
import sqlite3

from src.benchmarks.reply_system_quality import generate_fixed_scenarios


def _message(sender_type, text, timestamp):
    return {
        "sender": "private-name",
        "sender_type": sender_type,
        "text": text,
        "create_time": timestamp,
    }


def _create_db(path):
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE tick_log (
            id INTEGER PRIMARY KEY, tick_id INTEGER, created_at TEXT,
            chat_name TEXT, is_group INTEGER, session_input_messages_json TEXT,
            session_output_unreplied_json TEXT, should_reply INTEGER,
            send_success INTEGER, replies_sent_json TEXT,
            human_is_badcase INTEGER, human_badcase_type TEXT, human_notes TEXT
        )
        """
    )
    target = [_message("other", "new question", 100)]
    rows = [
        (1, 10, "2026-08-01 10:00:00", "private-chat", 1, target, target, 1, 1, ["old reply"]),
        # Same opportunity in a later polling tick: must not become a second scenario.
        (2, 11, "2026-08-01 10:00:02", "private-chat", 1, target, target, 1, 0, []),
        (3, 12, "2026-08-01 10:01:00", "private-chat", 1,
         target + [_message("other", "another message", 120)],
         [_message("other", "another message", 120)], 1, 0, []),
    ]
    for row in rows:
        connection.execute(
            "INSERT INTO tick_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)",
            (*row[:5], json.dumps(row[5]), json.dumps(row[6]), *row[7:9], json.dumps(row[9])),
        )
    connection.commit()
    connection.close()


def test_fixed_scenarios_cover_reply_and_silence_opportunities(tmp_path):
    db_path = tmp_path / "cases.db"
    _create_db(db_path)

    manifest, packet = generate_fixed_scenarios(db_path)

    assert manifest["scope"]["database_ticks"] == 3
    assert manifest["scope"]["archived_reply_opportunities"] == 3
    assert manifest["scope"]["deduplicated_fixed_scenarios"] == 2
    assert len(packet["cases"]) == 2
    assert {case["historical_system_evidence"]["action"] for case in packet["cases"]} == {
        "replied",
        "generated_silence_or_send_failed",
    }


def test_manifest_is_not_mislabeled_as_current_system_score(tmp_path):
    db_path = tmp_path / "cases.db"
    _create_db(db_path)

    manifest, packet = generate_fixed_scenarios(db_path)

    assert manifest["current_system_baseline"] == {
        "status": "replay_pending",
        "completed_runs": 0,
        "human_reviewed_scenarios": 0,
        "score": None,
    }
    assert all(case["current_system_runs"] == [] for case in packet["cases"])
    assert all(case["human_review"] is None for case in packet["cases"])


def test_public_manifest_contains_no_chat_text_or_identity(tmp_path):
    db_path = tmp_path / "cases.db"
    _create_db(db_path)

    manifest, packet = generate_fixed_scenarios(db_path)
    public = json.dumps(manifest, ensure_ascii=False)
    private = json.dumps(packet, ensure_ascii=False)

    for value in ["private-chat", "private-name", "new question", "old reply"]:
        assert value not in public
    assert "new question" in private

