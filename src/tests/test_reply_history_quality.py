import json
import sqlite3
from datetime import datetime

from src.benchmarks.reply_history_quality import generate_evaluation


def _ts(value):
    return datetime.fromisoformat(value).timestamp()


def _msg(role, text, timestamp):
    return {
        "sender": "真实姓名" if role == "other" else "自己",
        "sender_type": role,
        "text": text,
        "create_time": timestamp,
        "reply_time": timestamp if role == "self" else None,
    }


def _create_db(path):
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE tick_log (
            id INTEGER PRIMARY KEY, tick_id INTEGER, created_at TEXT,
            chat_name TEXT, is_group INTEGER, replies_sent_json TEXT,
            session_input_messages_json TEXT, session_output_unreplied_json TEXT,
            human_is_badcase INTEGER, human_badcase_type TEXT, human_notes TEXT,
            judge_score REAL, judge_is_badcase INTEGER, judge_reason TEXT,
            send_success INTEGER
        )
        """
    )
    t0 = _ts("2026-08-01 10:00:00")
    first_input = [
        _msg("other", "妈的这也太离谱了", t0 - 20),
        _msg("other", "你怎么看？", t0 - 2),
    ]
    second_input = first_input + [
        _msg("self", "妈的，这件事情确实离谱得很", t0 + 5),
        _msg("other", "哈哈确实", t0 + 20),
        _msg("other", "下一个问题", t0 + 115),
    ]
    connection.executemany(
        "INSERT INTO tick_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                1, 10, "2026-08-01 10:00:00", "私密群", 1,
                json.dumps(["妈的，这件事情确实离谱得很"]), json.dumps(first_input),
                json.dumps([first_input[-1]]), None, None, None, None, None, None, 1,
            ),
            (
                2, 11, "2026-08-01 10:02:00", "私密群", 1,
                json.dumps(["普通回答"]), json.dumps(second_input),
                json.dumps([second_input[-1]]), None, None, None, 92, 0, "仅供事实检查", 1,
            ),
            (
                3, 12, "2026-08-02 10:00:00", "私聊对象", 0,
                json.dumps(["未发送"]), json.dumps([]), json.dumps([]),
                None, None, None, None, None, None, 0,
            ),
        ],
    )
    connection.commit()
    connection.close()


def test_evaluation_covers_sends_and_reconstructs_reaction(tmp_path):
    db_path = tmp_path / "cases.db"
    _create_db(db_path)

    report, packet = generate_evaluation(db_path)

    assert report["scope"]["database_ticks"] == 3
    assert report["scope"]["sent_ticks"] == 2
    assert report["scope"]["sent_messages"] == 2
    assert report["scope"]["archived_review_cases"] == 2
    assert len(packet["cases"]) == 2
    first = packet["cases"][0]
    assert first["historical_reaction_evidence"]["type"] == "positive"
    assert first["post_reaction_window"][0]["text"] == "哈哈确实"
    assert first["human_ground_truth"] is None


def test_public_report_has_no_identity_text_prompt_or_notes(tmp_path):
    db_path = tmp_path / "cases.db"
    _create_db(db_path)

    report, packet = generate_evaluation(db_path)
    serialized = json.dumps(report, ensure_ascii=False)

    for private_value in ["私密群", "私聊对象", "真实姓名", "你怎么看", "普通回答", "仅供事实检查"]:
        assert private_value not in serialized
    assert "你怎么看" in json.dumps(packet, ensure_ascii=False)


def test_judge_is_diagnostic_only_and_human_labels_are_pending(tmp_path):
    db_path = tmp_path / "cases.db"
    _create_db(db_path)

    report, packet = generate_evaluation(db_path)

    assert report["judge_role"]["sent_cases_with_judge_hint"] == 1
    assert report["judge_role"]["excluded_from_experience_grade"] is True
    assert report["human_ground_truth"] == {
        "labeled_cases": 0,
        "total_cases": 2,
        "coverage": 0.0,
        "status": "pending",
    }
    assert report["baseline_status"] == "fully_archived_ai_provisional"
    assert report["release_decision"] == "human_ground_truth_pending"
    assert packet["cases"][1]["judge_fact_safety"]["usage"] == "fact_and_safety_hint_only"


def test_rough_language_is_not_an_automatic_bad_or_critical_label(tmp_path):
    db_path = tmp_path / "cases.db"
    _create_db(db_path)

    _, packet = generate_evaluation(db_path)
    first = packet["cases"][0]

    assert first["ai_provisional"]["grade"] == "good"
    assert "rough_language_context_review" in first["ai_provisional"]["diagnostic_tags"]
    assert first["historical_reaction_evidence"]["type"] == "positive"


def test_cross_turn_content_reuse_is_a_bad_candidate(tmp_path):
    db_path = tmp_path / "cases.db"
    _create_db(db_path)
    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE tick_log SET replies_sent_json = ? WHERE id = 2",
        (json.dumps(["妈的，这件事情确实离谱得很"]),),
    )
    connection.commit()
    connection.close()

    _, packet = generate_evaluation(db_path)
    second = packet["cases"][1]

    assert second["ai_provisional"]["grade"] == "bad"
    assert "cross_turn_repetition_candidate" in second["ai_provisional"]["diagnostic_tags"]
