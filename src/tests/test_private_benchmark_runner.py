import sqlite3

import pytest

from src.benchmarks.private_runner import mine_review_candidates, run_private_benchmarks


def _create_tick_db(path):
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE tick_log (
            id INTEGER PRIMARY KEY,
            tick_id INTEGER,
            created_at TEXT,
            chat_name TEXT,
            judge_score REAL,
            judge_is_badcase INTEGER,
            judge_badcase_type TEXT,
            human_is_badcase INTEGER,
            feedback_decision TEXT,
            self_refine_applied INTEGER
        )
        """
    )
    connection.executemany(
        "INSERT INTO tick_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 101, "2026-07-01", "private-a", 90, 0, "none", None, "pass", 0),
            (2, 102, "2026-07-02", "private-b", 45, 1, "hallucination", None, "", 0),
            (3, 103, "2026-07-03", "private-c", 80, 0, "none", 1, "fail", 1),
        ],
    )
    connection.commit()
    connection.close()


def test_mine_review_candidates_prioritizes_disagreement_without_private_text(tmp_path):
    db_path = tmp_path / "cases.db"
    _create_tick_db(db_path)

    result = mine_review_candidates(db_path, limit=10)

    assert result["candidate_pool"] == 2
    assert result["items"][0]["db_id"] == 3
    assert result["items"][0]["reasons"] == [
        "human_judge_disagreement",
        "self_refine_failed",
    ]
    serialized = str(result)
    assert "private-a" not in serialized
    assert "private-b" not in serialized
    assert "private-c" not in serialized


def test_private_runner_rejects_unknown_refresh_before_running():
    with pytest.raises(ValueError, match="unsupported refresh mode"):
        run_private_benchmarks(refresh="unexpected")
