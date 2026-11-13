import numpy as np

from scripts.preview_offline_style_action_index import (
    _commitment_risk,
    _fact_sensitive,
    rank_action_rows,
    select_action,
)


def test_select_action_rejects_low_confidence_and_small_margin():
    actions = {
        "banter": np.array([[1.0, 0.0]], dtype=np.float32),
        "arrival": np.array([[0.99, 0.01]], dtype=np.float32),
    }
    assert select_action(np.array([0.2, 0.2], dtype=np.float32), actions)[0] is None
    assert select_action(np.array([1.0, 0.0], dtype=np.float32), actions)[0] is None


def test_select_action_accepts_clear_match():
    actions = {
        "banter": np.array([[1.0, 0.0]], dtype=np.float32),
        "arrival": np.array([[0.0, 1.0]], dtype=np.float32),
    }
    assert select_action(np.array([1.0, 0.0], dtype=np.float32), actions)[0] == "banter"


def test_ranking_blocks_fact_risk_commitments_and_recent_repetition():
    rows = [
        {"id": "fact", "action": "answer", "reply": ["没有"], "fact_risk": True, "commitment_risk": False},
        {"id": "promise", "action": "answer", "reply": ["我转账给你"], "fact_risk": False, "commitment_risk": True},
        {"id": "repeat", "action": "answer", "reply": ["刚说过"], "fact_risk": False, "commitment_risk": False},
        {"id": "safe", "action": "answer", "reply": ["这个我真不知道"], "fact_risk": False, "commitment_risk": False},
    ]
    for row in rows:
        row.update({"chat_id": "chat", "relationship": "group", "context": ["问题"]})
    ranked = rank_action_rows(
        np.array([1.0, 0.0], dtype=np.float32),
        "answer",
        rows,
        np.array([[1.0, 0.0]] * 4, dtype=np.float32),
        chat_id="chat",
        relationship="group",
        query_text="他们是不是领证了",
        recent_text="刚说过",
        limit=5,
    )
    assert [row["id"] for row in ranked] == ["safe"]


def test_explicit_risk_detectors_keep_jokes_separate_from_real_promises():
    assert _fact_sensitive("请回答领了还是没领") is True
    assert _commitment_risk(["我转账给你"]) is True
    assert _commitment_risk(["你这是要发财了"]) is False
