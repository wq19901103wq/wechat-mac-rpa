import json

import pytest
from fastapi.testclient import TestClient

from src.benchmarks.reply_quality_review_app import (
    create_app,
    render_review_page,
    save_human_review,
)


def _packet():
    return {
        "cases": [
            {
                "case_id": "case-one",
                "pre_context": [{"role": "human", "text": "<script>private</script>"}],
                "target_messages": [{"role": "human", "text": "question"}],
                "candidate": {"action": "reply", "replies": ["candidate"]},
                "human_review": {"status": "pending"},
            }
        ]
    }


def test_review_page_escapes_private_chat_html():
    page = render_review_page(_packet(), 0)
    assert "&lt;script&gt;private&lt;/script&gt;" in page
    assert "<script>private</script>" not in page
    assert "已完成 0/1" in page


def test_save_requires_complete_human_judgment(tmp_path):
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(_packet()), encoding="utf-8")
    with pytest.raises(ValueError, match="理由"):
        save_human_review(
            path,
            {
                "case_id": "case-one",
                "action": "reply",
                "experience_grade": "good",
                "reason": "",
            },
        )


def test_save_completes_review_and_preserves_private_permissions(tmp_path):
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(_packet()), encoding="utf-8")
    result = save_human_review(
        path,
        {
            "case_id": "case-one",
            "action": "reply",
            "experience_grade": "good",
            "reason": "很像当前关系里的真人，而且推进了话题",
            "relationship_and_context_fit": "fits",
            "information_emotion_or_momentum_value": "clear",
            "stale_joke_or_repetition": "none",
            "fact_identity_or_boundary_risk": "none",
        },
    )
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert result == {"saved": True, "completed": 1, "total": 1}
    assert saved["cases"][0]["human_review"]["status"] == "completed"
    assert path.stat().st_mode & 0o777 == 0o600


def test_review_app_rejects_missing_or_wrong_access_token(tmp_path):
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(_packet()), encoding="utf-8")
    client = TestClient(create_app(path, access_token="temporary-secret"))

    assert client.get("/").status_code == 403
    assert client.get("/?token=wrong").status_code == 403
    assert client.post("/api/review", json={}).status_code == 403


def test_review_app_accepts_token_then_uses_private_cookie(tmp_path):
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(_packet()), encoding="utf-8")
    client = TestClient(create_app(path, access_token="temporary-secret"))

    response = client.get("/?token=temporary-secret")

    assert response.status_code == 200
    assert "回复质量 Human Gold" in response.text
    assert "reply_quality_review_token" in client.cookies
