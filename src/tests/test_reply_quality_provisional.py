import pytest

from src.benchmarks.reply_quality_provisional import apply_provisional_annotations


def test_applies_provisional_without_completing_human_review():
    packet = {
        "source_fingerprint": "abc",
        "cases": [{"case_id": "one", "human_review": {"status": "pending"}}],
    }
    annotations = [
        {
            "case_id": "one",
            "recommended_action": "reply",
            "experience_grade": "good",
            "confidence": "medium",
            "reason": "接住上下文并有推进价值",
        }
    ]

    updated, report = apply_provisional_annotations(packet, annotations)

    assert updated["cases"][0]["human_review"]["status"] == "pending"
    assert updated["cases"][0]["ai_provisional"]["experience_grade"] == "good"
    assert report["eligible_to_calibrate_judge"] is False


def test_requires_exact_case_coverage():
    with pytest.raises(ValueError, match="case_id"):
        apply_provisional_annotations({"cases": [{"case_id": "one"}]}, [])
