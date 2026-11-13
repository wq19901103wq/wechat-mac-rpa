from scripts.build_persona_few_shots import Candidate
from scripts.preview_strict_same_chat_style_matching import (
    accept_match,
    candidate_anchor,
    commitment_risk,
    directed_to_other,
    has_topic_evidence,
    question_intent,
    response_anchor,
    response_window,
)


def _candidate(context: list[str]) -> Candidate:
    return Candidate(
        relation="group",
        chat_id="same_chat",
        context=context,
        replies=["真人回复"],
        score=1.0,
        timestamp=1,
        source_provenance="before_automation_cutoff",
    )


def test_last_message_is_reply_target_instead_of_merging_two_questions():
    messages = ["你在摆摊么", "你属什么"]
    assert response_anchor(messages) == "你属什么"
    assert response_window(messages) == "你在摆摊么\n你属什么"


def test_candidate_uses_last_incoming_message_as_reply_target():
    assert candidate_anchor(_candidate(["旧话题", "最后的问题"])) == "最后的问题"


def test_non_substantive_or_url_target_is_not_matched():
    assert response_anchor(["前面的话", "?"]) == ""
    assert response_anchor(["https://example.com/"]) == ""


def test_topic_evidence_can_come_from_lead_in_but_cannot_replace_target_similarity():
    assert has_topic_evidence("到杭州了\n开始坐地铁", "快到了，21:48到杭州")
    accepted, reasons = accept_match(
        anchor_similarity=0.40,
        context_similarity=0.80,
        joint_score=0.66,
        margin=0.10,
        topic_evidence=True,
        response_alignment=0.70,
        question_intent_compatible=True,
        target_directed_to_other=False,
    )
    assert accepted is False
    assert "reply_target_similarity_below_threshold" in reasons


def test_match_requires_both_similarity_checks_margin_and_topic_evidence():
    accepted, reasons = accept_match(
        anchor_similarity=0.70,
        context_similarity=0.68,
        joint_score=0.69,
        margin=0.06,
        topic_evidence=True,
        response_alignment=0.55,
        question_intent_compatible=True,
        target_directed_to_other=False,
    )
    assert accepted is True
    assert reasons == []


def test_realistic_commitment_is_blocked_but_non_commitment_banter_is_not():
    assert commitment_risk(["我给你发红包"])
    assert commitment_risk(["下次请你喝奶茶"])
    assert not commitment_risk(["你先把欠我的两万还了"])


def test_question_intent_and_explicit_other_person_are_separate_boundaries():
    assert question_intent("你属什么")
    assert question_intent("你到家了吗")
    assert not question_intent("ai没关")
    assert directed_to_other("@腐生 注意安全")
    assert not directed_to_other("注意安全")


def test_match_rejects_unaligned_historical_reply_and_intent_mismatch():
    accepted, reasons = accept_match(
        anchor_similarity=0.80,
        context_similarity=0.75,
        joint_score=0.78,
        margin=0.10,
        topic_evidence=True,
        response_alignment=0.35,
        question_intent_compatible=False,
        target_directed_to_other=False,
    )
    assert accepted is False
    assert "historical_reply_not_aligned_with_incoming" in reasons
    assert "question_vs_statement_intent_mismatch" in reasons
