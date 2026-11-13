from scripts.build_persona_few_shots import Candidate
from scripts.preview_full_history_style_retrieval import (
    _bm25_scores,
    _current_response_anchor,
    _response_anchor,
    rank_same_chat,
    shortlist_same_chat,
)


def _candidate(context: str, reply: str) -> Candidate:
    return Candidate(
        relation="group",
        chat_id="chat_same",
        context=[context],
        replies=[reply],
        score=1.0,
        timestamp=1,
        source_provenance="before_automation_cutoff",
    )


def test_same_chat_bm25_rewards_specific_topic_overlap():
    scores = _bm25_scores("杭州地铁怎么坐", ["到杭州了开始坐地铁", "晚上吃什么"])
    assert scores[0] > scores[1]


def test_ranker_keeps_genuine_context_reply_and_provenance():
    row = rank_same_chat("杭州地铁", [_candidate("到杭州了", "开始坐地铁")], dense_scores=None)[0]
    assert row["historical_context"] == ["到杭州了"]
    assert row["human_reply"] == ["开始坐地铁"]
    assert row["source_provenance"] == "before_automation_cutoff"
    assert row["candidate_id"].startswith("human_")


def test_shortlist_allows_zero_when_same_chat_history_has_no_topic_overlap():
    rows = [_candidate("年终奖和股票总包", "拿完期权")]
    assert shortlist_same_chat("绿藻头请大家喝奶茶", rows) == []


def test_response_anchor_uses_last_incoming_message_instead_of_unrelated_earlier_topic():
    row = _candidate("奶茶什么时候兑现", "肖剑这是关出病了？")
    row.context = ["奶茶什么时候兑现", "马上就输了", "你们"]
    assert _response_anchor(row) == "马上就输了\n你们"


def test_response_anchor_keeps_question_before_short_vocative():
    row = _candidate("这周六一起加班啊", "不了")
    row.context = ["这周六一起加班啊", "干哥"]
    assert _response_anchor(row) == "这周六一起加班啊\n干哥"


def test_current_response_anchor_ignores_old_topics_in_long_unreplied_batch():
    messages = ["好羡慕离职的", "耳朵疼", "https://sakuran.org/", "上次买的这个"]
    assert _current_response_anchor(messages) == "https://sakuran.org/\n上次买的这个"


def test_current_response_anchor_keeps_short_group_banter_lead_in():
    messages = ["芊呢", "芊小微出来", "急于撇清的感觉有"]
    assert _current_response_anchor(messages) == "芊小微出来\n急于撇清的感觉有"


def test_shortlist_requires_matching_high_precision_conversation_move():
    rows = [
        _candidate("chatgpt任务没了", "再建一个"),
        _candidate("智能床垫有啥功能", "零重力模式"),
    ]
    assert shortlist_same_chat("为什么chatgpt任务会丢", rows) == [rows[0]]


def test_shortlist_requires_strict_social_request_move_over_generic_word_overlap():
    rows = [
        _candidate("第一批，后面还有", "送到美国"),
        _candidate("奶茶什么时候兑现", "下次请"),
    ]
    assert shortlist_same_chat("请群友第一杯奶茶", rows) == [rows[1]]


def test_shortlist_requires_system_state_not_any_ai_topic():
    rows = [
        _candidate("ai定价很便宜", "定价199"),
        _candidate("ai还开着吗", "没关"),
    ]
    assert shortlist_same_chat("ai没关", rows) == [rows[1]]


def test_shortlist_blocks_generic_answer_for_fact_sensitive_question():
    rows = [
        _candidate("打过去了，收到了吗", "没有"),
        _candidate("你们领证了吗", "这个我真不知道"),
    ]
    assert shortlist_same_chat("王芊领结婚证了吗，请回答领了或者没领", rows) == [rows[1]]


def test_exact_reuse_risk_detects_historical_context_inside_larger_target():
    row = rank_same_chat("来 嗯 猪呢", [_candidate("猪呢", "稍等")], dense_scores=None)[0]
    assert row["exact_target_reuse_risk"] is True
