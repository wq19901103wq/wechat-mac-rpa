import json

import pytest

from scripts.build_group_chat_judge_review import (
    COMMON_CHAT_ID,
    DISCLAIMER,
    GENERATION_DISCLAIMER,
    YUZU_CHAT_ID,
    _ScaledTimeoutClient,
    _balanced_group_comparison,
    _boundary_tradeoffs,
    _context_view,
    _record,
    _reply_view,
    _target_group_candidates,
    _target_group_export_payload,
    freeze_holdout_selection,
    load_target_group_inventory,
    load_target_group_identity_audit,
    render_html,
    _timestamp_seconds,
)
from scripts.build_persona_few_shots import Candidate
from scripts.evaluate_human_ai_judge import rebuild_generation_inputs


def _candidate(chat_id: str, replies=None, timestamp=1_700_000_000):
    return Candidate(
        relation="group",
        chat_id=chat_id,
        context=["群友：前一句", "群友：问题"],
        replies=replies or ["真人回复"],
        score=1,
        timestamp=timestamp,
    )


def test_historical_approximation_label_is_explicit_and_yuzu_has_no_answer():
    common = _record(_candidate(COMMON_CHAT_ID), "共同富裕群", True)
    yuzu = _record(_candidate(YUZU_CHAT_ID, ["不得进入答案"]), "柚子群2", False)

    assert common["disclaimer"] == DISCLAIMER
    assert common["production_reproduction"] is False
    assert common["replies"] == ["真人回复"]
    assert yuzu["disclaimer"] == DISCLAIMER
    assert yuzu["scored"] is False
    assert yuzu["replies"] == []
    assert "不得进入答案" not in json.dumps(yuzu, ensure_ascii=False)
    assert GENERATION_DISCLAIMER == "C0 逻辑近似，非生产 C0"


def test_offline_timeout_wrapper_scales_only_supplied_timeout():
    class Client:
        def __init__(self):
            self.kwargs = None

        def chat(self, *args, **kwargs):
            self.kwargs = kwargs
            return "ok"

    client = Client()
    wrapped = _ScaledTimeoutClient(client, 3)

    assert wrapped.chat(messages=[], timeout=12) == "ok"
    assert client.kwargs["timeout"] == 36


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [(1_700_000_000, 1_700_000_000), (1_700_000_000_000, 1_700_000_000)],
)
def test_timestamp_seconds_handles_seconds_and_milliseconds(timestamp, expected):
    assert _timestamp_seconds(timestamp) == expected


def test_balanced_group_comparison_shows_original_context_and_all_judges():
    summary, cards = _balanced_group_comparison()

    assert "13 题" in summary
    assert "真人位置 A=6、B=7" in summary
    assert "V2 12/13" in summary
    assert "V3 12/13" in summary
    assert "边界候选 13/13" in summary
    assert "完整聊天上下文" in cards
    assert "真人原回复" in cards
    assert "C0 原回复" in cards
    assert "查看原始提示词、上下文与调用记录" in cards
    assert "class='chat-view'" in cards
    assert "&lt;conversation_history&gt;" in cards


def test_boundary_tradeoffs_show_fixes_regressions_and_raw_evidence():
    summary, cards = _boundary_tradeoffs()

    assert "V3 89/100" in summary
    assert "边界候选 88/100" in summary
    assert "修正 4 题、新增误判 5 题" in summary
    assert cards.count("旧题诊断，非隐藏终测") == 9
    assert "完整聊天上下文" in cards
    assert "真人原回复" in cards
    assert "C0 原回复" in cards
    assert "查看原始提示词、上下文与调用记录" in cards
    assert "class='chat-view'" in cards


def test_mobile_reading_views_hide_machine_wrappers_but_keep_message_content():
    context = [
        '<conversation_history><message role="other" type="text">小王（2024-01-01 10:00）：在吗</message>'
        '<message role="self" type="text">我（2024-01-01 10:01）：在</message></conversation_history>'
    ]

    rendered = _context_view(context)

    assert "<conversation_history>" not in rendered
    assert "class='chat-row other'" in rendered
    assert "class='chat-row self'" in rendered
    assert "小王（2024-01-01 10:00）" in rendered
    assert "在吗" in rendered
    assert "我（2024-01-01 10:01）" in rendered
    assert _reply_view(["第一句", "第二句"]).count("class='reply-line'") == 2


def test_target_group_has_no_unattempted_holdout_cases_left():
    with pytest.raises(RuntimeError, match="未尝试的共同富裕群候选不足：0/12"):
        freeze_holdout_selection()


def test_target_group_inventory_is_the_group_containing_xixi():
    inventory = load_target_group_inventory()

    assert inventory["chat_id"] == "chat_26"
    assert inventory["chat_name"] == "共同富裕群"
    assert "招财鲁" in inventory["members"]
    assert len(inventory["messages"]) == 200
    assert len(inventory["self_indexes"]) == 12
    assert len(inventory["verified_evidence"]) == 12
    assert all(row["same_text_decision_count"] == 0 for row in inventory["verified_evidence"])
    assert all(row["same_text_send_count"] == 0 for row in inventory["verified_evidence"])
    assert all("生产日志已排除当前 Bot" in row["verdict"] for row in inventory["verified_evidence"])


def test_target_group_identity_audit_separates_bot_human_and_unknown():
    audit = load_target_group_identity_audit()

    assert len(audit["confirmed_bot"]) >= 100
    assert audit["representative_bot"]
    replies = [reply for row in audit["representative_bot"] for reply in row["replies"]]
    assert any("人均36带饮料" in reply for reply in replies)
    assert any("复读机三号实锤" in reply for reply in replies)
    assert all(row["send_success"] == 1 for row in audit["confirmed_bot"])
    assert audit["unresolved_self"]


def test_target_group_candidates_merge_consecutive_self_messages_once():
    candidates = _target_group_candidates()

    assert len(candidates) == 10
    replies = [reply for candidate in candidates for reply in candidate.replies]
    assert len(replies) == 12
    assert len(set(replies)) == 12
    assert ["你现在已经和科长混这么熟了", "你不是刚入职吗"] in [row.replies for row in candidates]
    assert ["买入有巨额现金流的公司", "pdd"] in [row.replies for row in candidates]
    assert all(row.chat_id == "chat_26" for row in candidates)


def test_target_group_payload_rebuild_excludes_target_human_reply():
    payload = _target_group_export_payload()
    candidates = _target_group_candidates()

    for candidate in candidates:
        _to_reply, all_messages, metadata = rebuild_generation_inputs(candidate, payload)
        target_index = metadata["target_index"]
        assert len(all_messages) == target_index
        assert payload["messages"][target_index]["content"] == candidate.replies[0]
        assert [
            row["content"] for row in payload["messages"][target_index:target_index + len(candidate.replies)]
        ] == candidate.replies


def test_rendered_review_shows_mobile_first_target_group_audit():
    page = render_html().read_text(encoding="utf-8")

    assert "Judge 迭代 · 私聊 80 题 + 群聊 80 题" in page
    assert "不采用候选，保留当前 Judge 开发基线" in page
    assert "149/160 = 93.125%" in page
    assert "143/160 = 89.375%" in page
    assert "77/80 → 74/80" in page
    assert "72/80 → 69/80" in page
    assert "候选修正 1" in page
    assert "候选新增判错 7" in page
    assert "任一版本判错 18" in page
    assert "p=0.0703125" in page
    assert "四项均未通过" in page
    assert "不能把随机变化强行归因于这个功能" in page
    assert page.count("class='card case balanced-case ") == 160
    assert page.count("class='chat-view'") == 160
    assert "完整聊天时间线" in page
    assert "回复 A" in page
    assert "回复 B" in page
    assert "已知真人答案" in page
    assert "真人原回复" in page
    assert "当前基线 Judge" in page
    assert "候选 Judge" in page
    assert "当前基线 Judge · 完整原始调用记录" in page
    assert "候选 Judge · 完整原始调用记录" in page
    assert "case_3fc8d1c443c978399fac8c29" in page
    assert "Judge v5 已升级" not in page
    assert "固定验证集 · 30 题" not in page
    assert "viewport-fit=cover" in page
    assert "class='chat-view'" in page
    assert "@media(max-width:700px)" in page
    assert "row.getAttribute('data-'+kind)" in page
