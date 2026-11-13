import json

from scripts import build_youzi_identity_review as review


def _message(server_id, timestamp, sender, content, *, is_send=0):
    return {
        "serverId": server_id,
        "createTime": timestamp,
        "senderUsername": sender,
        "content": content,
        "isSend": is_send,
        "localType": 1,
    }


def test_target_self_bubble_is_confirmed_by_user_identity_rule_and_keeps_context():
    messages = [
        _message("before", 100, "wxid_friend", "前一句"),
        _message("self", 105, review.SELF_SENDER, "本人账号气泡", is_send=1),
        _message("after", 110, "wxid_friend", "后一句"),
    ]
    sends = [{
        "tick_id": "tick-1",
        "timestamp": 107.0,
        "observed_at": "2026-01-01T00:00:07+08:00",
        "reply_text": "Bot 生成文本",
        "send_success": True,
        "send_preview": "Bot 生成文本",
        "decision_reason": "测试理由",
        "production_context": [{"sender": "鲁西西", "text": "生产上下文"}],
    }]

    candidates = review.build_candidates(
        messages,
        sends,
        {review.SELF_SENDER: "我", "wxid_friend": "鲁西西"},
    )

    assert len(candidates) == 1
    case = candidates[0]
    assert case["identity_status"] == "用户已确认真人"
    assert [item["sender"] for item in case["event_context"]] == ["鲁西西", "我", "鲁西西"]
    assert [item["content"] for item in case["event_context"]] == ["前一句", "本人账号气泡", "后一句"]
    assert case["wechat_message"]["content"] == "本人账号气泡"
    assert case["bot_send"]["reply_text"] == "Bot 生成文本"
    assert case["bot_send"]["delta_seconds"] == 2.0


def test_explicit_user_confirmation_is_traceable_and_does_not_claim_bot_text():
    server_id = next(iter(review.USER_CONFIRMED_HUMAN_SERVER_IDS))
    candidates = review.build_candidates(
        [_message(server_id, 105, review.SELF_SENDER, "真人气泡", is_send=1)],
        [{
            "tick_id": "tick-1",
            "timestamp": 105.0,
            "observed_at": "2026-01-01T00:00:05+08:00",
            "reply_text": "不同的 Bot 日志文本",
            "send_success": True,
            "send_preview": "不同的 Bot 日志文本",
            "decision_reason": "理由",
            "production_context": [],
        }],
        {review.SELF_SENDER: "我"},
    )

    case = candidates[0]
    assert case["identity_status"] == "用户已确认真人"
    assert case["identity_confirmation"]["confirmed_on"] == "2026-08-19"
    assert case["wechat_message"]["content"] == "真人气泡"
    assert case["bot_send"]["reply_text"] == "不同的 Bot 日志文本"
    assert "不确认邻近 Bot 日志文本" in case["identity_confirmation"]["scope"]


def test_report_does_not_publish_new_hidden_human_content():
    candidates = review.build_candidates(
        [_message("self", 105, review.SELF_SENDER, "本人账号气泡", is_send=1)],
        [{
            "tick_id": "tick-1",
            "timestamp": 105.0,
            "observed_at": "2026-01-01T00:00:05+08:00",
            "reply_text": "Bot 生成文本",
            "send_success": True,
            "send_preview": "预览",
            "decision_reason": "理由",
            "production_context": [],
        }],
        {review.SELF_SENDER: "我"},
    )

    page = review.render_report(candidates)

    assert "本人账号气泡" not in page
    assert "Bot 生成文本" not in page
    assert "其余真人原文不在网页展开" in page


def test_pending_confirmation_candidates_are_not_scored_and_keep_full_context():
    messages = [
        _message("before", 100, "wxid_friend", "前一句"),
        _message("pending", 105, review.SELF_SENDER, "待确认原话", is_send=1),
        _message("after", 110, "wxid_friend", "后一句"),
    ]
    pending = review.build_pending_confirmation_candidates(
        messages,
        {review.SELF_SENDER: "本人昵称", "wxid_friend": "鲁西西"},
        ("pending",),
    )

    assert len(pending) == 1
    assert pending[0]["identity_status"] == "身份尚未确认，不计分"
    assert pending[0]["sender"] == "本人昵称"
    assert [item["sender"] for item in pending[0]["event_context"]] == ["鲁西西", "本人昵称", "鲁西西"]
    assert [item["content"] for item in pending[0]["event_context"]] == ["前一句", "待确认原话", "后一句"]

    page = review.render_report([], pending)
    for expected in ("待你确认的全新事件", "待确认原话", "serverId pending", "身份尚未确认，不计分"):
        assert expected in page


def test_write_outputs_does_not_touch_existing_stage2_manifest(tmp_path):
    old_manifest = tmp_path / "existing-stage2.json"
    old_manifest.write_text("do-not-change", encoding="utf-8")
    output_dir = tmp_path / "identity-review"

    review.write_outputs([], output_dir)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidate_count"] == 0
    assert "Wang Qian account bubbles" in manifest["identity_policy"]
    assert manifest["stage2_manifest_updated"] is False
    assert manifest["judge_run"] is False
    assert manifest["pending_confirmation_count"] == 0
    assert old_manifest.read_text(encoding="utf-8") == "do-not-change"


def test_real_confirmation_rule_has_hidden_independent_events():
    candidates = review.build_candidates(
        review.load_messages(), review.load_successful_sends(), review.load_members()
    )
    qualification = review.assess_qualification(candidates)

    assert qualification["confirmed_bubbles"] == 471
    assert qualification["distinct_confirmed_events"] == 38
    assert qualification["known_exposed_bubbles"] == 14
    assert qualification["qualified_independent_event_representatives"] == 38
    assert qualification["gap"] == 0
    assert qualification["ready_to_seal"] is True
    assert set(review.USER_CONFIRMED_HUMAN_SERVER_IDS) == set(
        qualification["exposed_server_ids"]
    )
    assert len(qualification["selected_server_ids"]) == len(
        set(qualification["selected_server_ids"])
    )
