from scripts.discover_youzi_judge_candidates import discover_candidates
from scripts.seal_youzi_stage2_quota import SELF_SENDER


def _message(
    timestamp: int,
    content: str,
    *,
    sender: str = "other",
    is_send: bool = False,
    local_type: int = 1,
    server_id: str = "",
):
    return {
        "createTime": timestamp,
        "content": content,
        "senderUsername": sender,
        "isSend": is_send,
        "localType": local_type,
        "serverId": server_id or str(timestamp),
    }


def _discover(messages, *, exposed_events=None):
    return discover_candidates(
        messages,
        exposed_case_ids=set(),
        exposed_events=exposed_events or set(),
        exposed_reply_fingerprints=set(),
        comparison_replies=[],
    )


def test_discovers_one_verified_reply_per_clean_event():
    messages = [
        _message(1, "甲"),
        _message(2, "乙"),
        _message(3, "丙"),
        _message(4, "真人回复", sender=SELF_SENDER, is_send=True),
        _message(5, "同事件第二条", sender=SELF_SENDER, is_send=True),
    ]

    cases, _ = _discover(messages)

    assert len(cases) == 1
    assert cases[0]["human_reply"]["content"] == "真人回复"
    assert cases[0]["review"]["tail_three_messages_text_only"] is True


def test_excludes_event_with_media_in_last_three_messages():
    messages = [
        _message(1, "甲"),
        _message(2, "图片", local_type=3),
        _message(3, "丙"),
        _message(4, "真人回复", sender=SELF_SENDER, is_send=True),
    ]

    cases, exclusions = _discover(messages)

    assert cases == []
    assert exclusions["no_clean_text_tail"] == 1


def test_excludes_previously_exposed_event_cluster():
    messages = [
        _message(1, "甲"),
        _message(2, "乙"),
        _message(3, "丙"),
        _message(4, "真人回复", sender=SELF_SENDER, is_send=True),
    ]
    first, _ = _discover(messages)

    cases, exclusions = _discover(
        messages, exposed_events={first[0]["event_cluster"]}
    )

    assert cases == []
    assert exclusions["event_already_exposed"] == 1
