import hashlib
import json

from scripts import build_adversarial_stage2_holdout as stage2
from scripts.discover_fixed_judge_candidates import (
    CATEGORY_TARGETS,
    COMPLEX_CHAT_CAP,
    _common_candidates,
    _tail_is_clean,
    filter_unexposed,
    select_fixed_non_youzi,
)


def test_common_candidate_skips_dirty_first_reply_and_uses_later_clean_reply(
    tmp_path, monkeypatch, synthetic_private_case_config
):
    source = tmp_path / "common.json"
    messages = [
        {
            "createTime": 99,
            "localType": 1,
            "content": "前情",
            "isSend": False,
            "senderUsername": "member",
        },
        {
            "createTime": 100,
            "localType": 3,
            "content": "[图片]",
            "isSend": False,
            "senderUsername": "member",
        },
        {
            "createTime": 101,
            "localType": 1,
            "content": "第一条真人回复",
            "isSend": True,
            "senderUsername": synthetic_private_case_config["self_sender"],
        },
        {
            "createTime": 102,
            "localType": 1,
            "content": "甲",
            "isSend": False,
            "senderUsername": "member",
        },
        {
            "createTime": 103,
            "localType": 1,
            "content": "乙",
            "isSend": False,
            "senderUsername": "member",
        },
        {
            "createTime": 104,
            "localType": 1,
            "content": "丙",
            "isSend": False,
            "senderUsername": "member",
        },
        {
            "createTime": 105,
            "localType": 1,
            "content": "第二条真人回复",
            "isSend": True,
            "senderUsername": synthetic_private_case_config["self_sender"],
        },
    ]
    source.write_text(
        json.dumps({"talker": synthetic_private_case_config["common_talker"], "messages": messages}),
        encoding="utf-8",
    )
    monkeypatch.setattr(stage2, "COMMON_SOURCE", source)
    monkeypatch.setattr(stage2, "ROOT", tmp_path)
    monkeypatch.setattr(
        "scripts.discover_fixed_judge_candidates.COMMON_EXTRA_SOURCES", ()
    )

    cases = _common_candidates(set())

    assert len(cases) == 1
    assert cases[0]["reply"] == "第二条真人回复"


def test_extra_common_source_has_separate_ids_and_preserves_original(
    tmp_path, monkeypatch, synthetic_private_case_config,
):
    def write_source(path):
        path.write_text(json.dumps({
            "talker": synthetic_private_case_config["common_talker"],
            "messages": [
                {"createTime": 100, "localType": 1, "content": "甲", "isSend": False,
                 "senderUsername": "member"},
                {"createTime": 101, "localType": 1, "content": "乙", "isSend": False,
                 "senderUsername": "member"},
                {"createTime": 102, "localType": 1, "content": "丙", "isSend": False,
                 "senderUsername": "member"},
                {"createTime": 103, "localType": 1, "content": path.stem, "isSend": True,
                 "senderUsername": synthetic_private_case_config["self_sender"]},
            ],
        }), encoding="utf-8")

    original = tmp_path / "original.json"
    extra = tmp_path / "extra.json"
    write_source(original)
    write_source(extra)
    monkeypatch.setattr(stage2, "COMMON_SOURCE", original)
    monkeypatch.setattr(stage2, "ROOT", tmp_path)
    monkeypatch.setattr(
        "scripts.discover_fixed_judge_candidates.COMMON_EXTRA_SOURCES",
        (("extra_period", extra),),
    )

    cases = _common_candidates(set())

    common_talker = synthetic_private_case_config["common_talker"]
    original_event = hashlib.sha256(f"{common_talker}:0".encode()).hexdigest()[:20]
    assert len(cases) == 2
    assert cases[0]["event_cluster"] == original_event
    assert cases[1]["event_cluster"] != original_event
    assert stage2.COMMON_SOURCE == original


def _case(index: int, category: str, *, chat: str | None = None) -> dict:
    row = {
        "internal_id": f"case-{index}",
        "event_cluster": f"event-{index}",
        "chat_id": chat or f"chat-{index // 3}",
        "chat_type": "group",
        "complex_large_group": False,
        "specified_group": None,
        "reply": chr(0x4E00 + index) * 8,
    }
    if category == "specified_common":
        row["specified_group"] = "correct_common_prosperity_with_luxixi"
        row["chat_id"] = "common"
    elif category == "other_group_complex":
        row["complex_large_group"] = True
    elif category == "private":
        row["chat_type"] = "private"
    return row


def test_tail_rejects_media_among_last_three_event_messages():
    messages = [
        {"localType": 1, "content": "甲"},
        {"localType": 3, "content": "[图片]"},
        {"localType": 1, "content": "乙"},
        {"localType": 1, "content": "真人回复"},
    ]
    assert _tail_is_clean(messages, 3, [0, 0, 0, 0]) is False


def test_filter_excludes_case_event_reply_and_near_duplicate_exposure(monkeypatch):
    cases = [_case(index, "other_group") for index in range(4)]
    monkeypatch.setattr(
        "scripts.discover_fixed_judge_candidates.stage2._near_duplicate",
        lambda left, right: left == cases[3]["reply"] and right == "reference",
    )
    accepted, counts = filter_unexposed(
        cases,
        exposed_case_ids={"case-0"},
        exposed_events={"event-1"},
        exposed_reply_fingerprints={stage2._fingerprint(cases[2]["reply"])},
        comparison_replies=["reference"],
    )
    assert accepted == []
    assert counts == {
        "case_id_exposed": 1,
        "event_exposed": 1,
        "reply_exposed": 1,
        "reply_near_duplicate": 1,
    }


def test_fixed_non_youzi_selection_meets_quotas_and_is_deterministic():
    cases = []
    index = 0
    for category, target in CATEGORY_TARGETS.items():
        for _ in range(target):
            cases.append(_case(index, category))
            index += 1
    first = select_fixed_non_youzi(cases, seed="fixed")
    second = select_fixed_non_youzi(list(reversed(cases)), seed="fixed")
    assert [row["internal_id"] for row in first] == [row["internal_id"] for row in second]
    assert len(first) == sum(CATEGORY_TARGETS.values())
    assert len({row["event_cluster"] for row in first}) == len(first)


def test_complex_selection_uses_separate_six_case_chat_cap():
    cases = [
        _case(index, "other_group_complex", chat=f"complex-{index // COMPLEX_CHAT_CAP}")
        for index in range(CATEGORY_TARGETS["other_group_complex"])
    ]

    selected = select_fixed_non_youzi(cases, seed="fixed")
    complex_rows = [row for row in selected if row["complex_large_group"]]
    chat_counts = {
        chat_id: sum(row["chat_id"] == chat_id for row in complex_rows)
        for chat_id in {row["chat_id"] for row in complex_rows}
    }

    assert len(complex_rows) == CATEGORY_TARGETS["other_group_complex"]
    assert max(chat_counts.values()) <= COMPLEX_CHAT_CAP
