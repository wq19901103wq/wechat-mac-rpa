import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import src.reply.few_shot as few_shot_module

from src.models.base import ChatMessage, SenderType
from src.reply.few_shot import PersonaFewShotRetriever, _latest_turn_form
from src.reply.generator import ReplyGenerator


def _write_rows(path, rows):
    for row in rows:
        row.setdefault("source_provenance", "explicit_human_marker")
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("我擦", "brief_exclamation"),
        ("你去还是我去？", "choice_question"),
        ("为什么还没来", "wh_question"),
        ("你到了吗", "yes_no_question"),
        ("这事确定？", "other_question"),
        ("难受", "brief_statement"),
        ("250 offer在路上了", "statement"),
    ],
)
def test_latest_turn_form_has_exact_classes(text, expected):
    assert _latest_turn_form(text) == expected


def test_latest_turn_form_feature_changes_only_candidate_ranking(tmp_path, monkeypatch):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {
            "id": "same_words_statement",
            "relationship": "friend",
            "chat_id": "a",
            "context": ["这事情真的很难受"],
            "reply": ["确实难受"],
        },
        {
            "id": "same_form",
            "relationship": "friend",
            "chat_id": "b",
            "context": ["难顶"],
            "reply": ["难受"],
        },
    ])
    monkeypatch.setattr(few_shot_module, "_situation_tags", lambda *args, **kwargs: {"complaint"})
    monkeypatch.setattr(few_shot_module, "_situation_profile", lambda *args, **kwargs: {})

    default_rows = PersonaFewShotRetriever(path).retrieve(
        "难受", "朋友", is_group=False, limit=2
    )
    candidate_rows = PersonaFewShotRetriever(path, prefer_latest_turn_form=True).retrieve(
        "难受", "朋友", is_group=False, limit=2
    )

    assert [row["id"] for row in default_rows] == ["same_words_statement", "same_form"]
    assert [row["id"] for row in candidate_rows] == ["same_form", "same_words_statement"]


def test_group_participant_structure_beats_word_overlap_with_wrong_reply_target(tmp_path, monkeypatch):
    path = tmp_path / "persona_examples.jsonl"
    rows = [
        {
            "id": "wrong_target",
            "relationship": "group",
            "context": ["老王喝酒吗", "喝"],
            "reply": ["走"],
            "source_provenance": "before_automation_cutoff",
            "context_messages": [
                {"sender": "小李", "text": "老王喝酒吗"},
                {"sender": "老王", "text": "喝"},
            ],
        },
        {
            "id": "right_target",
            "relationship": "group",
            "context": ["我不去了", "你来吗"],
            "reply": ["临时有事"],
            "source_provenance": "before_automation_cutoff",
            "context_messages": [
                {"sender": "我", "text": "我不去了"},
                {"sender": "小李", "text": "你来吗"},
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    monkeypatch.setattr(PersonaFewShotRetriever, "_load_embeddings", lambda self: {})

    baseline = PersonaFewShotRetriever(path).retrieve(
        query="我不去了\n老王喝酒吗",
        chat_name="群",
        is_group=True,
        limit=2,
        current_context_messages=[
            {"sender": "我", "text": "我不去了"},
            {"sender": "小李", "text": "老王喝酒吗"},
        ],
    )
    candidate = PersonaFewShotRetriever(path, prefer_group_participant_structure=True).retrieve(
        query="我不去了\n老王喝酒吗",
        chat_name="群",
        is_group=True,
        limit=2,
        current_context_messages=[
            {"sender": "我", "text": "我不去了"},
            {"sender": "小李", "text": "老王喝酒吗"},
        ],
    )

    assert baseline[0]["id"] == "wrong_target"
    assert candidate[0]["id"] == "right_target"


def test_group_participant_structure_does_not_change_private_ranking(tmp_path, monkeypatch):
    path = tmp_path / "persona_examples.jsonl"
    rows = [
        {
            "id": row_id,
            "relationship": "acquaintance",
            "context": [text],
            "reply": ["好"],
            "source_provenance": "before_automation_cutoff",
            "context_messages": [{"sender": "朋友", "text": text}],
        }
        for row_id, text in (("a", "喝酒吗"), ("b", "吃饭吗"))
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    monkeypatch.setattr(PersonaFewShotRetriever, "_load_embeddings", lambda self: {})
    kwargs = {
        "query": "喝酒吗",
        "chat_name": "朋友",
        "is_group": False,
        "limit": 2,
        "current_context_messages": [{"sender": "朋友", "text": "喝酒吗"}],
    }

    baseline = PersonaFewShotRetriever(path).retrieve(**kwargs)
    candidate = PersonaFewShotRetriever(path, prefer_group_participant_structure=True).retrieve(**kwargs)

    assert [row["id"] for row in candidate] == [row["id"] for row in baseline]


def test_group_retrieval_uses_same_chat_as_soft_boost(tmp_path):
    path = tmp_path / "examples.jsonl"
    chat_name = "柚子群2"
    same_chat_id = "chat_" + hashlib.sha256(chat_name.encode()).hexdigest()[:10]
    _write_rows(path, [
        {"id": "other", "relationship": "group", "chat_id": "chat_other", "context": ["股票跌了"], "reply": ["躺平"]},
        {"id": "same", "relationship": "group", "chat_id": same_chat_id, "context": ["股票跌了"], "reply": ["喝西北风"]},
        {"id": "private", "relationship": "friend", "chat_id": "chat_private", "context": ["股票跌了"], "reply": ["知道了"]},
    ])

    rows = PersonaFewShotRetriever(path).retrieve("股票跌了", chat_name, is_group=True, limit=2)

    assert [row["id"] for row in rows] == ["same", "other"]


def test_retrieval_prefers_relevance_over_unrelated_same_chat(tmp_path):
    path = tmp_path / "examples.jsonl"
    chat_name = "柚子群2"
    same_chat_id = "chat_" + hashlib.sha256(chat_name.encode()).hexdigest()[:10]
    _write_rows(path, [
        {"id": "same", "relationship": "group", "chat_id": same_chat_id, "context": ["晚上吃啥"], "reply": ["火锅"]},
        {"id": "relevant", "relationship": "group", "chat_id": "other", "context": ["股票跌惨了"], "reply": ["又当韭菜了"]},
    ])

    rows = PersonaFewShotRetriever(path).retrieve("股票跌惨了", chat_name, is_group=True, limit=2)

    assert [row["id"] for row in rows] == ["relevant"]


def test_retrieval_limits_repetitive_laugh_examples(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": f"laugh_{i}", "relationship": "friend", "chat_id": f"c{i}", "context": ["今天真离谱"], "reply": [f"哈哈哈{i}"], "reply_shape": "laugh"}
        for i in range(5)
    ] + [
        {"id": "plain", "relationship": "friend", "chat_id": "plain", "context": ["今天真离谱"], "reply": ["你可真行"]},
    ])

    rows = PersonaFewShotRetriever(path).retrieve("今天真离谱", "朋友", is_group=False, limit=4)

    assert "plain" in [row["id"] for row in rows]
    assert sum(row["id"].startswith("laugh_") for row in rows) <= 2


def test_retrieval_does_not_collapse_legacy_default_metadata(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {
            "id": f"row_{index}",
            "relationship": "friend",
            "chat_id": f"chat_{index}",
            "context": ["同一类输入"],
            "reply": [f"不同回复{index}"],
            "intent": "comment",
            "topic": "daily_chat",
        }
        for index in range(6)
    ])

    rows = PersonaFewShotRetriever(path).retrieve("同一类输入", "朋友", is_group=False, limit=6)

    assert len(rows) == 6


def test_retrieval_softly_diversifies_verified_semantic_moves(tmp_path):
    path = tmp_path / "examples.jsonl"
    repeated_profile = {"incoming_act": "hostile_teasing", "response_move": "wording_reversal"}
    _write_rows(path, [
        {
            "id": f"reversal_{index}",
            "relationship": "group",
            "chat_id": f"chat_{index}",
            "context": ["你可真会说"],
            "reply": [f"反转{index}"],
            "semantic_profile": repeated_profile,
        }
        for index in range(5)
    ] + [{
        "id": "contrast",
        "relationship": "group",
        "chat_id": "contrast",
        "context": ["你可真会说"],
        "reply": ["反差"],
        "semantic_profile": {"incoming_act": "hostile_teasing", "response_move": "comic_contrast"},
    }])

    rows = PersonaFewShotRetriever(path).retrieve("你可真会说", "群", is_group=True, limit=6)

    assert len(rows) == 4
    assert sum(row["id"].startswith("reversal_") for row in rows) == 3
    assert any(row["id"] == "contrast" for row in rows)


def test_retrieval_can_expand_after_preserving_diverse_prefix(tmp_path):
    path = tmp_path / "examples.jsonl"
    repeated_profile = {"incoming_act": "hostile_teasing", "response_move": "wording_reversal"}
    _write_rows(path, [
        {
            "id": f"reversal_{index}",
            "relationship": "group",
            "chat_id": f"chat_{index}",
            "context": ["你可真会说"],
            "reply": [f"反转{index}"],
            "semantic_profile": repeated_profile,
        }
        for index in range(10)
    ] + [{
        "id": "contrast",
        "relationship": "group",
        "chat_id": "contrast",
        "context": ["你可真会说"],
        "reply": ["反差"],
        "semantic_profile": {"incoming_act": "hostile_teasing", "response_move": "comic_contrast"},
    }])
    retriever = PersonaFewShotRetriever(path)

    default_rows = retriever.retrieve("你可真会说", "群", is_group=True, limit=10)
    expanded_rows = retriever.retrieve(
        "你可真会说",
        "群",
        is_group=True,
        limit=10,
        diversity_prefix_size=3,
    )

    assert len(default_rows) == 4
    assert len(expanded_rows) == 10
    assert expanded_rows[:3] == default_rows[:3]


def test_retrieval_does_not_infer_topic_from_keyword_lists(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "z_finance", "relationship": "friend", "chat_id": "f1", "context": ["今天怎么了"], "reply": ["又跌了"], "topic": "finance", "intent": "answer"},
        {"id": "a_food", "relationship": "friend", "chat_id": "f2", "context": ["今天怎么了"], "reply": ["去吃饭"], "topic": "food_travel", "intent": "answer"},
    ])

    rows = PersonaFewShotRetriever(path).retrieve("今天股票怎么了", "朋友", is_group=False, limit=2, relationship="friend")

    assert rows[0]["id"] == "a_food"


def test_retrieval_can_use_explicit_hashed_chat_id(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "same", "relationship": "friend", "chat_id": "chat_exact", "context": ["普通话题"], "reply": ["同对象"]},
        {"id": "other", "relationship": "friend", "chat_id": "other", "context": ["普通话题"], "reply": ["其他对象"]},
    ])

    rows = PersonaFewShotRetriever(path).retrieve("普通话题", "", is_group=False, limit=2, chat_id="chat_exact")

    assert rows[0]["id"] == "same"


def test_multi_message_coverage_boost_only_applies_to_multi_message_query(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {
            "id": "a_single_reply",
            "relationship": "friend",
            "chat_id": "a",
            "context": ["前一句", "现在呢"],
            "reply": ["一句回复"],
        },
        {
            "id": "z_multi_reply",
            "relationship": "friend",
            "chat_id": "z",
            "context": ["前一句", "现在呢"],
            "reply": ["先接前一句", "再回现在"],
        },
    ])

    baseline = PersonaFewShotRetriever(path).retrieve(
        "前一句\n现在呢", "朋友", is_group=False, limit=2
    )
    candidate = PersonaFewShotRetriever(
        path, prefer_multi_message_reply_coverage=True
    ).retrieve("前一句\n现在呢", "朋友", is_group=False, limit=2)
    single_turn = PersonaFewShotRetriever(
        path, prefer_multi_message_reply_coverage=True
    ).retrieve("现在呢", "朋友", is_group=False, limit=2)

    assert [row["id"] for row in baseline] == ["a_single_reply", "z_multi_reply"]
    assert [row["id"] for row in candidate] == ["z_multi_reply", "a_single_reply"]
    assert [row["id"] for row in single_turn] == ["a_single_reply", "z_multi_reply"]


def test_retrieval_prefers_same_situation_over_same_words(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "same_words_wrong_scene", "relationship": "friend", "chat_id": "a", "context": ["这个项目真厉害吗"], "reply": ["还行"]},
        {"id": "same_scene", "relationship": "friend", "chat_id": "b", "context": ["你也太牛逼了"], "reply": ["那必须的"]},
    ])

    rows = PersonaFewShotRetriever(path).retrieve(
        "羡慕你们这些大佬",
        "朋友",
        is_group=False,
        limit=2,
        interaction_context="handling_praise",
    )

    assert rows[0]["id"] == "same_scene"


def test_private_hybrid_route_keeps_legacy_word_retrieval(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "same_words", "relationship": "friend", "chat_id": "a", "context": ["这个项目真厉害吗"], "reply": ["还行"]},
        {"id": "same_scene", "relationship": "friend", "chat_id": "b", "context": ["你也太牛逼了"], "reply": ["那必须的"]},
    ])

    rows = PersonaFewShotRetriever(path).retrieve(
        "这个项目真厉害吗",
        "朋友",
        is_group=False,
        limit=2,
        interaction_context="handling_praise",
        use_situation=False,
    )

    assert rows[0]["id"] == "same_words"


def test_retrieval_prefers_exact_chat_only_after_situation_matches(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "same_chat_wrong_scene", "relationship": "friend", "chat_id": "chat_exact", "context": ["晚上吃什么"], "reply": ["火锅"]},
        {"id": "other_chat_same_scene", "relationship": "friend", "chat_id": "other", "context": ["你可太强了"], "reply": ["低调"]},
        {"id": "same_chat_same_scene", "relationship": "friend", "chat_id": "chat_exact", "context": ["真羡慕你"], "reply": ["别闹"]},
    ])

    rows = PersonaFewShotRetriever(path).retrieve(
        "大佬真牛逼",
        "",
        is_group=False,
        limit=3,
        chat_id="chat_exact",
        interaction_context="handling_praise",
    )

    assert [row["id"] for row in rows[:2]] == ["same_chat_same_scene", "other_chat_same_scene"]


def test_candidate_strengthens_same_chat_only_for_a_compatible_situation(tmp_path, monkeypatch):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {
            "id": "same_chat",
            "relationship": "friend",
            "chat_id": "chat_exact",
            "context": ["你真厉害"],
            "reply": ["少来"],
        },
        {
            "id": "other_chat_more_words",
            "relationship": "friend",
            "chat_id": "other",
            "context": ["你真厉害，太强了，大佬牛逼"],
            "reply": ["运气好"],
        },
    ])
    np.savez(
        tmp_path / "persona_embeddings.npz",
        ids=np.array(["same_chat", "other_chat_more_words"]),
        embeddings=np.array([[0.5, 0.866], [1.0, 0.0]], dtype=np.float32),
        examples_sha256=np.array(hashlib.sha256(path.read_bytes()).hexdigest()),
    )

    class Encoder:
        def encode(self, texts):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(few_shot_module, "_get_dense_encoder", lambda: Encoder())

    baseline = PersonaFewShotRetriever(path)
    candidate = PersonaFewShotRetriever(path, prefer_same_chat_after_situation=True)

    baseline_rows = baseline.retrieve(
        "你真厉害，太强了，大佬牛逼",
        "",
        is_group=False,
        limit=2,
        chat_id="chat_exact",
        interaction_context="handling_praise",
    )
    candidate_rows = candidate.retrieve(
        "你真厉害，太强了，大佬牛逼",
        "",
        is_group=False,
        limit=2,
        chat_id="chat_exact",
        interaction_context="handling_praise",
    )

    assert baseline_rows[0]["id"] == "other_chat_more_words"
    assert candidate_rows[0]["id"] == "same_chat"


def test_candidate_does_not_prioritize_same_chat_with_wrong_response_move(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {
            "id": "same_chat_wrong_setup",
            "relationship": "friend",
            "chat_id": "chat_exact",
            "context": ["他是真的厉害"],
            "reply": ["确实"],
        },
        {
            "id": "other_chat_right_setup",
            "relationship": "friend",
            "chat_id": "other",
            "context": ["你真厉害"],
            "reply": ["运气好"],
        },
    ])

    rows = PersonaFewShotRetriever(path, prefer_same_chat_after_situation=True).retrieve(
        "你真厉害",
        "",
        is_group=False,
        limit=2,
        chat_id="chat_exact",
        interaction_context="handling_praise",
    )

    assert rows[0]["id"] == "other_chat_right_setup"


def test_candidate_fills_from_other_chats_when_same_chat_has_too_few_examples(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "same", "relationship": "group", "chat_id": "chat_exact", "context": ["你真厉害"], "reply": ["少来"]},
        {"id": "other_a", "relationship": "group", "chat_id": "a", "context": ["大佬真强"], "reply": ["低调"]},
        {"id": "other_b", "relationship": "group", "chat_id": "b", "context": ["太厉害了"], "reply": ["运气"]},
    ])

    rows = PersonaFewShotRetriever(path, prefer_same_chat_after_situation=True).retrieve(
        "你真厉害",
        "",
        is_group=True,
        limit=3,
        chat_id="chat_exact",
        interaction_context="handling_praise",
    )

    assert rows[0]["id"] == "same"
    assert {row["id"] for row in rows} == {"same", "other_a", "other_b"}


def test_retrieval_scores_only_matching_situation_bucket(tmp_path, monkeypatch):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "praise", "relationship": "group", "chat_id": "a", "context": ["你真厉害"], "reply": ["低调"]},
        {"id": "question", "relationship": "group", "chat_id": "b", "context": ["晚上吃啥"], "reply": ["火锅"]},
        {"id": "private", "relationship": "friend", "chat_id": "c", "context": ["你真厉害"], "reply": ["还行"]},
    ])
    retriever = PersonaFewShotRetriever(path)

    candidates = retriever._candidate_rows(True, {"praise"})

    assert [row["id"] for row in candidates] == ["praise"]


def test_retrieval_falls_back_to_relationship_when_situation_bucket_is_missing(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "group_a", "relationship": "group", "chat_id": "a", "context": ["普通聊天"], "reply": ["嗯"]},
        {"id": "group_b", "relationship": "group", "chat_id": "b", "context": ["另一个话题"], "reply": ["行"]},
        {"id": "private", "relationship": "friend", "chat_id": "c", "context": ["普通聊天"], "reply": ["好"]},
    ])
    retriever = PersonaFewShotRetriever(path)

    candidates = retriever._candidate_rows(True, {"complaint"})

    assert [row["id"] for row in candidates] == ["group_a", "group_b"]


@pytest.mark.parametrize(
    ("text", "not_situation"),
    [
        ("我最近好像又瘦回来了", "arrival_status"),
        ("钉子理论上可以拔出来", "calling_presence"),
        ("聊聊你的分享欲", "share"),
        ("哈哈，今天吃什么", "teasing"),
    ],
)
def test_situation_detection_does_not_confuse_topic_words_with_social_situation(text, not_situation):
    from src.reply.few_shot import _situation_tags

    assert not_situation not in _situation_tags(text, "group_banter")


def test_situation_detection_treats_envy_you_all_as_self_deprecation():
    from src.reply.few_shot import _situation_tags

    assert _situation_tags("羡慕你们这些大佬") == {"self_deprecation"}


def test_situation_detection_does_not_treat_sleep_departure_as_self_deprecation():
    from src.reply.few_shot import _situation_tags

    assert "self_deprecation" not in _situation_tags("我不行了，我要去睡觉了")


def test_trailing_presence_phrase_does_not_override_established_topic():
    from src.reply.few_shot import _situation_tags

    text = "我们刚才一直在对比两家公司的收入和估值差距\n前面已经把利润率和用户数都算了一遍\n妈的，人呢"

    assert "calling_presence" not in _situation_tags(text)


def test_retrieval_prefers_listener_group_praise_with_self_lowering(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {
            "id": "generic_self_deprecation",
            "relationship": "group",
            "chat_id": "a",
            "context": ["我今天真的不行了"],
            "reply": ["别这么说"],
        },
        {
            "id": "praise_others_while_lowering_self",
            "relationship": "group",
            "chat_id": "b",
            "context": ["羡慕各位老板，我这种穷人只能看看"],
            "reply": ["你少来"],
        },
    ])

    rows = PersonaFewShotRetriever(path).retrieve(
        "羡慕你们这些资本家，我只能搬砖了",
        "群",
        is_group=True,
        limit=2,
    )

    assert rows[0]["id"] == "praise_others_while_lowering_self"


def test_retrieval_distinguishes_direct_praise_from_third_party_praise(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {
            "id": "third_party",
            "relationship": "friend",
            "chat_id": "a",
            "context": ["他是真的厉害"],
            "reply": ["确实"],
        },
        {
            "id": "direct",
            "relationship": "friend",
            "chat_id": "b",
            "context": ["恭喜你啊，太强了"],
            "reply": ["运气好"],
        },
    ])

    rows = PersonaFewShotRetriever(path).retrieve(
        "真羡慕你，太厉害了",
        "朋友",
        is_group=False,
        limit=2,
    )

    assert rows[0]["id"] == "direct"


@pytest.mark.parametrize(
    ("query", "expected_id", "rows"),
    [
        (
            "你可真会吹牛",
            "teasing",
            [
                {"id": "complaint", "context": ["吹牛这事烦死了"], "reply": ["确实烦"]},
                {"id": "teasing", "context": ["你可真能装"], "reply": ["跟你学的"]},
            ],
        ),
        (
            "晚上出来吃饭不",
            "invitation",
            [
                {"id": "question", "context": ["晚上吃什么"], "reply": ["火锅"]},
                {"id": "invitation", "context": ["晚上要不要一起吃"], "reply": ["几点"]},
            ],
        ),
        (
            "芊哥，出来冒个泡",
            "presence",
            [
                {"id": "invitation", "context": ["出来一起吃饭"], "reply": ["不去了"]},
                {"id": "presence", "context": ["人呢，出来冒个泡"], "reply": ["来了"]},
            ],
        ),
    ],
)
def test_retrieval_prefers_the_same_response_move_over_shared_words(tmp_path, query, expected_id, rows):
    path = tmp_path / "examples.jsonl"
    for index, row in enumerate(rows):
        row.update({"relationship": "group", "chat_id": str(index)})
    _write_rows(path, rows)

    selected = PersonaFewShotRetriever(path).retrieve(query, "群", is_group=True, limit=2)

    assert selected[0]["id"] == expected_id


@pytest.mark.parametrize(
    ("query", "expected_id", "rows"),
    [
        (
            "我到了，是304吗？",
            "address_confirmation",
            [
                {"id": "arrival_status", "context": ["我已经出发了，在路上"], "reply": ["好"]},
                {"id": "address_confirmation", "context": ["到了，是3号楼吗"], "reply": ["对"]},
            ],
        ),
        (
            "人呢，你现在在哪？",
            "location_followup",
            [
                {"id": "arrival_status", "context": ["我快到了"], "reply": ["行"]},
                {"id": "location_followup", "context": ["你到哪了，人呢"], "reply": ["门口"]},
            ],
        ),
        (
            "晚上一起吃饭不？",
            "invitation",
            [
                {"id": "task_request", "context": ["帮我带份晚饭"], "reply": ["行"]},
                {"id": "invitation", "context": ["周末一起吃饭吗"], "reply": ["几点"]},
            ],
        ),
        (
            "能不能帮我取一下快递",
            "task_request",
            [
                {"id": "invitation", "context": ["要不要一起去拿快递"], "reply": ["走"]},
                {"id": "task_request", "context": ["帮我拿一下外卖"], "reply": ["几楼"]},
            ],
        ),
        (
            "今天真是烦死了",
            "emotion_vent",
            [
                {"id": "task_request", "context": ["烦死了，帮我改一下"], "reply": ["发来"]},
                {"id": "emotion_vent", "context": ["今天真的好烦"], "reply": ["咋了"]},
            ],
        ),
        (
            "你可真能吹",
            "playful_teasing",
            [
                {"id": "emotion_vent", "context": ["他天天吹牛，烦死了"], "reply": ["别理他"]},
                {"id": "playful_teasing", "context": ["你又装起来了"], "reply": ["跟你学的"]},
            ],
        ),
        (
            "那你昨天说的那个后来怎么样了？",
            "thread_followup",
            [
                {"id": "plain_question", "context": ["这个怎么样"], "reply": ["还行"]},
                {"id": "thread_followup", "context": ["那你前面说的事后来呢"], "reply": ["解决了"]},
            ],
        ),
    ],
)
def test_precise_situation_matching_separates_similar_words_with_different_setups(
    tmp_path, query, expected_id, rows
):
    path = tmp_path / "examples.jsonl"
    for index, row in enumerate(rows):
        row.update({"relationship": "friend", "chat_id": str(index)})
    _write_rows(path, rows)

    selected = PersonaFewShotRetriever(path, enable_precise_situation=True).retrieve(
        query,
        "朋友",
        is_group=False,
        limit=2,
        use_situation=True,
    )

    assert selected[0]["id"] == expected_id


def test_precise_address_confirmation_does_not_confuse_time_confirmation():
    from src.reply.few_shot import _precise_situation

    assert _precise_situation("我到了，是304吗？") == "address_confirmation"
    assert _precise_situation("还没到时间呢，不是11？") == ""


def test_retrieval_clarifies_vague_problem_before_giving_generic_advice(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {
            "id": "generic_advice",
            "relationship": "group",
            "chat_id": "a",
            "context": ["翻译这个事好烦，不知道怎么办"],
            "reply": ["技术词直接留英文"],
            "situation_tags": ["complaint", "question"],
            "situation_profile": {"expected_response_move": "acknowledge_or_share_feeling"},
        },
        {
            "id": "clarify_first",
            "relationship": "group",
            "chat_id": "b",
            "context": ["有些词我也不知道怎么翻"],
            "reply": ["比如"],
            "situation_tags": ["complaint", "question"],
            "situation_profile": {"expected_response_move": "acknowledge_or_share_feeling"},
        },
    ])

    rows = PersonaFewShotRetriever(path).retrieve(
        "我好烦有些词我也不知道怎么翻好",
        "群",
        is_group=True,
        limit=2,
    )

    assert rows[0]["id"] == "clarify_first"


def test_human_clarification_reply_labels_the_example_by_response_move():
    from src.reply.few_shot import _row_situation_profile, _row_situation_tags

    row = {
        "context": ["我感觉我们做的事情很没有必要"],
        "reply": ["比如"],
        "situation_tags": ["complaint"],
        "situation_profile": {"expected_response_move": "acknowledge_or_share_feeling"},
    }

    assert "needs_clarification" in _row_situation_tags(row)
    assert _row_situation_profile(row)["expected_response_move"] == "clarify_detail"


def test_clarification_candidate_can_be_disabled_for_paired_baseline(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {
            "id": "clarify_first",
            "relationship": "group",
            "chat_id": "a",
            "context": ["我感觉我们做的事情很没有必要"],
            "reply": ["比如"],
            "situation_tags": ["complaint"],
            "situation_profile": {"expected_response_move": "acknowledge_or_share_feeling"},
        },
    ])

    baseline = PersonaFewShotRetriever(path, enable_clarification=False)
    candidate = PersonaFewShotRetriever(path, enable_clarification=True)

    baseline._load()
    assert (True, "needs_clarification") not in baseline._rows_by_group_situation
    assert [row["id"] for row in candidate._candidate_rows(True, {"needs_clarification"})] == ["clarify_first"]


def test_concrete_question_still_expects_a_direct_answer():
    from src.reply.few_shot import _situation_profile

    assert _situation_profile("这个词怎么翻")["expected_response_move"] == "direct_answer"


def test_unrelated_complaint_does_not_become_a_clarification_case():
    from src.reply.few_shot import _situation_profile, _situation_tags

    text = "今天翻译改了五遍，烦死了"
    assert "needs_clarification" not in _situation_tags(text)
    assert _situation_profile(text)["expected_response_move"] == "acknowledge_or_share_feeling"


@pytest.mark.parametrize(
    "text",
    [
        "我感觉我们做的事情很没有必要",
        "我觉得有很多解决办法",
        "我现在跟以前挺不一样的",
        "太坑了",
        "今天刚面的",
    ],
)
def test_verified_human_clarification_setups_ask_for_the_missing_detail(text):
    from src.reply.few_shot import _situation_profile

    assert _situation_profile(text)["expected_response_move"] == "clarify_detail"


@pytest.mark.parametrize(
    "text",
    [
        "这个词怎么翻",
        "今天刚面完，面试官问了三道算法题",
        "这个软件太坑了，付款以后还是打不开",
        "我现在跟以前不一样了，我已经不熬夜了",
    ],
)
def test_concrete_details_do_not_trigger_clarification_first(text):
    from src.reply.few_shot import _situation_tags

    assert "needs_clarification" not in _situation_tags(text)


def test_dense_retrieval_drops_low_similarity_examples(tmp_path, monkeypatch):
    path = tmp_path / "persona_examples.jsonl"
    _write_rows(path, [
        {"id": "high", "relationship": "friend", "chat_id": "a", "context": ["相关"], "reply": ["相关回复"]},
        {"id": "low", "relationship": "friend", "chat_id": "b", "context": ["无关"], "reply": ["无关回复"]},
    ])
    np.savez(
        tmp_path / "persona_embeddings.npz",
        ids=np.array(["high", "low"]),
        embeddings=np.array([[0.8, 0.6], [0.2, 0.98]], dtype=np.float32),
        examples_sha256=np.array(hashlib.sha256(path.read_bytes()).hexdigest()),
    )

    class Encoder:
        def encode(self, texts):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(few_shot_module, "_get_dense_encoder", lambda: Encoder())
    rows = PersonaFewShotRetriever(path).retrieve("相关查询", "朋友", is_group=False, limit=2)

    assert [row["id"] for row in rows] == ["high"]


def test_candidate_can_increase_semantic_topic_weight(tmp_path, monkeypatch):
    path = tmp_path / "persona_examples.jsonl"
    _write_rows(path, [
        {"id": "word_match", "relationship": "friend", "chat_id": "a", "context": ["股票怎么办"], "reply": ["先看看"]},
        {"id": "topic_match", "relationship": "friend", "chat_id": "b", "context": ["市场下跌处理"], "reply": ["先别动"]},
    ])
    np.savez(
        tmp_path / "persona_embeddings.npz",
        ids=np.array(["word_match", "topic_match"]),
        embeddings=np.array([[0.6, 0.8], [0.9, 0.43589]], dtype=np.float32),
        examples_sha256=np.array(hashlib.sha256(path.read_bytes()).hexdigest()),
    )

    class Encoder:
        def encode(self, texts):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(few_shot_module, "_get_dense_encoder", lambda: Encoder())
    monkeypatch.setattr(few_shot_module, "_situation_tags", lambda *args, **kwargs: set())
    monkeypatch.setattr(few_shot_module, "_situation_profile", lambda *args, **kwargs: {})

    baseline = PersonaFewShotRetriever(path).retrieve("股票怎么办", "朋友", is_group=False, limit=1)
    candidate = PersonaFewShotRetriever(
        path,
        increase_semantic_topic_weight=True,
    ).retrieve("股票怎么办", "朋友", is_group=False, limit=1)

    assert [row["id"] for row in baseline] == ["word_match"]
    assert [row["id"] for row in candidate] == ["topic_match"]


def test_dense_retrieval_falls_back_when_every_candidate_is_below_threshold(tmp_path, monkeypatch):
    path = tmp_path / "persona_examples.jsonl"
    _write_rows(path, [
        {"id": "closer", "relationship": "group", "chat_id": "a", "context": ["羡慕你们这些车主"], "reply": ["你也买"]},
        {"id": "farther", "relationship": "group", "chat_id": "b", "context": ["羡慕你们这些老板"], "reply": ["少来"]},
    ])
    np.savez(
        tmp_path / "persona_embeddings.npz",
        ids=np.array(["closer", "farther"]),
        embeddings=np.array([[0.4, 0.9165], [0.3, 0.9539]], dtype=np.float32),
        examples_sha256=np.array(hashlib.sha256(path.read_bytes()).hexdigest()),
    )

    class Encoder:
        def encode(self, texts):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(few_shot_module, "_get_dense_encoder", lambda: Encoder())
    rows = PersonaFewShotRetriever(path).retrieve("羡慕你们这些车主", "群", is_group=True, limit=1)

    assert [row["id"] for row in rows] == ["closer"]


def test_matching_social_setup_can_override_low_word_similarity(tmp_path, monkeypatch):
    path = tmp_path / "persona_examples.jsonl"
    _write_rows(path, [
        {
            "id": "same_setup",
            "relationship": "group",
            "chat_id": "a",
            "context": ["羡慕你们这些车主"],
            "reply": ["租的车"],
            "situation_tags": ["self_deprecation"],
            "situation_profile": {
                "praise_target": "listener_group",
                "speaker_posture": "self_lowering",
                "expected_response_move": "downplay_or_counter_praise",
            },
        },
        {
            "id": "word_match_only",
            "relationship": "group",
            "chat_id": "b",
            "context": ["90后又亏麻了"],
            "reply": ["抄底"],
            "situation_tags": ["self_deprecation"],
        },
    ])
    np.savez(
        tmp_path / "persona_embeddings.npz",
        ids=np.array(["same_setup", "word_match_only"]),
        embeddings=np.array([[0.3, 0.9539], [0.8, 0.6]], dtype=np.float32),
        examples_sha256=np.array(hashlib.sha256(path.read_bytes()).hexdigest()),
    )

    class Encoder:
        def encode(self, texts):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(few_shot_module, "_get_dense_encoder", lambda: Encoder())
    rows = PersonaFewShotRetriever(path).retrieve("羡慕你们90后", "群", is_group=True, limit=2)

    assert rows[0]["id"] == "same_setup"


def test_private_retrieval_excludes_group_examples(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "group", "relationship": "group", "chat_id": "g", "context": ["在吗"], "reply": ["咋"]},
        {"id": "friend", "relationship": "friend", "chat_id": "f", "context": ["在吗"], "reply": ["咋啦"]},
    ])

    rows = PersonaFewShotRetriever(path).retrieve("在吗", "朋友", is_group=False, limit=8)

    assert [row["id"] for row in rows] == ["friend"]


def test_latest_turn_situation_ignores_older_context_when_matching(tmp_path, monkeypatch):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "old", "relationship": "friend", "chat_id": "a", "context": ["旧话题关键词", "结束陈述"], "reply": ["接旧话题"]},
        {"id": "latest", "relationship": "friend", "chat_id": "b", "context": ["无关前文", "现在问句"], "reply": ["回答问题"]},
    ])

    def fake_tags(text, *args, **kwargs):
        if text == "现在问句":
            return {"latest"}
        if text == "结束陈述":
            return {"old"}
        if "旧话题关键词" in text:
            return {"old"}
        return set()

    monkeypatch.setattr(few_shot_module, "_situation_tags", fake_tags)
    baseline = PersonaFewShotRetriever(path).retrieve(
        "旧话题关键词\n现在问句", "朋友", is_group=False, limit=1
    )
    candidate = PersonaFewShotRetriever(path, prioritize_latest_turn=True).retrieve(
        "旧话题关键词\n现在问句", "朋友", is_group=False, limit=1
    )

    assert [row["id"] for row in baseline] == ["old"]
    assert [row["id"] for row in candidate] == ["latest"]


def test_response_move_can_rank_before_topic_relevance(tmp_path, monkeypatch):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {
            "id": "same_words_wrong_move",
            "relationship": "friend",
            "chat_id": "a",
            "context": ["股票跌了怎么办"],
            "reply": ["先安慰你"],
            "situation_profile": {
                "expected_response_move": "acknowledge_or_share_feeling",
                "praise_target": "listener",
                "speaker_posture": "self_lowering",
            },
        },
        {
            "id": "right_move_less_relevant",
            "relationship": "friend",
            "chat_id": "b",
            "context": ["这个问题怎么处理"],
            "reply": ["直接回答"],
            "situation_profile": {"expected_response_move": "direct_answer"},
        },
        {
            "id": "right_move_more_relevant",
            "relationship": "friend",
            "chat_id": "c",
            "context": ["股票该怎么办"],
            "reply": ["直接回答股票问题"],
            "situation_profile": {"expected_response_move": "direct_answer"},
        },
    ])
    monkeypatch.setattr(few_shot_module, "_situation_tags", lambda *args, **kwargs: {"question"})
    monkeypatch.setattr(
        few_shot_module,
        "_situation_profile",
        lambda *args, **kwargs: {
            "expected_response_move": "direct_answer",
            "praise_target": "listener",
            "speaker_posture": "self_lowering",
        },
    )

    baseline = PersonaFewShotRetriever(path).retrieve(
        "股票跌了怎么办", "朋友", is_group=False, limit=3, chat_id="a"
    )
    candidate = PersonaFewShotRetriever(
        path,
        prefer_response_move_before_relevance=True,
    ).retrieve("股票跌了怎么办", "朋友", is_group=False, limit=3, chat_id="a")

    assert baseline[0]["id"] == "same_words_wrong_move"
    assert [row["id"] for row in candidate] == [
        "right_move_more_relevant",
        "right_move_less_relevant",
        "same_words_wrong_move",
    ]


def test_latest_turn_and_same_chat_priority_can_be_combined(tmp_path, monkeypatch):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "same_chat_latest", "relationship": "group", "chat_id": "chat_exact", "context": ["旧话题", "现在问句"], "reply": ["同群回答"]},
        {"id": "other_chat_latest", "relationship": "group", "chat_id": "other", "context": ["无关前文", "现在问句"], "reply": ["其他群回答"]},
        {"id": "same_chat_old", "relationship": "group", "chat_id": "chat_exact", "context": ["旧话题关键词", "结束陈述"], "reply": ["接旧话题"]},
    ])

    def fake_tags(text, *args, **kwargs):
        if text == "现在问句":
            return {"latest"}
        if text == "结束陈述":
            return {"old"}
        if "旧话题关键词" in text:
            return {"old"}
        return set()

    monkeypatch.setattr(few_shot_module, "_situation_tags", fake_tags)
    rows = PersonaFewShotRetriever(
        path,
        prioritize_latest_turn=True,
        prefer_same_chat_after_situation=True,
    ).retrieve(
        "旧话题关键词\n现在问句",
        "",
        is_group=True,
        limit=2,
        chat_id="chat_exact",
    )

    assert [row["id"] for row in rows] == ["same_chat_latest", "other_chat_latest"]


def test_sparse_turn_avoids_specific_topic_examples_but_keeps_short_human_examples(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "car_fact", "relationship": "group", "chat_id": "a", "context": ["车+车牌+车位，大几十万，亏麻了"], "reply": ["真顶不住"]},
        {"id": "short_vent", "relationship": "group", "chat_id": "b", "context": ["亏麻了"], "reply": ["惨"]},
        {"id": "brief_presence", "relationship": "group", "chat_id": "c", "context": ["人呢"], "reply": ["在呢"]},
    ])

    retriever = PersonaFewShotRetriever(
        path,
        prioritize_latest_turn=True,
        avoid_concrete_examples_for_sparse_query=True,
    )
    loss_rows = retriever.retrieve("亏麻了", "群", is_group=True, limit=2)
    presence_rows = retriever.retrieve("人呢", "群", is_group=True, limit=2)

    assert "car_fact" not in [row["id"] for row in loss_rows]
    assert "short_vent" in [row["id"] for row in loss_rows]
    assert "brief_presence" in [row["id"] for row in presence_rows]


def test_sparse_turn_avoids_replies_that_depend_on_hidden_context(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "pk", "relationship": "group", "chat_id": "a", "context": ["感觉情绪已经崩了"], "reply": ["我感觉pk挺好的"]},
        {"id": "banned", "relationship": "group", "chat_id": "b", "context": ["我真的要累死了"], "reply": ["发现居然被禁售了", "hhh"]},
        {"id": "portable", "relationship": "group", "chat_id": "c", "context": ["太惨了"], "reply": ["真的假的", "这么可怕"]},
    ])

    rows = PersonaFewShotRetriever(
        path,
        prioritize_latest_turn=True,
        avoid_concrete_examples_for_sparse_query=True,
    ).retrieve("难受", "群", is_group=True, limit=3)

    assert [row["id"] for row in rows] == ["portable"]


def test_sparse_praise_avoids_replies_that_assume_the_old_topic(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "benefits", "relationship": "group", "chat_id": "a", "context": ["羡慕"], "reply": ["其他还有N多福利"]},
        {"id": "travel", "relationship": "group", "chat_id": "b", "context": ["羡慕"], "reply": ["你们啥时候去"]},
        {"id": "portable", "relationship": "group", "chat_id": "c", "context": ["羡慕"], "reply": ["少来"]},
    ])

    rows = PersonaFewShotRetriever(
        path,
        prioritize_latest_turn=True,
        avoid_concrete_examples_for_sparse_query=True,
    ).retrieve("羡慕", "群", is_group=True, limit=3)

    assert [row["id"] for row in rows] == ["portable"]


def test_sparse_turn_rejects_short_replies_that_still_depend_on_old_context(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "average", "relationship": "group", "chat_id": "a", "context": ["没有覆盖到，难受"], "reply": ["被平均了么？"]},
        {"id": "restaurant", "relationship": "group", "chat_id": "b", "context": ["啊康没了，很难过"], "reply": ["东北饺子王", "还有啥"]},
        {"id": "loss", "relationship": "group", "chat_id": "c", "context": ["真的是难受"], "reply": ["你亏了多少"]},
        {"id": "portable", "relationship": "group", "chat_id": "d", "context": ["太惨了"], "reply": ["咋了", "没事吧"]},
    ])

    rows = PersonaFewShotRetriever(
        path,
        prioritize_latest_turn=True,
        avoid_concrete_examples_for_sparse_query=True,
    ).retrieve("难受", "群", is_group=True, limit=4)

    assert [row["id"] for row in rows] == ["portable"]


def test_sparse_praise_keeps_natural_portable_phrases_without_old_topic_facts(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "weight", "relationship": "group", "chat_id": "a", "context": ["羡慕"], "reply": ["好重现在"]},
        {"id": "layout", "relationship": "group", "chat_id": "b", "context": ["羡慕"], "reply": ["一个户型的"]},
        {"id": "meal", "relationship": "group", "chat_id": "c", "context": ["羡慕"], "reply": ["羡慕", "我也想吃", "下班了"]},
        {"id": "portable", "relationship": "group", "chat_id": "d", "context": ["羡慕"], "reply": ["少来", "哪有"]},
    ])

    rows = PersonaFewShotRetriever(
        path,
        prioritize_latest_turn=True,
        avoid_concrete_examples_for_sparse_query=True,
    ).retrieve("羡慕", "群", is_group=True, limit=4)

    assert [row["id"] for row in rows] == ["portable"]


def test_sparse_turn_keeps_normal_brief_actions():
    from src.reply.few_shot import _has_portable_sparse_reply

    for reply in ("问问", "问下", "确认下", "看看", "在呢"):
        assert _has_portable_sparse_reply({"reply": [reply]})


def test_brief_exclamation_matches_reaction_not_word_fragments(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "fitness", "relationship": "group", "chat_id": "same", "context": ["这个健身操我也跳过"], "reply": ["确实"]},
        {"id": "operation", "relationship": "group", "chat_id": "same", "context": ["我没操作"], "reply": ["看看"]},
        {"id": "reaction", "relationship": "group", "chat_id": "other", "context": ["卧槽"], "reply": ["怎么了"]},
    ])

    rows = PersonaFewShotRetriever(
        path,
        prioritize_latest_turn=True,
        prefer_same_chat_after_situation=True,
        avoid_concrete_examples_for_sparse_query=True,
    ).retrieve("我操", "群", is_group=True, limit=3, chat_id="same")

    assert [row["id"] for row in rows] == ["reaction"]


def test_brief_exclamation_does_not_reclassify_natural_short_phrases():
    from src.reply.few_shot import _situation_tags

    for text in ("问问", "问下", "确认下", "在吗"):
        assert "brief_exclamation" not in _situation_tags(text)


def test_latest_turn_situation_falls_back_after_excluding_only_bucket_match(tmp_path, monkeypatch):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "current", "relationship": "group", "chat_id": "a", "context": ["出来pk"], "reply": ["来"]},
        {"id": "fallback", "relationship": "group", "chat_id": "b", "context": ["普通聊天"], "reply": ["行"]},
        {"id": "private", "relationship": "friend", "chat_id": "c", "context": ["出来pk"], "reply": ["来"]},
    ])

    monkeypatch.setattr(
        few_shot_module,
        "_situation_tags",
        lambda text, *args, **kwargs: {"calling_presence"} if text == "出来pk" else set(),
    )
    rows = PersonaFewShotRetriever(path, prioritize_latest_turn=True).retrieve(
        "出来pk",
        "群",
        is_group=True,
        limit=2,
        exclude_ids={"current"},
    )

    assert [row["id"] for row in rows] == ["fallback"]


def test_retrieval_can_exclude_the_current_development_case(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "current", "relationship": "group", "chat_id": "a", "context": ["羡慕你们这些老板"], "reply": ["笑死"]},
        {"id": "other", "relationship": "group", "chat_id": "b", "context": ["羡慕各位老板"], "reply": ["少来"]},
    ])

    rows = PersonaFewShotRetriever(path).retrieve(
        "羡慕你们这些老板",
        "群",
        is_group=True,
        exclude_ids={"current"},
    )

    assert [row["id"] for row in rows] == ["other"]


def test_leave_one_out_wrapper_excludes_current_case(tmp_path):
    from scripts.run_situation_c0_ab import LeaveOneOutRetriever

    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "current", "relationship": "group", "chat_id": "a", "context": ["我好烦有些词不知道怎么翻"], "reply": ["比如"]},
        {"id": "other", "relationship": "group", "chat_id": "b", "context": ["有些地方不知道怎么写"], "reply": ["哪部分"]},
    ])
    retriever = LeaveOneOutRetriever(PersonaFewShotRetriever(path))
    retriever.exclude_ids = {"current"}

    rows = retriever.retrieve("我好烦有些词不知道怎么翻", "群", is_group=True, limit=2)
    _, ids = retriever.render(rows, max_chars=500)

    assert ids == ["other"]


def test_render_has_fact_isolation_and_budget(tmp_path):
    rows = [
        {"id": "one", "context": ["在吗"], "reply": ["咋啦"]},
        {"id": "two", "context": ["x" * 100], "reply": ["y" * 100]},
    ]

    content, ids = PersonaFewShotRetriever.render(rows, max_chars=500)

    assert ids == ["one"]
    assert "不是当前对话事实" in content
    assert "不得复制示例中的具体笑点" in content
    assert "不能覆盖 consumed_self_replies" in content
    assert "<response>\n<message>咋啦</message>\n</response>" in content


def test_render_keeps_named_continuous_context_and_escapes_xml():
    rows = [{
        "id": "group_one",
        "context": ["旧格式不会被选"],
        "reply": ["走起"],
        "context_messages": [
            {"sender": "小周", "text": "吃什么"},
            {"sender": "我", "text": "都行"},
            {"sender": "小李 & 老王", "text": "火锅 <可以>"},
        ],
        "reply_messages": [{"sender": "我", "text": "走起"}],
    }]

    content, ids = PersonaFewShotRetriever.render(rows, max_chars=1000)

    assert ids == ["group_one"]
    assert '<message sender="小周">吃什么</message>' in content
    assert '<message sender="我">都行</message>' in content
    assert '<message sender="小李 &amp; 老王">火锅 &lt;可以&gt;</message>' in content
    assert '<message sender="我">走起</message>' in content


def test_render_selected_keeps_ranked_ids_but_uses_named_context(tmp_path):
    ranked_path = tmp_path / "ranked.jsonl"
    render_path = tmp_path / "named.jsonl"
    _write_rows(ranked_path, [{
        "id": "same_id",
        "relationship": "group",
        "chat_id": "chat_one",
        "context": ["短上下文"],
        "reply": ["短回复"],
    }])
    _write_rows(render_path, [{
        "id": "same_id",
        "relationship": "group",
        "chat_id": "chat_one",
        "context": ["第一句", "第二句"],
        "reply": ["真人回复"],
        "context_messages": [
            {"sender": "小周", "text": "第一句"},
            {"sender": "小李", "text": "第二句"},
        ],
        "reply_messages": [{"sender": "我", "text": "真人回复"}],
    }])
    retriever = PersonaFewShotRetriever(ranked_path, render_path=render_path)
    ranked = retriever.retrieve("短上下文", "群聊", is_group=True, limit=1)

    content, ids = retriever.render_selected(ranked, max_chars=1000)

    assert ids == ["same_id"]
    assert '<message sender="小周">第一句</message>' in content
    assert '<message sender="小李">第二句</message>' in content
    assert '<message sender="我">真人回复</message>' in content


def test_missing_file_degrades_to_empty(tmp_path):
    retriever = PersonaFewShotRetriever(tmp_path / "missing.jsonl")

    assert retriever.retrieve("你好", "朋友", is_group=False) == []


def test_generator_injects_examples_and_records_ids(tmp_path, monkeypatch):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "style_one", "relationship": "friend", "chat_id": "f", "context": ["在吗"], "reply": ["咋啦"]},
    ])
    monkeypatch.setenv("PERSONA_FEW_SHOT_PATH", str(path))
    monkeypatch.setenv("ENABLE_PERSONA_FEW_SHOTS", "1")
    monkeypatch.setenv("PERSONA_FEW_SHOT_ALLOW_UNREVIEWED", "1")

    class LLM:
        def __init__(self):
            self.responses = ['{"skills": []}', '{"replies": ["在的"]}']

        def chat(self, messages, tools=None, **kwargs):
            return self.responses.pop(0)

    message = ChatMessage(text="在吗", sender="朋友", sender_type=SenderType.OTHER, chat_name="朋友")
    generator = ReplyGenerator(llm_client=LLM())
    generator.enable_self_refine = False
    retrieve = generator.persona_few_shot_retriever.retrieve
    retrieval_options = []

    def record_retrieve(*args, **kwargs):
        retrieval_options.append(kwargs["use_situation"])
        return retrieve(*args, **kwargs)

    generator.persona_few_shot_retriever.retrieve = record_retrieve

    replies = generator.generate([message], [message], is_group=False)

    assert replies == ["在的"]
    assert generator.last_few_shot_ids == ["style_one"]
    assert "真人本人历史回复" in generator.last_user_prompt
    request = ET.fromstring(generator.last_user_prompt)
    assert request.find('./active_skill[@name="casual_chat"]/skill') is not None
    assert request.find("./style_examples/example") is not None
    assert any(item.get("type") == "persona_few_shot" for item in generator.last_generation_trace)
    assert "咋啦" not in generator.text_for_logging(generator.last_user_prompt)
    assert "style_one" in generator.text_for_logging(generator.last_user_prompt)
    assert retrieval_options == [True]


def test_latest_request_candidate_adds_fact_boundary_after_examples(tmp_path, monkeypatch):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {
            "id": "specific_fact",
            "relationship": "friend",
            "chat_id": "f",
            "context": ["你到了么？"],
            "reply": ["到了，政策是补贴20%"],
        },
    ])
    monkeypatch.setenv("PERSONA_FEW_SHOT_PATH", str(path))
    monkeypatch.setenv("ENABLE_PERSONA_FEW_SHOTS", "1")
    monkeypatch.setenv("PERSONA_FEW_SHOT_ALLOW_UNREVIEWED", "1")

    class LLM:
        def __init__(self):
            self.responses = ['{"skills": []}', '{"replies": ["还没"]}']

        def chat(self, messages, tools=None, **kwargs):
            return self.responses.pop(0)

    message = ChatMessage(
        text="你到了么？",
        sender="朋友",
        sender_type=SenderType.OTHER,
        chat_name="朋友",
    )
    generator = ReplyGenerator(llm_client=LLM(), enable_latest_request_first=True)
    generator.enable_self_refine = False

    assert generator.generate([message], [message], is_group=False) == ["还没"]

    request = ET.fromstring(generator.last_user_prompt)
    boundary = request.find("./style_example_usage")
    assert [child.tag for child in request][-3:] == [
        "style_examples",
        "style_example_usage",
        "reply_guard",
    ]
    assert boundary is not None
    assert boundary.attrib == {"priority": "final", "enforcement": "hard"}
    boundary_text = "".join(boundary.itertext())
    assert "其中的人物、事实、肯定或否定状态、理由、数字、计划和后续动作一律不能带入当前回复" in boundary_text
    assert "先只根据当前 unread、history、可信人设、可靠记忆或工具决定“回什么”" in boundary_text
    assert "再用 style_examples 决定“怎么说”" in boundary_text
    assert "删掉 style_examples 后无法得出的内容必须删除" in boundary_text


def test_current_baseline_does_not_add_style_example_fact_boundary(tmp_path, monkeypatch):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {
            "id": "specific_fact",
            "relationship": "friend",
            "chat_id": "f",
            "context": ["你到了么？"],
            "reply": ["到了"],
        },
    ])
    monkeypatch.setenv("PERSONA_FEW_SHOT_PATH", str(path))
    monkeypatch.setenv("ENABLE_PERSONA_FEW_SHOTS", "1")
    monkeypatch.setenv("PERSONA_FEW_SHOT_ALLOW_UNREVIEWED", "1")

    class LLM:
        def __init__(self):
            self.responses = ['{"skills": []}', '{"replies": ["还没"]}']

        def chat(self, messages, tools=None, **kwargs):
            return self.responses.pop(0)

    message = ChatMessage(
        text="你到了么？",
        sender="朋友",
        sender_type=SenderType.OTHER,
        chat_name="朋友",
    )
    generator = ReplyGenerator(llm_client=LLM(), enable_latest_request_first=False)
    generator.enable_self_refine = False

    assert generator.generate([message], [message], is_group=False) == ["还没"]
    request = ET.fromstring(generator.last_user_prompt)
    assert request.find("./style_examples/example") is not None
    assert request.find("./style_example_usage") is None


def test_generator_uses_situation_library_only_for_group_chat(tmp_path, monkeypatch):
    private_path = tmp_path / "private.jsonl"
    group_path = tmp_path / "group.jsonl"
    _write_rows(private_path, [
        {"id": "private_style", "relationship": "friend", "chat_id": "f", "context": ["真厉害"], "reply": ["还行"]},
    ])
    _write_rows(group_path, [
        {"id": "group_style", "relationship": "group", "chat_id": "g", "context": ["你们真厉害"], "reply": ["少来"]},
    ])
    monkeypatch.setenv("PERSONA_FEW_SHOT_PATH", str(private_path))
    monkeypatch.setenv("GROUP_PERSONA_FEW_SHOT_PATH", str(group_path))
    monkeypatch.setenv("ENABLE_PERSONA_FEW_SHOTS", "1")
    monkeypatch.setenv("PERSONA_FEW_SHOT_ALLOW_UNREVIEWED", "1")

    class LLM:
        def __init__(self):
            self.responses = ['{"skills": ["handling_praise"]}', '{"replies": ["哪有"]}']

        def chat(self, messages, tools=None, **kwargs):
            return self.responses.pop(0)

    message = ChatMessage(text="你们真厉害", sender="群友", sender_type=SenderType.OTHER, chat_name="群")
    generator = ReplyGenerator(llm_client=LLM())
    generator.enable_self_refine = False
    retrieve = generator.group_persona_few_shot_retriever.retrieve
    retrieval_options = []

    def record_retrieve(*args, **kwargs):
        retrieval_options.append(kwargs["use_situation"])
        return retrieve(*args, **kwargs)

    generator.group_persona_few_shot_retriever.retrieve = record_retrieve
    replies = generator.generate([message], [message], is_group=True)

    assert replies == ["哪有"]
    assert generator.last_few_shot_ids == ["group_style"]
    assert retrieval_options == [True]


def test_unapproved_report_is_not_ready(tmp_path):
    path = tmp_path / "persona_examples.jsonl"
    path.write_text("", encoding="utf-8")
    (tmp_path / "report.json").write_text('{"review_status":"pending"}', encoding="utf-8")

    assert PersonaFewShotRetriever(path).is_approved() is False


def test_untrusted_examples_are_never_retrieved(tmp_path):
    path = tmp_path / "persona_examples.jsonl"
    _write_rows(path, [
        {"id": "human", "relationship": "friend", "context": ["在吗"], "reply": ["在"]},
        {"id": "bot", "relationship": "friend", "context": ["在吗"], "reply": ["又是旧梗"], "source_provenance": "unverified"},
    ])

    rows = PersonaFewShotRetriever(path).retrieve("在吗", "朋友", is_group=False)

    assert [row["id"] for row in rows] == ["human"]


def test_approval_requires_trusted_human_provenance(tmp_path):
    path = tmp_path / "persona_examples.jsonl"
    path.write_text(json.dumps({
        "id": "legacy", "relationship": "friend", "context": ["在吗"], "reply": ["在"]
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    examples_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "report.json").write_text(json.dumps({
        "review_status": "approved", "examples_sha256": examples_sha256
    }), encoding="utf-8")

    assert PersonaFewShotRetriever(path).is_approved() is False


def test_approval_and_embeddings_are_bound_to_examples_hash(tmp_path):
    path = tmp_path / "persona_examples.jsonl"
    _write_rows(path, [{"id": "one", "relationship": "friend", "context": ["在吗"], "reply": ["在"]}])
    examples_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "report.json").write_text(
        json.dumps({"review_status": "approved", "examples_sha256": examples_sha256}),
        encoding="utf-8",
    )
    np.savez(
        tmp_path / "persona_embeddings.npz",
        ids=np.array(["one"]),
        embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        examples_sha256=np.array(examples_sha256),
    )
    retriever = PersonaFewShotRetriever(path)

    assert retriever.is_approved() is True
    assert retriever._load_embeddings()

    _write_rows(path, [{"id": "one", "relationship": "friend", "context": ["变了"], "reply": ["嗯"]}])

    assert retriever.is_approved() is False
    assert retriever._load_embeddings() == {}


def test_relative_few_shot_path_resolves_from_project_root(monkeypatch):
    monkeypatch.setenv("PERSONA_FEW_SHOT_PATH", "data/few_shot_v4/persona_examples.jsonl")

    generator = ReplyGenerator()

    assert generator.persona_few_shot_retriever.path.is_absolute()


def test_information_gap_response_move_is_opt_in():
    text = "听我叔叔说\n李一男回流的时候带回来好多老员工"

    assert "expected_response_move" not in few_shot_module._situation_profile(text)
    assert few_shot_module._situation_profile(
        text,
        enable_information_gap_response_move=True,
    )["expected_response_move"] == "verify_or_request_detail"


def test_information_gap_reply_overrides_explicit_direct_answer():
    row = {
        "context": ["说最近有裁员的可能？"],
        "reply": ["你听谁说的"],
        "situation_profile": {"expected_response_move": "direct_answer"},
    }

    assert few_shot_module._row_situation_profile(
        row,
        enable_information_gap_response_move=True,
    )["expected_response_move"] == "verify_or_request_detail"


def test_information_gap_detects_incomplete_identifier_on_last_line():
    assert few_shot_module._has_information_gap("快到了\n9Y956")


def test_bad_json_line_is_skipped_instead_of_emptying_library(tmp_path):
    """一行坏 JSON 只跳过该行，不得静默清空整个示例库。"""
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {
            "id": "good_row",
            "relationship": "friend",
            "chat_id": "a",
            "context": ["这事情真的很难受"],
            "reply": ["确实难受"],
        },
    ])
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"id": "broken", "context": [\n')  # 截断的坏行

    retriever = PersonaFewShotRetriever(path)
    rows = retriever._load()
    assert [row["id"] for row in rows] == ["good_row"]
    # 展示库同样只跳过坏行（render_path 是独立构造参数）
    examples_path = tmp_path / "examples.jsonl"
    _write_rows(examples_path, [
        {
            "id": "base_row",
            "relationship": "friend",
            "chat_id": "a",
            "context": ["ctx"],
            "reply": ["rep"],
        },
    ])
    render_path = tmp_path / "render.jsonl"
    _write_rows(render_path, [
        {
            "id": "render_good",
            "relationship": "friend",
            "chat_id": "a",
            "context": ["ctx"],
            "reply": ["rep"],
        },
    ])
    with render_path.open("a", encoding="utf-8") as handle:
        handle.write("not-json-at-all\n")
    render_retriever = PersonaFewShotRetriever(examples_path, render_path=render_path)
    assert list(render_retriever._load_render_rows()) == ["render_good"]
