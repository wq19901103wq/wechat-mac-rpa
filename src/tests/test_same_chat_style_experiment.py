import json

from scripts.build_verified_same_chat_style_view import _identity
from scripts.replay_same_chat_style_ab import RelevantSameChatRetriever, StrictSameChatRetriever


def test_identity_includes_relationship_chat_context_and_reply():
    row = {"relationship": "group", "chat_id": "chat_a", "context": ["前文"], "reply": ["回复"]}
    assert _identity(row) == ("group", "chat_a", ("前文",), ("回复",))


def test_strict_retriever_never_returns_cross_chat(tmp_path):
    rows = [
        {
            "id": "same",
            "relationship": "group",
            "chat_id": "chat_exact",
            "context": ["不太相关"],
            "reply": ["同群回复"],
            "source_provenance": "before_automation_cutoff",
        },
        {
            "id": "cross",
            "relationship": "group",
            "chat_id": "chat_other",
            "context": ["完全命中查询"],
            "reply": ["跨群回复"],
            "source_provenance": "before_automation_cutoff",
        },
    ]
    path = tmp_path / "persona_examples.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    selected = StrictSameChatRetriever(path).retrieve(
        "完全命中查询", "", is_group=True, chat_id="chat_exact", limit=3
    )
    assert [row["id"] for row in selected] == ["same"]


def test_relevant_retriever_allows_zero_instead_of_irrelevant_same_chat_rows(tmp_path):
    rows = [
        {
            "id": "work",
            "relationship": "group",
            "chat_id": "chat_exact",
            "context": ["年终奖和股票总包"],
            "reply": ["拿完期权"],
            "humor_type": "none",
            "response_mode": "neutral",
            "source_provenance": "before_automation_cutoff",
        },
    ]
    path = tmp_path / "persona_examples.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    selected = RelevantSameChatRetriever(path).retrieve(
        "绿藻头请大家喝奶茶", "", is_group=True, chat_id="chat_exact", limit=2
    )
    assert selected == []


def test_relevant_retriever_selects_at_most_one_topic_example(tmp_path):
    rows = [
        {
            "id": "milk_1",
            "relationship": "group",
            "chat_id": "chat_exact",
            "context": ["绿藻头请大家喝奶茶"],
            "reply": ["让绿藻头先请"],
            "humor_type": "none",
            "response_mode": "neutral",
            "source_provenance": "before_automation_cutoff",
        },
        {
            "id": "milk_2",
            "relationship": "group",
            "chat_id": "chat_exact",
            "context": ["第一杯秋天的奶茶谁请"],
            "reply": ["谁赚了谁请"],
            "humor_type": "none",
            "response_mode": "neutral",
            "source_provenance": "before_automation_cutoff",
        },
    ]
    path = tmp_path / "persona_examples.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    selected = RelevantSameChatRetriever(path).retrieve(
        "绿藻头请群友第一杯秋天的奶茶", "", is_group=True, chat_id="chat_exact", limit=2
    )
    assert len(selected) == 1
    assert selected[0]["retrieval_slot"] == "topic"


def test_relevant_retriever_does_not_cross_chat_or_relationship(tmp_path):
    rows = [
        {
            "id": "same_private",
            "relationship": "private",
            "chat_id": "chat_exact",
            "context": ["今晚一起吃火锅"],
            "reply": ["走起"],
            "humor_type": "none",
            "response_mode": "sincere",
            "source_provenance": "before_automation_cutoff",
        },
        {
            "id": "other_group",
            "relationship": "group",
            "chat_id": "chat_other",
            "context": ["今晚一起吃火锅"],
            "reply": ["走起"],
            "humor_type": "none",
            "response_mode": "sincere",
            "source_provenance": "before_automation_cutoff",
        },
    ]
    path = tmp_path / "persona_examples.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    selected = RelevantSameChatRetriever(path).retrieve(
        "今晚一起吃火锅", "", is_group=True, chat_id="chat_exact", limit=2
    )
    assert selected == []


def test_relevant_retriever_never_returns_more_than_two(tmp_path):
    rows = [
        {
            "id": f"hotpot_{index}",
            "relationship": "group",
            "chat_id": "chat_exact",
            "context": [f"今晚一起吃火锅第{index}次"],
            "reply": ["走起"],
            "humor_type": "wordplay" if index else "none",
            "response_mode": "playful" if index else "neutral",
            "source_provenance": "before_automation_cutoff",
        }
        for index in range(4)
    ]
    path = tmp_path / "persona_examples.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    selected = RelevantSameChatRetriever(path).retrieve(
        "今晚一起吃火锅第3次", "", is_group=True, chat_id="chat_exact", limit=8
    )
    assert len(selected) <= 2
