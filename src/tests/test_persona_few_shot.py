import hashlib
import json

import numpy as np
import src.reply.few_shot as few_shot_module

from src.models.base import ChatMessage, SenderType
from src.reply.few_shot import PersonaFewShotRetriever, _query_response_mode, resolve_relationship
from src.reply.generator import ReplyGenerator


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


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

    assert [row["id"] for row in rows] == ["relevant", "same"]


def test_retrieval_limits_repetitive_laugh_examples(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": f"laugh_{i}", "relationship": "friend", "chat_id": f"c{i}", "context": ["今天真离谱"], "reply": [f"哈哈哈{i}"]}
        for i in range(5)
    ] + [
        {"id": "plain", "relationship": "friend", "chat_id": "plain", "context": ["今天真离谱"], "reply": ["你可真行"]},
    ])

    rows = PersonaFewShotRetriever(path).retrieve("今天真离谱", "朋友", is_group=False, limit=4)

    assert "plain" in [row["id"] for row in rows]
    assert sum(row["id"].startswith("laugh_") for row in rows) <= 2


def test_retrieval_uses_topic_and_scenario_metadata(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "finance", "relationship": "friend", "chat_id": "f1", "context": ["今天怎么了"], "reply": ["又跌了"], "topic": "finance", "intent": "answer"},
        {"id": "food", "relationship": "friend", "chat_id": "f2", "context": ["今天怎么了"], "reply": ["去吃饭"], "topic": "food_travel", "intent": "answer"},
    ])

    rows = PersonaFewShotRetriever(path).retrieve("今天股票怎么了", "朋友", is_group=False, limit=2, relationship="friend")

    assert rows[0]["id"] == "finance"


def test_serious_query_excludes_playful_examples(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "joke", "relationship": "friend", "chat_id": "same", "context": ["面试挂了"], "reply": ["哈哈哈废物"], "response_mode": "playful"},
        {"id": "care", "relationship": "friend", "chat_id": "other", "context": ["面试挂了"], "reply": ["折腾这么久确实挺打击的"], "response_mode": "sincere"},
    ])

    rows = PersonaFewShotRetriever(path).retrieve("面试又挂了，感觉自己好菜", "朋友", is_group=False, limit=2)

    assert _query_response_mode("面试又挂了，感觉自己好菜") == "sincere"
    assert [row["id"] for row in rows] == ["care"]


def test_conditional_or_reassuring_distress_is_not_treated_as_personal():
    assert _query_response_mode("我同事问，如果我是他会不会很难受") == "neutral"
    assert _query_response_mode("我觉得不要太焦虑吧") == "neutral"
    assert _query_response_mode("我不焦虑，自己啥水平有数") == "neutral"
    assert _query_response_mode("我朋友失恋了，我该怎么安慰她") == "practical"
    assert _query_response_mode("住院手续怎么办？") == "practical"


def test_mode_hints_do_not_treat_negation_third_party_or_process_as_distress():
    for text in ("我不焦虑，自己有数", "我朋友失恋了，我该怎么安慰她", "住院手续怎么办？"):
        message = ChatMessage(text=text, sender="朋友", sender_type=SenderType.OTHER, chat_name="朋友")
        hints = ReplyGenerator._build_mode_hints([message], [message], False, 12)
        assert "对方是真的受挫/难过" not in hints


def test_retrieval_can_use_explicit_hashed_chat_id(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "same", "relationship": "friend", "chat_id": "chat_exact", "context": ["普通话题"], "reply": ["同对象"]},
        {"id": "other", "relationship": "friend", "chat_id": "other", "context": ["普通话题"], "reply": ["其他对象"]},
    ])

    rows = PersonaFewShotRetriever(path).retrieve("普通话题", "", is_group=False, limit=2, chat_id="chat_exact")

    assert rows[0]["id"] == "same"


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


def test_private_retrieval_excludes_group_examples(tmp_path):
    path = tmp_path / "examples.jsonl"
    _write_rows(path, [
        {"id": "group", "relationship": "group", "chat_id": "g", "context": ["在吗"], "reply": ["咋"]},
        {"id": "friend", "relationship": "friend", "chat_id": "f", "context": ["在吗"], "reply": ["咋啦"]},
    ])

    rows = PersonaFewShotRetriever(path).retrieve("在吗", "朋友", is_group=False, limit=8)

    assert [row["id"] for row in rows] == ["friend"]


def test_render_has_fact_isolation_and_budget(tmp_path):
    rows = [
        {"id": "one", "context": ["在吗"], "reply": ["咋啦"]},
        {"id": "two", "context": ["x" * 100], "reply": ["y" * 100]},
    ]

    content, ids = PersonaFewShotRetriever.render(rows, max_chars=260)

    assert ids == ["one"]
    assert "不是当前对话事实" in content
    assert "本人：咋啦" in content


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

    replies = generator.generate([message], [message], is_group=False)

    assert replies == ["在的"]
    assert generator.last_few_shot_ids == ["style_one"]
    assert "本人真实聊天风格示例" in generator.last_user_prompt
    assert any(item.get("type") == "persona_few_shot" for item in generator.last_generation_trace)
    assert "咋啦" not in generator.text_for_logging(generator.last_user_prompt)
    assert "style_one" in generator.text_for_logging(generator.last_user_prompt)


def test_relationship_resolution_reads_dedicated_section(tmp_path):
    (tmp_path / "同事甲.md").write_text(
        "# 人物\n## 与 Bot 的关系\n- 前同事，保持工作联系\n## 其他\n- 朋友喜欢旅游\n",
        encoding="utf-8",
    )

    assert resolve_relationship("同事甲", tmp_path) == "colleague"


def test_unapproved_report_is_not_ready(tmp_path):
    path = tmp_path / "persona_examples.jsonl"
    path.write_text("", encoding="utf-8")
    (tmp_path / "report.json").write_text('{"review_status":"pending"}', encoding="utf-8")

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
