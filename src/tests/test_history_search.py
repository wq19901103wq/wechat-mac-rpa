#!/usr/bin/env python3
"""search_history 工具后端（history_search）单元测试。

不依赖真实的 1.7GB 索引或 torch/onnxruntime：用合成数据 + monkeypatch 覆盖
检索逻辑、去重、阈值、格式化与优雅降级。
"""

import numpy as np

from src.memory import history_search
from src.memory.history_search import HistorySearchIndex, search_history

# ── 优雅降级 ──


class TestAvailability:
    def test_unavailable_when_index_missing(self, monkeypatch, tmp_path):
        """索引文件不存在时 is_available 为 False，且不 import 重依赖。"""
        monkeypatch.setenv("WECHAT_HISTORY_INDEX_PATH", str(tmp_path / "nope.pkl"))
        assert history_search.is_available() is False

    def test_search_history_returns_message_when_no_index(self, monkeypatch, tmp_path):
        """索引未就绪时，工具入口返回明确不可用提示而非抛异常。"""
        monkeypatch.setenv("WECHAT_HISTORY_INDEX_PATH", str(tmp_path / "nope.pkl"))
        history_search.reset_singleton()
        out = search_history("随便查")
        assert "不可用" in out


# ── 检索逻辑（合成索引，bypass __init__）──


def _norm(rows):
    a = np.array(rows, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def _make_index():
    """用 __new__ 绕过 pickle/编码器加载，构造合成索引。

    查询向量 = [1,0,0,0]，所以 cosine = 向量第 0 维（归一化后）：
      f_0 → 1.0, f_1 → 0.8, g_0 → 0.6
    f_0 与 f_1 共享 context_ids（同一对话窗口）。
    """
    idx = HistorySearchIndex.__new__(HistorySearchIndex)
    messages = [
        {
            "id": "f_0", "text": "明天去上海吗", "sender": "王海",
            "is_self": False, "chat_name": "王海", "chat_type": "single",
            "context_ids": ["f_0", "f_1"],
        },
        {
            "id": "f_1", "text": "嗯去出差", "sender": "王海",
            "is_self": True, "chat_name": "王海", "chat_type": "single",
            "context_ids": ["f_0", "f_1"],
        },
        {
            "id": "g_0", "text": "那家日料不错", "sender": "李四",
            "is_self": False, "chat_name": "饭团群", "chat_type": "group",
            "context_ids": ["g_0"],
        },
    ]
    idx.messages = messages
    idx.msg_by_id = {m["id"]: m for m in messages}
    idx.id_to_idx = {m["id"]: i for i, m in enumerate(messages)}
    idx.sender_index = {"王海": ["f_0", "f_1"], "李四": ["g_0"]}
    idx.chat_type_index = {"single": ["f_0", "f_1"], "group": ["g_0"]}
    idx.embeddings = _norm([
        [1.0, 0.0, 0.0, 0.0],   # f_0: cos 1.0
        [0.8, 0.6, 0.0, 0.0],   # f_1: cos 0.8
        [0.6, 0.0, 0.8, 0.0],   # g_0: cos 0.6
    ])

    class _FakeEncoder:
        def encode(self, texts):
            v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return (v / np.linalg.norm(v)).reshape(1, -1)

    idx.encoder = _FakeEncoder()
    return idx


def _make_near_tied_index():
    """f_0(王海) 与 g_0(李四) 近似平手：cos 0.98 vs 0.97，+0.05 加权可翻转。"""
    idx = HistorySearchIndex.__new__(HistorySearchIndex)
    messages = [
        {
            "id": "f_0", "text": "A", "sender": "王海", "is_self": False,
            "chat_name": "王海", "chat_type": "single", "context_ids": ["f_0"],
        },
        {
            "id": "g_0", "text": "B", "sender": "李四", "is_self": False,
            "chat_name": "饭团群", "chat_type": "group", "context_ids": ["g_0"],
        },
    ]
    idx.messages = messages
    idx.msg_by_id = {m["id"]: m for m in messages}
    idx.id_to_idx = {m["id"]: i for i, m in enumerate(messages)}
    idx.sender_index = {"王海": ["f_0"], "李四": ["g_0"]}
    idx.chat_type_index = {"single": ["f_0"], "group": ["g_0"]}
    idx.embeddings = _norm([
        [0.98, 0.199, 0.0, 0.0],  # f_0: cos 0.98
        [0.97, 0.0, 0.243, 0.0],  # g_0: cos 0.97
    ])

    class _FakeEncoder:
        def encode(self, texts):
            v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return (v / np.linalg.norm(v)).reshape(1, -1)

    idx.encoder = _FakeEncoder()
    return idx


class TestSearchLogic:
    def test_dedup_by_context_ids(self):
        """m0 与 m1 共享 context_ids，只应返回一次（取分数高的那条）。"""
        idx = _make_index()
        results = idx.search("去上海", top_k=5)
        assert len(results) == 2  # m0/m1 合并为 1 + m2 = 2
        # 命中 m0（分数最高），上下文含 m0+m1 两条
        hit_ids = {r["hit_message"]["id"] for r in results}
        assert "g_0" in hit_ids
        first = max(results, key=lambda r: r["score"])
        assert first["hit_message"]["id"] == "f_0"
        ctx_ids = [m["id"] for m in first["context_messages"]]
        assert ctx_ids == ["f_0", "f_1"]

    def test_top_k_limit(self):
        idx = _make_index()
        results = idx.search("去上海", top_k=1)
        assert len(results) == 1

    def test_min_score_filter(self):
        """分数低于阈值的尾部结果应被丢弃。"""
        idx = _make_index()
        # f_0=1.0, g_0=0.6, f_1=0.8(因 ctx 与 f_0 重复被跳过)
        # 阈值 0.7 → f_0 命中，g_0(0.6) 被丢弃
        results = idx.search("去上海", top_k=5, min_score=0.7)
        assert all(r["score"] >= 0.7 for r in results)
        assert len(results) == 1

    def test_empty_query(self):
        idx = _make_index()
        assert idx.search("", top_k=5) == []
        assert idx.search("   ", top_k=5) == []

    def test_sender_boost(self):
        """同发送者加权应让该发送者的消息排名上升。"""
        idx = _make_near_tied_index()
        plain = idx.search("x", top_k=5)
        # 无加权：f_0(0.98) > g_0(0.97)
        assert plain[0]["hit_message"]["id"] == "f_0"
        # 给李四加权 +0.05，g_0 → 1.02 > f_0 0.98，翻到第一
        boosted = idx.search("x", top_k=5, sender_name="李四")
        assert boosted[0]["hit_message"]["id"] == "g_0"


# ── 格式化 ──


class TestFormatResults:
    def test_empty(self):
        out = HistorySearchIndex.format_results([], "某话题")
        assert "未找到" in out
        assert "某话题" in out

    def test_format_contains_role_and_text(self):
        results = [
            {
                "score": 0.823,
                "hit_message": {
                    "chat_name": "王海", "chat_type": "single",
                    "id": "f_0", "is_self": False, "sender": "王海",
                },
                "context_messages": [
                    {"is_self": False, "sender": "王海", "text": "明天去上海吗"},
                    {"is_self": True, "sender": "我", "text": "嗯去出差"},
                ],
            }
        ]
        out = HistorySearchIndex.format_results(results, "去上海", max_chars=4000)
        assert "历史聊天原文检索" in out
        assert "片段 1" in out
        assert "0.823" in out
        assert "王海: 明天去上海吗" in out
        assert "你: 嗯去出差" in out

    def test_format_truncates(self):
        """超长结果应截断且带提示。"""
        ctx = [
            {"is_self": False, "sender": "x", "text": "内容" * 500}
        ]
        results = [
            {
                "score": 0.9,
                "hit_message": {"chat_name": "c", "chat_type": "group", "id": "x", "is_self": False, "sender": "x"},
                "context_messages": ctx,
            }
        ]
        out = HistorySearchIndex.format_results(results, "q", max_chars=200)
        assert len(out) <= 200
        assert "截断" in out


# ── 工具入口 wiring ──


class TestSearchHistoryEntry:
    def test_empty_query_rejected(self):
        assert "空" in search_history("")

    def test_uses_singleton(self, monkeypatch):
        """search_history 应走 get_history_index 单例并格式化其结果。"""
        idx = _make_index()

        class _FakeIdx:
            def search(self, query, top_k=5):
                return idx.search(query, top_k=top_k)
            format_results = staticmethod(HistorySearchIndex.format_results)

        monkeypatch.setattr(history_search, "get_history_index", lambda: _FakeIdx())
        out = search_history("去上海", top_k=3)
        assert "历史聊天原文检索" in out
        assert "去上海" in out

    def test_search_exception_swallowed(self, monkeypatch):
        """检索内部异常应被吞掉，返回出错提示而非抛出。"""
        class _Boom:
            def search(self, *a, **k):
                raise RuntimeError("boom")

        monkeypatch.setattr(history_search, "get_history_index", lambda: _Boom())
        out = search_history("x")
        assert "出错" in out
