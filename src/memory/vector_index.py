#!/usr/bin/env python3
"""
语义检索索引构建器

ChatVectorIndex: 基于 TF-IDF + 余弦相似度，检索 QA pair（问题-回复对）

历史消息级原文检索由 src/memory/history_search.py（BGE dense + keyword 两路融合）
承担，旧 MessageVectorIndex（TF-IDF，humor RAG）已移除。

缓存使用 JSON 格式保存，避免 pickle 反序列化带来的安全风险（B301/B403）。
TfidfVectorizer 通过保存 vocabulary_/idf_/初始化参数并在加载时重建，稀疏矩阵保存
CSR 的 data/indices/indptr/shape。
"""

import json
import logging
import string
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_logger = logging.getLogger("src.memory.vector_index")


# ---------------------------------------------------------------------------
# JSON 序列化辅助函数
# ---------------------------------------------------------------------------

def _vectorizer_to_json(vectorizer: TfidfVectorizer) -> Dict[str, Any]:
    """将已拟合的 TfidfVectorizer 转为可 JSON 序列化的字典。"""
    params = dict(vectorizer.get_params())
    # dtype 在 sklearn 中通常是 type 对象（如 numpy.float64），需要转成字符串
    dtype = params.get("dtype")
    if dtype is not None and not isinstance(dtype, str):
        params["dtype"] = np.dtype(dtype).name

    return {
        "params": params,
        "vocabulary_": {k: int(v) for k, v in vectorizer.vocabulary_.items()},
        "idf_": vectorizer.idf_.tolist() if hasattr(vectorizer, "idf_") else None,
        "stop_words_": (
            sorted(vectorizer.stop_words_)
            if hasattr(vectorizer, "stop_words_") and vectorizer.stop_words_
            else None
        ),
    }


def _vectorizer_from_json(data: Dict[str, Any]) -> TfidfVectorizer:
    """从 JSON 字典重建 TfidfVectorizer，保持与原始 vectorizer 相同的编码行为。"""
    params = dict(data["params"])
    # JSON 不保留元组，sklearn 期望 ngram_range 为元组
    for key in ("ngram_range",):
        if key in params and isinstance(params[key], list):
            params[key] = tuple(params[key])

    # dtype 序列化时为字符串，还原为 numpy type 以保持完全一致的参数
    dtype = params.get("dtype")
    if isinstance(dtype, str):
        params["dtype"] = np.dtype(dtype).type

    vectorizer = TfidfVectorizer(**params)

    vocabulary = data.get("vocabulary_")
    if vocabulary:
        vectorizer.vocabulary_ = {k: int(v) for k, v in vocabulary.items()}

    idf_ = data.get("idf_")
    if idf_ is not None:
        vectorizer.idf_ = np.array(idf_, dtype=np.float64)

    stop_words = data.get("stop_words_")
    if stop_words is not None:
        vectorizer.stop_words_ = set(stop_words)

    return vectorizer


def _csr_to_json(matrix: sp.csr_matrix) -> Dict[str, Any]:
    """将 CSR 稀疏矩阵转为可 JSON 序列化的字典。"""
    return {
        "data": matrix.data.tolist(),
        "indices": matrix.indices.tolist(),
        "indptr": matrix.indptr.tolist(),
        "shape": list(matrix.shape),
    }


def _csr_from_json(data: Dict[str, Any]) -> sp.csr_matrix:
    """从 JSON 字典重建 CSR 稀疏矩阵。"""
    return sp.csr_matrix(
        (data["data"], data["indices"], data["indptr"]),
        shape=tuple(data["shape"]),
    )


class ChatVectorIndex:
    """对话向量索引 - 支持语义检索历史回复（QA pair 模式）"""

    # 权重配置
    SENDER_MATCH_BOOST = 0.15      # 同发送者加分
    CHAT_TYPE_MATCH_BOOST = 0.08   # 同聊天类型加分
    MIN_SIMILARITY = 0.05          # 最低相似度阈值

    # 缓存文件名
    CACHE_FILE_NAME = "vector_index.json"

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.qa_pairs: List[Dict] = []
        self.question_vectors = None
        self.answer_vectors = None

        # 辅助索引：sender -> 向量索引列表
        self.sender_index: Dict[str, List[int]] = defaultdict(list)
        # 辅助索引：chat_type -> 向量索引列表
        self.chat_type_index: Dict[str, List[int]] = defaultdict(list)

    def _preprocess_text(self, text: str) -> str:
        """文本预处理用于向量化"""
        if not text:
            return ""
        text = text.lower().strip()
        # 把空白和常见标点统一成空格，再规范化（Rule 3.4：用 str.translate 替代正则）
        trans = str.maketrans({
            ch: " " for ch in string.whitespace + " .。，,！!？?~～"
        })
        text = text.translate(trans)
        return " ".join(text.split())

    def _load_cache_data(self, cache_file: Path) -> Dict[str, Any]:
        """从 JSON 缓存文件读取原始数据字典。"""
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _apply_cache_data(self, data: Dict[str, Any]) -> None:
        """将原始数据字典恢复到当前实例状态。"""
        self.vectorizer = _vectorizer_from_json(data['vectorizer'])
        self.qa_pairs = data['qa_pairs']
        self.question_vectors = _csr_from_json(data['question_vectors'])
        self.answer_vectors = _csr_from_json(data['answer_vectors'])
        self.sender_index = defaultdict(list, data.get('sender_index', {}))
        self.chat_type_index = defaultdict(list, data.get('chat_type_index', {}))

    def load(self, cache_file: Path) -> "ChatVectorIndex":
        """从指定 JSON 缓存文件加载索引（不重新构建）。"""
        cache_file = Path(cache_file)
        _logger.info(f"从缓存加载向量索引: {cache_file}")
        data = self._load_cache_data(cache_file)
        self._apply_cache_data(data)
        _logger.info(f"  加载完成: {len(self.qa_pairs)} 条对话")
        return self

    def build(self, qa_pairs: List[Dict]) -> "ChatVectorIndex":
        """构建索引"""
        cache_file = self.cache_dir / self.CACHE_FILE_NAME

        if cache_file.exists():
            _logger.info("从缓存加载向量索引...")
            data = self._load_cache_data(cache_file)
            self._apply_cache_data(data)
            _logger.info(f"  加载完成: {len(self.qa_pairs)} 条对话")
            return self

        _logger.info("构建语义检索索引...")
        self.qa_pairs = qa_pairs

        # 构建辅助索引
        for i, pair in enumerate(qa_pairs):
            sender = pair.get('q_sender', '') or pair.get('sender', '')
            if sender:
                self.sender_index[sender].append(i)
            chat_type = pair.get('chat_type', 'single')
            self.chat_type_index[chat_type].append(i)

        # 准备语料
        questions = [self._preprocess_text(p['question']) for p in qa_pairs]
        answers = [self._preprocess_text(p['answer']) for p in qa_pairs]

        # 构建 TF-IDF 向量器
        all_texts = questions + answers
        self.vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            token_pattern=r'(?u)\b\w+\b',  # nosec B106
        )
        self.vectorizer.fit(all_texts)

        # 编码
        self.question_vectors = self.vectorizer.transform(questions)
        self.answer_vectors = self.vectorizer.transform(answers)

        # 保存缓存
        cache_payload = {
            'vectorizer': _vectorizer_to_json(self.vectorizer),
            'qa_pairs': self.qa_pairs,
            'question_vectors': _csr_to_json(self.question_vectors),
            'answer_vectors': _csr_to_json(self.answer_vectors),
            'sender_index': dict(self.sender_index),
            'chat_type_index': dict(self.chat_type_index),
        }
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_payload, f, ensure_ascii=False, separators=(',', ':'))

        _logger.info(f"  索引完成: {len(self.qa_pairs)} 条对话, {len(self.vectorizer.vocabulary_)} 维特征")
        _logger.info(f"  发送者索引: {len(self.sender_index)} 个唯一发送者")
        _logger.info(f"  聊天类型索引: {dict((k, len(v)) for k, v in self.chat_type_index.items())}")
        return self

    def search(
        self,
        query: str,
        top_k: int = 5,
        search_in: str = "question"
    ) -> List[Tuple[float, Dict]]:
        """
        语义检索相似对话

        Args:
            query: 查询文本
            top_k: 返回条数
            search_in: "question" | "answer" | "both"

        Returns:
            [(相似度分数, QAPair), ...]
        """
        if self.vectorizer is None:
            return []

        query_vec = self.vectorizer.transform([self._preprocess_text(query)])

        if search_in == "question":
            scores = cosine_similarity(query_vec, self.question_vectors)[0]
        elif search_in == "answer":
            scores = cosine_similarity(query_vec, self.answer_vectors)[0]
        else:
            q_scores = cosine_similarity(query_vec, self.question_vectors)[0]
            a_scores = cosine_similarity(query_vec, self.answer_vectors)[0]
            scores = q_scores * 0.7 + a_scores * 0.3

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > self.MIN_SIMILARITY:
                results.append((float(scores[idx]), self.qa_pairs[idx]))

        return results

    def search_by_sender(
        self,
        query: str,
        sender_name: str,
        top_k: int = 3,
        fallback_to_all: bool = True
    ) -> List[Tuple[float, Dict]]:
        """按发送者检索：优先返回同发送者的历史对话"""
        if self.vectorizer is None:
            return []

        all_results = self.search(query, top_k=top_k * 3, search_in="question")

        same_sender = []
        other_sender = []
        for score, pair in all_results:
            pair_sender = pair.get('q_sender', '') or pair.get('sender', '')
            if pair_sender == sender_name:
                same_sender.append((score + self.SENDER_MATCH_BOOST, pair))
            else:
                other_sender.append((score, pair))

        results = sorted(same_sender, key=lambda x: x[0], reverse=True)

        if fallback_to_all and len(results) < top_k:
            results.extend(sorted(other_sender, key=lambda x: x[0], reverse=True))

        return results[:top_k]

    def search_with_context(
        self,
        query: str,
        sender_name: Optional[str] = None,
        chat_type: Optional[str] = None,
        top_k: int = 5
    ) -> List[Tuple[float, Dict]]:
        """带语境的语义检索：综合考虑消息内容、发送者、聊天类型"""
        if self.vectorizer is None:
            return []

        query_vec = self.vectorizer.transform([self._preprocess_text(query)])
        q_scores = cosine_similarity(query_vec, self.question_vectors)[0]

        weighted_scores = np.array(q_scores, dtype=np.float64)

        if sender_name:
            sender_indices = set(self.sender_index.get(sender_name, []))
            for idx in sender_indices:
                if idx < len(weighted_scores):
                    weighted_scores[idx] += self.SENDER_MATCH_BOOST

        if chat_type:
            chat_indices = set(self.chat_type_index.get(chat_type, []))
            for idx in chat_indices:
                if idx < len(weighted_scores):
                    weighted_scores[idx] += self.CHAT_TYPE_MATCH_BOOST

        top_indices = np.argsort(weighted_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if weighted_scores[idx] > self.MIN_SIMILARITY:
                results.append((float(weighted_scores[idx]), self.qa_pairs[idx]))

        return results
