#!/usr/bin/env python3
"""
语义检索索引构建器（兼容两种模式）

1. ChatVectorIndex: 基于 TF-IDF + 余弦相似度，检索 QA pair（问题-回复对）
2. MessageVectorIndex: 消息级检索，检索最相似的消息并返回前后多轮完整上下文
"""

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ChatVectorIndex:
    """对话向量索引 - 支持语义检索历史回复（QA pair 模式）"""

    # 权重配置
    SENDER_MATCH_BOOST = 0.15      # 同发送者加分
    CHAT_TYPE_MATCH_BOOST = 0.08   # 同聊天类型加分
    MIN_SIMILARITY = 0.05          # 最低相似度阈值

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
        import re
        text = re.sub(r'[\s\.。，,！!？?~～]+', ' ', text)
        return text

    def load(self, cache_file: Path) -> "ChatVectorIndex":
        """从指定 pkl 文件加载索引（不重新构建）。"""
        cache_file = Path(cache_file)
        print(f"📦 从缓存加载向量索引: {cache_file}")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        self.vectorizer = data['vectorizer']
        self.qa_pairs = data['qa_pairs']
        self.question_vectors = data['question_vectors']
        self.answer_vectors = data['answer_vectors']
        self.sender_index = data.get('sender_index', defaultdict(list))
        self.chat_type_index = data.get('chat_type_index', defaultdict(list))
        print(f"   加载完成: {len(self.qa_pairs)} 条对话")
        return self

    def build(self, qa_pairs: List[Dict]) -> "ChatVectorIndex":
        """构建索引"""
        cache_file = self.cache_dir / "vector_index.pkl"

        if cache_file.exists():
            print("📦 从缓存加载向量索引...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.qa_pairs = data['qa_pairs']
                self.question_vectors = data['question_vectors']
                self.answer_vectors = data['answer_vectors']
                self.sender_index = data.get('sender_index', defaultdict(list))
                self.chat_type_index = data.get('chat_type_index', defaultdict(list))
            print(f"   加载完成: {len(self.qa_pairs)} 条对话")
            return self

        print("🔢 构建语义检索索引...")
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
            token_pattern=r'(?u)\b\w+\b',
        )
        self.vectorizer.fit(all_texts)

        # 编码
        self.question_vectors = self.vectorizer.transform(questions)
        self.answer_vectors = self.vectorizer.transform(answers)

        # 保存缓存
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'qa_pairs': self.qa_pairs,
                'question_vectors': self.question_vectors,
                'answer_vectors': self.answer_vectors,
                'sender_index': dict(self.sender_index),
                'chat_type_index': dict(self.chat_type_index),
            }, f)

        print(f"   索引完成: {len(self.qa_pairs)} 条对话, {len(self.vectorizer.vocabulary_)} 维特征")
        print(f"   发送者索引: {len(self.sender_index)} 个唯一发送者")
        print(f"   聊天类型索引: {dict((k, len(v)) for k, v in self.chat_type_index.items())}")
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


class MessageVectorIndex:
    """
    消息级向量索引加载器
    每条消息单独编码，检索后拉出前后多轮完整对话上下文
    """

    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)
        self.vectorizer = None
        self.vectors = None
        self.messages: List[Dict] = []
        self.msg_by_id: Dict[int, Dict] = {}
        self.sender_index: Dict[str, List[int]] = {}
        self.chat_type_index: Dict[str, List[int]] = {}
        self._load()

    def _load(self):
        if not self.cache_path.exists():
            print(f"[MessageVectorIndex] 警告: 缓存不存在 {self.cache_path}")
            return

        print("[MessageVectorIndex] 加载消息级向量索引...")
        with open(self.cache_path, 'rb') as f:
            data = pickle.load(f)

        self.vectorizer = data['vectorizer']
        self.vectors = data['vectors']
        self.messages = data['messages']
        self.sender_index = data.get('sender_index', {})
        self.chat_type_index = data.get('chat_type_index', {})
        self.msg_by_id = {m['id']: m for m in self.messages}

        print(f"[MessageVectorIndex] 加载完成: {len(self.messages)} 条消息, {self.vectors.shape[1]} 维")

    def search(
        self,
        query: str,
        sender_name: str = "",
        chat_type: str = "",
        top_k: int = 5,
        context_radius: int = 5
    ) -> List[Dict]:
        """
        消息级检索：检索最相似的消息，返回包含上下文的对话片段

        Returns:
            [{"score": float, "hit_message": msg, "context_messages": [msg, ...]}, ...]
        """
        if self.vectorizer is None or self.vectors is None:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.vectors)[0]

        # 加权：同发送者 +0.05，同聊天类型 +0.03
        if sender_name and sender_name in self.sender_index:
            for mid in self.sender_index[sender_name]:
                if mid < len(scores):
                    scores[mid] += 0.05

        if chat_type and chat_type in self.chat_type_index:
            for mid in self.chat_type_index[chat_type]:
                if mid < len(scores):
                    scores[mid] += 0.03

        top_indices = np.argsort(scores)[::-1][:top_k * 2]

        results = []
        seen_contexts = set()

        for idx in top_indices:
            if scores[idx] < 0.01:
                continue

            msg = self.messages[idx]
            context_ids = msg.get('context_ids', [msg['id']])
            context_key = tuple(context_ids)

            # 去重：避免返回相同的上下文窗口
            if context_key in seen_contexts:
                continue
            seen_contexts.add(context_key)

            context_msgs = [self.msg_by_id.get(cid, {}) for cid in context_ids]
            context_msgs = [m for m in context_msgs if m]

            results.append({
                'score': float(scores[idx]),
                'hit_message': msg,
                'context_messages': context_msgs,
            })

            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def format_context(context_messages: List[Dict]) -> str:
        """把上下文消息格式化为对话文本"""
        lines = []
        for m in context_messages:
            sender = m.get('sender', '对方')
            role = "你" if m.get('is_self') else sender
            text = m.get('text', '')
            lines.append(f'{role}: "{text}"')
        return "\n".join(lines)
