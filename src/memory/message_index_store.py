"""内存增量消息索引。

为 HistorySearchIndex 提供运行时单条 add/update/remove 能力。
增量消息先保存在内存中，检索时与持久化主索引合并；flush 时写回 pkl。
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MessageIndexStore:
    """内存中的消息向量索引。

    维护与持久化主索引相同的数据结构：
      - messages: 消息元数据列表
      - embeddings: (N, dim) float32 数组，已 L2 归一化
      - msg_by_id / id_to_idx / sender_index / chat_type_index

    encoder 通过参数传入，避免循环 import。
    """

    dim: int = 512
    messages: List[Dict[str, Any]] = field(default_factory=list)
    embeddings: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 512), dtype=np.float32)
    )
    msg_by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    id_to_idx: Dict[str, int] = field(default_factory=dict)
    sender_index: Dict[str, List[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    chat_type_index: Dict[str, List[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    dirty: bool = False

    # ── 写操作 ──

    def add_or_update(self, msg: Dict[str, Any], encoder: Any) -> None:
        """添加或更新一条消息。

        Args:
            msg: 必须包含 id、text；建议包含 sender、chat_type、chat_name、
                 is_self、timestamp、file、index_in_file
            encoder: 拥有 .encode(texts: List[str]) -> np.ndarray 的对象
        """
        msg_id = msg.get("id")
        if not msg_id:
            raise ValueError("消息必须包含 id")

        # 如果已存在，先删除旧记录
        if msg_id in self.msg_by_id:
            self.remove(msg_id)

        text = (msg.get("text") or "").strip()
        if not text or len(text) < 2:
            raise ValueError("消息 text 为空或太短")

        emb = encoder.encode([text])
        if emb.shape[0] != 1:
            raise ValueError(f"编码器返回形状异常: {emb.shape}")

        self.embeddings = np.vstack([self.embeddings, emb.astype(np.float32)])
        self.messages.append(msg)
        self._rebuild_maps()
        self.dirty = True

    def remove(self, msg_id: str) -> bool:
        """删除一条消息。返回是否真实删除了数据。"""
        if msg_id not in self.msg_by_id:
            return False

        idx = self.id_to_idx[msg_id]
        self.messages.pop(idx)
        self.embeddings = np.delete(self.embeddings, idx, axis=0)
        self._rebuild_maps()
        self.dirty = True
        return True

    def clear(self) -> None:
        """清空所有增量数据（通常在 flush 到 pkl 后调用）。"""
        self.messages = []
        self.embeddings = np.zeros((0, self.dim), dtype=np.float32)
        self.msg_by_id = {}
        self.id_to_idx = {}
        self.sender_index = defaultdict(list)
        self.chat_type_index = defaultdict(list)
        self.dirty = False

    def _rebuild_maps(self) -> None:
        """重建 id 映射和辅助索引。"""
        self.msg_by_id = {m["id"]: m for m in self.messages}
        self.id_to_idx = {m["id"]: i for i, m in enumerate(self.messages)}
        self.sender_index = defaultdict(list)
        self.chat_type_index = defaultdict(list)
        for m in self.messages:
            self.sender_index[m.get("sender", "未知")].append(m["id"])
            self.chat_type_index[m.get("chat_type", "single")].append(m["id"])

    # ── 读操作 ──

    def is_empty(self) -> bool:
        return len(self.messages) == 0

    def __len__(self) -> int:
        return len(self.messages)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        sender_name: str = "",
        chat_type: str = "",
        min_score: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """在增量区做 dense 检索，返回与 HistorySearchIndex._dense_search 兼容的结构。"""
        if self.is_empty():
            return []

        # query_embedding 已归一化，点积即 cosine
        scores = self.embeddings @ query_embedding

        # sender/chat_type 轻微加权，与主索引一致
        if sender_name and sender_name in self.sender_index:
            for mid in self.sender_index[sender_name]:
                i = self.id_to_idx.get(mid)
                if i is not None:
                    scores[i] += 0.05
        if chat_type and chat_type in self.chat_type_index:
            for mid in self.chat_type_index[chat_type]:
                i = self.id_to_idx.get(mid)
                if i is not None:
                    scores[i] += 0.03

        top_indices = np.argsort(scores)[::-1]
        results: List[Dict[str, Any]] = []
        seen_contexts: set = set()

        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                break

            msg = self.messages[idx]
            context_ids = msg.get("context_ids") or [msg["id"]]
            context_key = tuple(context_ids)
            if context_key in seen_contexts:
                continue
            seen_contexts.add(context_key)

            context_msgs = [self.msg_by_id.get(cid) for cid in context_ids]
            context_msgs = [m for m in context_msgs if m]

            results.append(
                {
                    "score": score,
                    "hit_message": msg,
                    "context_messages": context_msgs,
                    "context_key": context_key,
                    "source": "incremental",
                }
            )
            if len(results) >= top_k:
                break

        return results

    def recall_candidate_ids(
        self,
        query_embedding: np.ndarray,
        top_n: int = 30,
        sender_name: str = "",
        chat_type: str = "",
        min_score: float = 0.01,
    ) -> List[str]:
        """返回召回阶段候选 id 列表（用于 benchmark/debug）。"""
        results = self.search(
            query_embedding,
            top_k=top_n,
            sender_name=sender_name,
            chat_type=chat_type,
            min_score=min_score,
        )
        ids: List[str] = []
        seen: set = set()
        for r in results:
            for m in [r["hit_message"]] + r.get("context_messages", []):
                if m and m.get("id") and m["id"] not in seen:
                    seen.add(m["id"])
                    ids.append(m["id"])
        return ids

    def get_message(self, msg_id: str) -> Optional[Dict[str, Any]]:
        return self.msg_by_id.get(msg_id)
