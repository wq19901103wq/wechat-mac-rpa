"""历史聊天原文语义检索（search_history 工具的后端）。

复用 digital-twin 已建好的消息级 BGE dense 索引（77 万条历史微信消息原文），
对新查询做语义检索，返回**原文对话片段**（命中消息 + 前后多轮上下文）。

与 `MemoryEngine.search_keyword`（wiki 人物记忆摘要）正交：
- search_keyword  → 检索编译后的 wiki 摘要（"这个人是谁、和我的关系"）
- search_history  → 检索历史聊天原文（"当时到底说了什么"）

设计原则（低风险接入）：
- 工具**仅在索引与编码器依赖就绪时**才注册（`is_available()`），否则 generator
  完全不暴露 search_history，bot 行为零变化。
- 索引（1.7GB pickle）与 BGE 模型**懒加载**，首次调用时载入并缓存为模块单例。
- 编码器优先用 ONNX Runtime（轻量，无需 torch），不可用则回退 torch+transformers。
  两者都与建库时的 bge-small-zh-v1.5 嵌入空间一致（导出时已校验 cosine > 0.999）。

索引/模型路径与依赖均可通过环境变量覆盖，便于跨机器迁移：
- WECHAT_HISTORY_INDEX_PATH  dense 消息索引 pickle 路径
- WECHAT_BGE_MODEL_PATH      BGE 模型目录（含 tokenizer + onnx/pytorch 权重）
"""

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_logger = logging.getLogger("src.memory.history_search")

# ── 默认路径：指向 digital-twin 已建好的资产，可用环境变量覆盖 ──
_DEFAULT_INDEX_PATH = Path(
    "/Users/yihanwang/wechat-digital-twin/outputs/cache/vector_index_dense_messages.pkl"
)
_DEFAULT_MODEL_PATH = Path(
    "/Users/yihanwang/wechat-digital-twin/models/bge-small-zh-v1.5"
)


def _index_path() -> Path:
    return Path(os.environ.get("WECHAT_HISTORY_INDEX_PATH", _DEFAULT_INDEX_PATH))


def _model_path() -> Path:
    return Path(os.environ.get("WECHAT_BGE_MODEL_PATH", _DEFAULT_MODEL_PATH))


def _try_import_encoder_deps() -> Optional[str]:
    """探测可用的编码器后端，返回 'onnx' / 'torch' / None。

    优先级（轻量优先）：
      onnx : tokenizers + onnxruntime（无需 transformers/torch，bot 运行环境即可满足）
      onnx : transformers + onnxruntime（兜底）
      torch: transformers + torch（最重，最后兜底）

    不做任何重活，只检查 import 是否可用，供 generator 决定是否注册工具。
    """
    # 首选：tokenizers + onnxruntime（最轻量，base conda 即有）
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
        return "onnx"
    except ImportError:
        pass
    # 兜底 1：transformers + onnxruntime
    try:
        import onnxruntime  # noqa: F401
        import transformers  # noqa: F401
        return "onnx"
    except ImportError:
        pass
    # 兜底 2：transformers + torch
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return "torch"
    except ImportError:
        pass
    return None


def is_available() -> bool:
    """search_history 是否可用：索引文件存在 + 编码器依赖可 import。

    Generator 用它决定是否注册 search_history 工具。轻量探测，不加载索引/模型。
    """
    if not _index_path().exists():
        return False
    return _try_import_encoder_deps() is not None


# ── 编码器：ONNX 优先，torch 兜底 ──


class _BGEEncoder:
    """BGE-small-zh-v1.5 编码器。输出 L2 归一化的 (N, 512) float32 向量。

    与建库脚本一致：取 CLS token（last_hidden_state[:,0]），L2 归一化，不加查询
    前缀（corpus 编码时也未加）。

    tokenizer 优先用轻量的 `tokenizers` 库直读 tokenizer.json（无需 transformers），
    不可用则回退 transformers AutoTokenizer。
    """

    _tok: Any = None  # tokenizers.Tokenizer 或 transformers AutoTokenizer
    _session: Any = None  # onnxruntime.InferenceSession（onnx 后端）
    _model: Any = None    # transformers AutoModel（torch 后端）
    _torch: Any = None
    _device: Any = None

    def __init__(self, model_path: Path, backend: str):
        self.model_path = model_path
        self.backend = backend
        self._tokenizer_kind = self._init_tokenizer(model_path)
        if backend == "onnx":
            import onnxruntime as ort

            onnx_path = model_path / "model_optimized.onnx"
            if not onnx_path.exists():
                onnx_path = model_path / "model.onnx"
            if not onnx_path.exists():
                raise FileNotFoundError(f"ONNX 模型不存在: {onnx_path}")
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 0
            opts.inter_op_num_threads = 0
            self._session = ort.InferenceSession(
                str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self._model = None
            self._device = None
        else:  # torch
            import torch
            from transformers import AutoModel

            self._torch = torch
            # 模型从本地路径加载，非 Hub 远程下载；nosec B615
            self._model = AutoModel.from_pretrained(str(model_path))  # nosec B615
            self._model.eval()
            self._device = torch.device("cpu")
            self._model.to(self._device)
            self._session = None

    @staticmethod
    def _init_tokenizer(model_path: Path) -> str:
        """加载 tokenizer，返回 'tokenizers' / 'transformers'。"""
        tok_json = model_path / "tokenizer.json"
        if tok_json.exists():
            try:
                from tokenizers import Tokenizer

                _BGEEncoder._tok = Tokenizer.from_file(str(tok_json))
                _BGEEncoder._tok.enable_padding(pad_id=0, pad_token="[PAD]")  # nosec B106
                _BGEEncoder._tok.enable_truncation(max_length=200)
                return "tokenizers"
            except ImportError:
                pass
        from transformers import AutoTokenizer

        # 模型从本地路径加载，非 Hub 远程下载；nosec B615
        _BGEEncoder._tok = AutoTokenizer.from_pretrained(str(model_path))  # nosec B615
        return "transformers"

    def _tokenize(self, texts: List[str]):
        """返回 (input_ids, attention_mask)，均为 np.int64 数组。"""
        tok = self._tok
        assert tok is not None  # _init_tokenizer 已保证赋值
        if self._tokenizer_kind == "tokenizers":
            encs = tok.encode_batch(texts)
            input_ids = np.array([e.ids for e in encs], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            return input_ids, attention_mask
        # transformers
        inputs = tok(
            texts, padding=True, truncation=True, max_length=200, return_tensors="np"
        )
        return (
            inputs["input_ids"].astype(np.int64),
            inputs["attention_mask"].astype(np.int64),
        )

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 512), dtype=np.float32)

        if self.backend == "onnx":
            input_ids, attention_mask = self._tokenize(texts)
            outputs = self._session.run(
                None, {"input_ids": input_ids, "attention_mask": attention_mask}
            )
            emb = outputs[0][:, 0, :]  # CLS token (last_hidden_state[:,0])
        else:
            torch = self._torch
            input_ids, attention_mask = self._tokenize(texts)
            encoded = {
                "input_ids": torch.tensor(input_ids).to(self._device),
                "attention_mask": torch.tensor(attention_mask).to(self._device),
            }
            with torch.no_grad():
                emb = self._model(**encoded).last_hidden_state[:, 0]
            emb = emb.cpu().numpy()

        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (emb / norms).astype(np.float32)


# ── 索引：懒加载 digital-twin 的 dense 消息索引 ──


class HistorySearchIndex:
    """消息级 BGE dense 索引的检索器。

    一次加载、常驻内存（embeddings ~1.6GB + messages 列表）。多线程安全只读检索。
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        backend: Optional[str] = None,
    ):
        import pickle

        self.index_path = index_path or _index_path()
        self.model_path = model_path or _model_path()
        if backend is None:
            backend = _try_import_encoder_deps()
        if backend is None:
            raise RuntimeError(
                "无可用的 BGE 编码器后端（需安装 onnxruntime 或 torch + transformers）"
            )

        _logger.info("[HistorySearch] 加载 dense 消息索引: %s", self.index_path)
        with open(self.index_path, "rb") as f:
            # 索引由本地 digital-twin 自产，非不可信来源；nosec B301
            data = pickle.load(f)  # nosec B301

        self.embeddings: np.ndarray = data["embeddings"].astype(np.float32)
        # 行归一化兜底：建库时已归一化，但防御性再保一次，确保 dot == cosine
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms

        self.messages: List[Dict[str, Any]] = data["messages"]
        self.msg_by_id: Dict[str, Dict[str, Any]] = data.get(
            "msg_by_id", {m["id"]: m for m in self.messages}
        )
        self.id_to_idx: Dict[str, int] = data.get(
            "id_to_idx", {m["id"]: i for i, m in enumerate(self.messages)}
        )
        # 真实索引里 sender_index / chat_type_index 的值是消息 id 字符串列表，
        # 需经 id_to_idx 映射回行下标，不能直接当整数下标用。
        self.sender_index: Dict[str, List[str]] = data.get("sender_index", {})
        self.chat_type_index: Dict[str, List[str]] = data.get("chat_type_index", {})

        _logger.info(
            "[HistorySearch] 索引就绪: %d 条消息, %d 维 (backend=%s)",
            len(self.messages),
            self.embeddings.shape[1],
            backend,
        )

        _logger.info("[HistorySearch] 加载 BGE 编码器: %s", self.model_path)
        self.encoder = _BGEEncoder(self.model_path, backend)

    # ── 召回参数 ──
    # 融合权重：dense 主导（语义），keyword 补精确词；两路共识额外提升
    _FUSE_ALPHA = 0.6  # fusion = alpha*dense + (1-alpha)*keyword
    _BOTH_BOOST = 1.15  # 两路都命中的 context 乘以该系数
    _RECALL_N = 20  # 每路召回数（大于 top_k，给 rerank 留余量）

    def search(
        self,
        query: str,
        top_k: int = 5,
        sender_name: str = "",
        chat_type: str = "",
        min_score: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """两路召回 + 分数融合 rerank。

        dense 向量路（语义相似）+ keyword 关键字路（精确词命中）各召回 top_n，
        按 context_ids 合并、分数归一化加权融合后取 top_k。

        返回 top_k 条结果，每条含：
            score             融合分数（归一化后，含 both 加成）
            dense_score       dense 路原始 cosine（未命中则为 0）
            keyword_score     keyword 路归一化分（未命中则为 0）
            source            "dense" / "keyword" / "both"
            hit_message       命中的单条消息
            context_messages  命中消息前后多轮上下文（含命中消息本身，按对话顺序）

        按 context_ids 去重，避免同一对话片段反复占用 TopN。
        """
        if not query or not query.strip():
            return []
        if self.embeddings is None or len(self.messages) == 0:
            return []

        q = query.strip()
        recall_n = max(top_k, self._RECALL_N)
        dense_results = self._dense_search(
            q, top_n=recall_n, sender_name=sender_name, chat_type=chat_type,
            min_score=min_score,
        )
        kw_results = self._keyword_search(
            q, top_n=recall_n, sender_name=sender_name, chat_type=chat_type,
        )
        return self._fuse_results(dense_results, kw_results, top_k, min_score)

    def _dense_search(
        self,
        query: str,
        top_n: int = 20,
        sender_name: str = "",
        chat_type: str = "",
        min_score: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """dense 向量召回：BGE 语义相似度 top_n。"""
        if self.embeddings is None or len(self.messages) == 0:
            return []

        q_emb = self.encoder.encode([query])  # (1, 512)
        # 已归一化，点积即 cosine
        scores = self.embeddings @ q_emb[0]

        # 同发送者 / 同聊天类型轻微加权（与 digital-twin V4 一致）
        # 注意：真实索引的 sender_index / chat_type_index 存的是消息 id 字符串，
        # 需经 id_to_idx 映射回行下标，不能直接当整数下标用。
        if sender_name and sender_name in self.sender_index:
            for mid in self.sender_index[sender_name]:
                i = self.id_to_idx.get(mid)
                if i is not None and i < len(scores):
                    scores[i] += 0.05
        if chat_type and chat_type in self.chat_type_index:
            for mid in self.chat_type_index[chat_type]:
                i = self.id_to_idx.get(mid)
                if i is not None and i < len(scores):
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
                    "source": "dense",
                }
            )
            if len(results) >= top_n:
                break
        return results

    def _keyword_search(
        self,
        query: str,
        top_n: int = 20,
        sender_name: str = "",
        chat_type: str = "",
    ) -> List[Dict[str, Any]]:
        """keyword 关键字召回：精确词命中 top_n。

        分词参照 MemoryEngine.search_keyword（按空格切，保留 len>=2 的词），但不做
        wiki 别名扩展（那是 wiki 检索专有，原文检索不需要）。打分用命中词召回率
        （命中词数 / 总查询词数），sender/chat_type 同向加权。
        """
        # 分词：按空格切，去掉太短的词
        raw_keywords = [kw.strip() for kw in query.split() if len(kw.strip()) >= 2]
        if not raw_keywords:
            raw_keywords = [query.strip()]
        keywords = list(dict.fromkeys(raw_keywords))
        total_kw = len(keywords)
        if total_kw == 0:
            return []

        kw_set_sender = (
            set(self.sender_index.get(sender_name, [])) if sender_name else None
        )
        kw_set_chat = (
            set(self.chat_type_index.get(chat_type, [])) if chat_type else None
        )

        scored: List[Dict[str, Any]] = []
        for msg in self.messages:
            text = msg.get("text") or ""
            if not text:
                continue
            hit = sum(1 for kw in keywords if kw in text)
            if hit == 0:
                continue
            # 命中词召回率（0~1）+ sender/chat_type 同向加权
            score = hit / total_kw
            mid = msg.get("id")
            if kw_set_sender is not None and mid in kw_set_sender:
                score += 0.05
            if kw_set_chat is not None and mid in kw_set_chat:
                score += 0.03
            scored.append({"_score": score, "msg": msg})

        if not scored:
            return []
        scored.sort(key=lambda x: x["_score"], reverse=True)

        results: List[Dict[str, Any]] = []
        seen_contexts: set = set()
        for item in scored:
            if len(results) >= top_n:
                break
            msg = item["msg"]
            context_ids = msg.get("context_ids") or [msg["id"]]
            context_key = tuple(context_ids)
            if context_key in seen_contexts:
                continue
            seen_contexts.add(context_key)

            context_msgs = [self.msg_by_id.get(cid) for cid in context_ids]
            context_msgs = [m for m in context_msgs if m]
            results.append(
                {
                    "score": item["_score"],
                    "hit_message": msg,
                    "context_messages": context_msgs,
                    "context_key": context_key,
                    "source": "keyword",
                }
            )
        return results

    def _fuse_results(
        self,
        dense_results: List[Dict[str, Any]],
        kw_results: List[Dict[str, Any]],
        top_k: int,
        min_score: float,
    ) -> List[Dict[str, Any]]:
        """两路结果按 context_key 合并、分数归一化加权融合。

        - dense 分数已是 cosine（约 0~1），直接用
        - keyword 分数 max 归一化到 0~1（除以该路最高分）
        - fusion = alpha*dense + (1-alpha)*keyword
        - 两路都命中的 context（source="both"）额外乘 _BOTH_BOOST
        - 按 fusion 降序取 top_k，min_score 作用于 fusion
        """
        if not dense_results and not kw_results:
            return []

        # keyword 路 max 归一化
        kw_max = max((r["score"] for r in kw_results), default=0.0)
        if kw_max <= 0:
            kw_max = 1.0

        merged: Dict[tuple, Dict[str, Any]] = {}
        for r in dense_results:
            key = r["context_key"]
            merged[key] = {
                "dense_score": r["score"],
                "keyword_score": 0.0,
                "source": "dense",
                "hit_message": r["hit_message"],
                "context_messages": r["context_messages"],
            }
        for r in kw_results:
            key = r["context_key"]
            if key in merged:
                merged[key]["keyword_score"] = r["score"] / kw_max
                merged[key]["source"] = "both"
                # keyword 命中的消息若与 dense 不同，保留更靠前的命中（取 dense 优先）
            else:
                merged[key] = {
                    "dense_score": 0.0,
                    "keyword_score": r["score"] / kw_max,
                    "source": "keyword",
                    "hit_message": r["hit_message"],
                    "context_messages": r["context_messages"],
                }

        alpha = self._FUSE_ALPHA
        fused: List[Dict[str, Any]] = []
        for item in merged.values():
            fusion = alpha * item["dense_score"] + (1 - alpha) * item["keyword_score"]
            if item["source"] == "both":
                fusion *= self._BOTH_BOOST
            item["score"] = fusion
            fused.append(item)

        fused.sort(key=lambda x: x["score"], reverse=True)

        results: List[Dict[str, Any]] = []
        for item in fused:
            if item["score"] < min_score:
                continue
            results.append(item)
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def format_results(
        results: List[Dict[str, Any]], query: str, max_chars: int = 4000
    ) -> str:
        """把检索结果格式化为 LLM 可读的原文片段文本。"""
        if not results:
            return f"未找到与「{query}」相关的历史聊天原文。"

        lines: List[str] = [f"【历史聊天原文检索】查询：{query}", f"共 {len(results)} 条相关片段："]
        for i, r in enumerate(results, 1):
            hit = r["hit_message"]
            source = r.get("source")
            source_tag = f"·{source}" if source else ""
            head = (
                f"\n--- 片段 {i}（相关度 {r['score']:.3f}{source_tag}）"
                f" | 会话：{hit.get('chat_name', '?')}"
                f" | 类型：{hit.get('chat_type', '?')} ---"
            )
            lines.append(head)
            for m in r["context_messages"]:
                role = "你" if m.get("is_self") else (m.get("sender") or "对方")
                text = (m.get("text") or "").strip()
                if not text:
                    continue
                lines.append(f"{role}: {text}")
            if len("\n".join(lines)) > max_chars:
                lines.append("（…更多片段已省略）")
                break
        out = "\n".join(lines)
        if len(out) > max_chars:
            out = out[: max_chars - 20].rstrip() + "\n（…已截断）"
        return out


# ── 模块单例：懒加载，线程安全 ──

_singleton: Optional[HistorySearchIndex] = None
_singleton_lock = threading.Lock()


def get_history_index() -> Optional[HistorySearchIndex]:
    """获取全局 HistorySearchIndex 单例，首次调用时懒加载。

    任何加载失败都返回 None 并记日志——调用方（工具 adapter）应优雅降级。
    """
    global _singleton
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is not None:
            return _singleton
        try:
            _singleton = HistorySearchIndex()
            return _singleton
        except Exception as e:
            _logger.warning("[HistorySearch] 索引加载失败，search_history 将不可用: %s", e, exc_info=True)
            return None


def reset_singleton() -> None:
    """测试用：清空单例缓存。"""
    global _singleton
    with _singleton_lock:
        _singleton = None


def search_history(query: str, top_k: int = 5, max_chars: int = 4000) -> str:
    """工具入口：检索历史聊天原文，返回格式化文本。

    索引/依赖不可用时返回明确的不可用提示，而非抛异常。
    """
    if not query or not query.strip():
        return "查询不能为空。"
    idx = get_history_index()
    if idx is None:
        return (
            "历史原文检索（search_history）当前不可用：索引或编码器依赖未就绪。"
            "可检查 WECHAT_HISTORY_INDEX_PATH / WECHAT_BGE_MODEL_PATH 及 onnxruntime+transformers 是否安装。"
        )
    try:
        results = idx.search(query.strip(), top_k=max(1, min(int(top_k), 20)))
        return idx.format_results(results, query.strip(), max_chars=max_chars)
    except Exception as e:
        _logger.warning("[HistorySearch] 检索失败 query=%r: %s", query, e, exc_info=True)
        return f"历史原文检索出错: {e}"
