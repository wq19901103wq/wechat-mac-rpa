"""全量历史原文检索（关键词精确匹配，不依赖 BGE 编码器）。

直接加载 digital-twin 的 vector_index_dense_messages.pkl（78 万条消息），
按关键词检索，返回带 is_self 标记的证据（is_self=True = Bot 发的）。

与 history_search.py 的区别：
- history_search 走 BGE 语义检索（需 onnxruntime/transformers，依赖常缺）
- 本模块走关键词精确匹配，零外部依赖，启动即用
- 用于 wiki audit 的事实核查：精确判定"谁说了什么"

证据可信度判定（audit 用）：
- is_self=True（Bot 发的）：Bot 以林岚身份自述 = 林岚授权说的，可信
- is_self=False（真人发的）：sender 是真实用户，最可信
- 关键词零命中：历史里没人说过，wiki 该事实可能是 LLM 编造
"""

import logging
import os
import pickle  # nosec B403
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_logger = logging.getLogger("src.memory.history_lookup")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INDEX_PATH = (
    _PROJECT_ROOT / "data" / "memory" / "cache" / "vector_index_dense_messages.pkl"
)


def _index_path() -> Path:
    return Path(os.environ.get("WECHAT_HISTORY_INDEX_PATH", _DEFAULT_INDEX_PATH))


class HistoryLookup:
    """单例全量历史检索。首次调用加载 1.7GB pkl（~10s），后续复用。"""

    _instance: Optional["HistoryLookup"] = None
    _lock = threading.Lock()

    def __init__(self, index_path: Optional[Path] = None) -> None:
        path = index_path or _index_path()
        if not path.exists():
            raise FileNotFoundError(f"历史索引不存在: {path}")
        _logger.info("[HistoryLookup] 加载 %s ...", path)
        with open(path, "rb") as f:
            data = pickle.load(f)  # nosec B301
        msgs = data.get("messages", [])
        # 预处理：text 转 str，is_self 统一成 bool
        self._msgs: List[Dict[str, Any]] = []
        for m in msgs:
            t = m.get("text", "")
            if t is None:
                continue
            self._msgs.append({
                "text": str(t),
                "sender": str(m.get("sender", "")),
                "is_self": str(m.get("is_self", "")).lower() == "true",
                "chat_name": str(m.get("chat_name", "")),
                "timestamp": m.get("timestamp"),
            })
        _logger.info("[HistoryLookup] 加载完成，%d 条消息", len(self._msgs))

    @classmethod
    def get(cls) -> Optional["HistoryLookup"]:
        """获取单例，失败返回 None（不抛异常，调用方降级）。"""
        if cls._instance is not None:
            return cls._instance
        with cls._lock:
            if cls._instance is not None:
                return cls._instance
            try:
                cls._instance = cls()
            except Exception as e:
                _logger.warning("[HistoryLookup] 加载失败: %s", e)
                return None
        return cls._instance

    def search(
        self,
        keywords: List[str],
        chats: Optional[List[str]] = None,
        limit: int = 8,
        exclude_self: bool = False,
    ) -> List[Dict[str, Any]]:
        """关键词 AND 检索（所有 keyword 都需出现在 text 里）。

        返回 [{chat_name, ts, who, is_self, text}]。
        - chats: 限定 chat_name 含其中任一关键词（None=不限）
        - exclude_self: 排除 Bot 自述（仅看真人说的）
        """
        kws = [k for k in keywords if k]
        if not kws:
            return []
        results: List[Dict[str, Any]] = []
        for m in self._msgs:
            if exclude_self and m["is_self"]:
                continue
            t = m["text"]
            if not all(k in t for k in kws):
                continue
            if chats:
                cn = m["chat_name"]
                if not any(c in cn for c in chats):
                    continue
            ts = ""
            if m["timestamp"]:
                try:
                    ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(int(m["timestamp"])))
                except (ValueError, TypeError, OSError):
                    ts = ""
            who = "Bot" if m["is_self"] else m["sender"]
            results.append({
                "chat_name": m["chat_name"],
                "ts": ts,
                "who": who,
                "is_self": m["is_self"],
                "text": t[:200],
            })
            if len(results) >= limit:
                break
        return results

    def count(self, keywords: List[str], chats: Optional[List[str]] = None) -> int:
        """统计命中数（用于判断"历史里有没有人说过"）。"""
        kws = [k for k in keywords if k]
        if not kws:
            return 0
        n = 0
        for m in self._msgs:
            t = m["text"]
            if not all(k in t for k in kws):
                continue
            if chats:
                cn = m["chat_name"]
                if not any(c in cn for c in chats):
                    continue
            n += 1
        return n


def format_evidence(hits: List[Dict[str, Any]]) -> str:
    """把检索结果格式化成 audit prompt 用的证据文本。"""
    if not hits:
        return "（历史消息中无匹配）"
    lines = []
    for h in hits:
        tag = "🤖Bot" if h["is_self"] else f"👤{h['who']}"
        lines.append(f"[{h['chat_name']}][{h['ts']}] {tag}: {h['text']}")
    return "\n".join(lines)
