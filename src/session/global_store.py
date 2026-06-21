#!/usr/bin/env python3
"""全局消息存储 - 管理所有聊天的消息历史和回复状态."""

import difflib
import hashlib
import json
import logging
import os
import threading
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.models.base import MEDIA_MESSAGE_TYPES, ChatMessage, SenderType
from src.utils.chat_utils import _is_group_chat_name

_logger = logging.getLogger("src.global_store")


@dataclass
class ChatState:
    """单个聊天的完整状态（消息历史 + 会话状态）"""
    chat_id: str
    chat_name: str
    is_group: bool = False
    messages: List[ChatMessage] = field(default_factory=list)
    _msg_ids: set = field(default_factory=set)  # 去重集合（不序列化）
    pending_self_messages: List[ChatMessage] = field(default_factory=list)  # Bot 发送成功但尚未被感知层确认的消息（诊断用，不序列化）


def _normalize_text(text: str) -> str:
    """文本归一化：压缩连续空白为单个空格，去除首尾空白。"""
    return " ".join(text.split())


def _safe_filename(name: str) -> str:
    """把聊天名转成安全的文件名（替换非法字符，限制长度）。"""
    invalid = '<>:"/\\|?*'
    for c in invalid:
        name = name.replace(c, '_')
    # 限制长度，保留尾部用于可读性
    if len(name) > 180:
        name = name[:180]
    return name


def _jaccard_2gram(a: str, b: str) -> float:
    """计算两个字符串的 2-gram Jaccard 相似度。"""
    if not a or not b:
        return 0.0
    ga = set(a[i:i + 2] for i in range(len(a) - 1))
    gb = set(b[i:i + 2] for i in range(len(b) - 1))
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union if union else 0.0


def _normalize_sender(chat_name: str, msg: ChatMessage, is_group: bool = False) -> str:
    """标准化 sender 用于去重匹配。

    规则：
    - 自己发的消息 → "自己"
    - 私聊对方：sender 是"对方"/空/"[未知]" → 用 chat_name（对方昵称）替代
    - 群聊对方：保留原始 sender（具体昵称或"对方"）
    """
    if msg.sender_type == SenderType.SELF:
        return "自己"
    if not is_group and not _is_group_chat_name(chat_name):
        # 私聊：对方 sender 统一为 chat_name，避免 API 昵称识别不稳定导致去重失效
        return chat_name
    # 群聊：保留原始 sender（具体昵称或"对方"）
    return msg.sender


def _msg_id(chat_name: str, msg: ChatMessage, is_group: bool = False) -> str:
    """消息唯一ID：用 chat_name + 标准化 sender + 内容指纹。

    文字消息：基于 text。
    图片/表情/混合消息：基于 message_type + image_description，避免不同图片
    因 text 都为空而被误判为相同。
    """
    if msg.message_type in MEDIA_MESSAGE_TYPES:
        content = f"[{msg.message_type}]{msg.image_description}"
    else:
        content = msg.text
    normalized = _normalize_text(content)
    text_hash = hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()[:16]
    normalized_sender = _normalize_sender(chat_name, msg, is_group)
    return f"{chat_name}|{normalized_sender}|{text_hash}"


def _is_fuzzy_duplicate(state, msg: ChatMessage, lookback: int = 10) -> bool:
    """模糊去重：对最近 lookback 条消息做文本相似度比较。

    OCR 偶尔错几个字，精确 hash 会失效。用 difflib.SequenceMatcher
    计算相似度，>= threshold 视为同一条消息。

    图片/表情/混合消息：基于 image_description 做 2-gram Jaccard 模糊去重。

    阈值按消息长度动态调整：越短的消息要求越严格（避免不同短句误判）。
    只对 lookback 条消息比较，避免遍历全部历史影响性能。
    """
    # 媒体消息：基于 image_description 做 2-gram Jaccard
    if msg.message_type in MEDIA_MESSAGE_TYPES:
        desc = msg.image_description
        if not desc:
            return False
        for hist_msg in state.messages[-lookback:]:
            if hist_msg.sender_type.value == "self":
                continue
            if hist_msg.message_type not in MEDIA_MESSAGE_TYPES:
                continue
            hist_desc = hist_msg.image_description
            if not hist_desc:
                continue
            sim = _jaccard_2gram(desc, hist_desc)
            if sim >= 0.08:
                return True
        return False

    text = msg.text
    if not text:
        return False

    # 按长度动态调整阈值（OCR 对中文短句容易错 1-2 个字，适当放宽）
    text_len = len(text)
    if text_len <= 3:
        threshold = 0.90
    elif text_len <= 8:
        threshold = 0.85
    elif text_len <= 20:
        threshold = 0.82
    else:
        threshold = 0.80

    normalized = _normalize_text(text)
    for hist_msg in state.messages[-lookback:]:
        # 跳过 Bot 自己的消息，避免拿 Bot 回复去重用户新消息
        if hist_msg.sender_type.value == "self":
            continue
        # 跳过媒体类消息（不参与文字模糊去重）
        if hist_msg.message_type in MEDIA_MESSAGE_TYPES:
            continue
        other = _normalize_text(hist_msg.text)
        if not other:
            continue
        similarity = difflib.SequenceMatcher(None, normalized, other).ratio()
        if similarity >= threshold:
            return True
    return False


def _match_single(a: ChatMessage, b: ChatMessage, chat_name: str, is_group: bool = False) -> bool:
    """直接比较两条消息是否匹配（用于对齐）。

    文字：SequenceMatcher >= 0.80
    图片：2-gram Jaccard >= 0.001（容错极大，应对 qwen 描述不稳定）
    """
    # 精确匹配（使用标准化 sender）
    if _msg_id(chat_name, a, is_group) == _msg_id(chat_name, b, is_group):
        return True
    # sender_type 不同直接不匹配（避免自己消息和对方消息误匹配）
    if a.sender_type != b.sender_type:
        return False
    # 类型不同直接不匹配
    if a.message_type != b.message_type:
        return False
    # 文字消息
    if a.message_type == "text":
        text_a = _normalize_text(a.text)
        text_b = _normalize_text(b.text)
        if not text_a or not text_b:
            return False
        return difflib.SequenceMatcher(None, text_a, text_b).ratio() >= 0.80
    # 图片/表情/混合：合并 image_description + image_text 后做 Jaccard
    # 应对 API 描述不稳定（有时把图上文字放 image_description，有时放 image_text）
    combined_a = f"{a.image_description or ''} {a.image_text or ''}".strip()
    combined_b = f"{b.image_description or ''} {b.image_text or ''}".strip()
    if not combined_a or not combined_b:
        return False
    sim = _jaccard_2gram(combined_a, combined_b)
    return sim >= 0.20


def _lcs_match(history: List[ChatMessage], tick: List[ChatMessage], chat_name: str, is_group: bool = False) -> set:
    """LCS 序列对齐：返回 tick 中匹配 history 的索引集合。

    使用二值 match_score：_match_single 返回 True → 得 1 分，否则 0 分。
    回溯得到 matched_tick_indices，用于判断 tick 中哪些消息是旧的。
    """
    m, n = len(history), len(tick)
    if m == 0 or n == 0:
        return set()

    # 性能保护：限制参与比对的 history 范围，避免大群 O(m*n) 爆炸
    _MAX_HISTORY_FOR_LCS = 80
    if m > _MAX_HISTORY_FOR_LCS:
        _logger.info(f"[LCS] history={m} 超过上限 {_MAX_HISTORY_FOR_LCS}，截取最近部分")
        history = history[-_MAX_HISTORY_FOR_LCS:]
        m = _MAX_HISTORY_FOR_LCS

    # dp[i][j] = history[0:i] 和 tick[0:j] 的 LCS 长度
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if _match_single(history[i - 1], tick[j - 1], chat_name, is_group):
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 回溯找匹配的 tick 索引
    matched = set()
    i, j = m, n
    while i > 0 and j > 0:
        if _match_single(history[i - 1], tick[j - 1], chat_name, is_group):
            # match 时 dp[i][j] 一定等于 dp[i-1][j-1]+1（单调性保证）
            matched.add(j - 1)
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j]:
            i -= 1
        else:
            j -= 1

    return matched


class GlobalStore:
    """全局存储：管理所有聊天的状态，统一去重、持久化."""

    def __init__(self, max_messages: int = 200, state_file: Optional[str] = None):
        if state_file is None:
            state_file = str(Path(__file__).parent.parent.parent / "data" / "global_state.json")
        self.chats: Dict[str, ChatState] = {}
        self.max_messages = max_messages
        self._state_file = Path(state_file)
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._screenshots_dir = self._state_file.parent / "screenshots"
        self._screenshots_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._dirty: set = set()  # 有变化的聊天名，增量保存用
        self._load()

    def _merge_tick_legacy(self, chat_name: str, messages: List[ChatMessage], is_group: bool = False) -> List[ChatMessage]:
        """旧算法：滑动前缀匹配（保留用于 A/B 对比测试）。"""
        state = self.chats[chat_name]

        def _in_history(msg: ChatMessage) -> bool:
            return _msg_id(chat_name, msg, is_group) in state._msg_ids or _is_fuzzy_duplicate(
                state, msg, lookback=len(state.messages)
            )

        # tick 内去重
        seen_ids = set()
        unique_messages = []
        for msg in messages:
            mid = _msg_id(chat_name, msg, is_group)
            if mid not in seen_ids:
                seen_ids.add(mid)
                unique_messages.append(msg)
        messages = unique_messages

        if not messages or not state.messages:
            return messages if not state.messages else []

        search_window = min(len(state.messages), max(50, len(messages) * 3))
        history_window = state.messages[-search_window:]

        best_match_len = 0
        best_match_start = -1
        for i in range(len(history_window)):
            match_len = 0
            for j in range(len(messages)):
                if i + j >= len(history_window):
                    break
                if _match_single(history_window[i + j], messages[j], chat_name, is_group):
                    match_len += 1
                else:
                    break
            if match_len > best_match_len:
                best_match_len = match_len
                best_match_start = i

        if best_match_len == len(messages):
            return []
        elif best_match_len >= 1:
            match_end_in_history = best_match_start + best_match_len
            if match_end_in_history >= len(history_window) - 2:
                return messages[best_match_len:]
            else:
                return [msg for msg in messages if not _in_history(msg)]
        else:
            return [msg for msg in messages if not _in_history(msg)]

    def _merge_tick_lcs(self, chat_name: str, messages: List[ChatMessage], is_group: bool = False) -> List[ChatMessage]:
        """新算法：LCS 序列对齐（独立出来用于 A/B 对比测试）。"""
        state = self.chats[chat_name]

        def _in_history(msg: ChatMessage) -> bool:
            return _msg_id(chat_name, msg, is_group) in state._msg_ids or _is_fuzzy_duplicate(
                state, msg, lookback=len(state.messages)
            )

        # tick 内去重
        seen_ids = set()
        unique_messages = []
        for msg in messages:
            mid = _msg_id(chat_name, msg, is_group)
            if mid not in seen_ids:
                seen_ids.add(mid)
                unique_messages.append(msg)
        messages = unique_messages

        if not messages or not state.messages:
            if not state.messages:
                _logger.debug("[LCS] %s history 为空，全部 %d 条视为新消息", chat_name, len(messages))
            return messages if not state.messages else []

        search_window = min(len(state.messages), 50)
        history_window = state.messages[-search_window:]
        matched = _lcs_match(history_window, messages, chat_name, is_group)

        # --- 诊断日志：LCS 匹配全过程 ---
        _logger.debug("[LCS] %s tick=%d条 history_window=%d条 matched=%s",
                      chat_name, len(messages), len(history_window), matched)
        for i, msg in enumerate(history_window[-10:]):
            _logger.debug("[LCS] history[-%d] %s(%s): %.40s",
                          len(history_window) - i, msg.sender, msg.sender_type.value, msg.text or "")
        for i, msg in enumerate(messages):
            match_flag = "✓" if i in matched else "✗"
            _logger.debug("[LCS] tick[%d] %s %s(%s): %.40s",
                          i, match_flag, msg.sender, msg.sender_type.value, msg.text or "")

        if not matched:
            new_messages = [msg for msg in messages if not _in_history(msg)]
            _logger.debug("[LCS] %s 无 LCS 匹配，_in_history 过滤后新消息=%d条", chat_name, len(new_messages))
            return new_messages

        max_matched = max(matched)

        # 分类统计
        discarded = []
        new_messages = []
        for i in range(len(messages)):
            if i not in matched:
                if i > max_matched:
                    new_messages.append(messages[i])
                else:
                    discarded.append(messages[i])

        # 告警：被丢弃的消息
        for msg in discarded:
            _logger.debug("[LCS] %s DISCARDED %s(%s): %.60s",
                          chat_name, msg.sender, msg.sender_type.value, msg.text or "")
            if msg.sender_type == SenderType.SELF:
                _logger.warning("[LCS] %s 未匹配的 self 消息（发送疑似失败或 OCR 误识别）: %.60s",
                                chat_name, msg.text or "")
            else:
                _logger.warning("[LCS] %s 被 i>max_matched 规则丢弃的真实消息: %.60s",
                                chat_name, msg.text or "")

        _logger.debug("[LCS] %s max_matched=%d discarded=%d new=%d",
                      chat_name, max_matched, len(discarded), len(new_messages))
        # --- 诊断日志结束 ---

        return new_messages

    def merge_tick(
        self,
        chat_name: str,
        messages: List[ChatMessage],
        mode: str = "ocr",
        is_group: bool = False,
    ) -> Tuple[ChatState, List[ChatMessage]]:
        """
        合并 tick 检测到的消息，返回 (state, 未回复的消息列表).

        mode: "ocr" | "weflow" | "hybrid"
            - ocr: 使用原有 LCS + 模糊匹配去重
            - weflow/hybrid: 使用 localId 精确去重（如果消息有 local_id）
        """
        if chat_name not in self.chats:
            self.chats[chat_name] = ChatState(
                chat_id=f"chat_{len(self.chats)}",
                chat_name=chat_name,
                is_group=is_group,
            )

        state = self.chats[chat_name]
        # 如果传入的 is_group 与当前状态不同，更新状态
        if state.is_group != is_group:
            state.is_group = is_group
            self._dirty.add(chat_name)

        # 选择去重策略
        if mode in ("weflow", "hybrid"):
            _logger.info("[GlobalStore] %s 使用 WeFlow 精确去重 (%d 条)", chat_name, len(messages))
            new_messages = self._merge_tick_weflow(chat_name, messages)
        else:
            new_messages = self._merge_tick_lcs(chat_name, messages, is_group)

        # 添加新消息到历史
        if new_messages:
            self._dirty.add(chat_name)
        for msg in new_messages:
            new_msg = replace(msg, chat_name=chat_name)
            state.messages.append(new_msg)
            state._msg_ids.add(_msg_id(chat_name, new_msg, is_group))

        # 裁剪历史消息，避免无限增长
        if len(state.messages) > self.max_messages:
            state.messages = state.messages[-self.max_messages:]
            state._msg_ids = {
                _msg_id(chat_name, m, state.is_group)
                for m in state.messages
            }
            self._dirty.add(chat_name)

        # 收集所有未回复的消息（按时间顺序）
        unreplied = [
            msg for msg in state.messages
            if not msg.replied and msg.sender_type != SenderType.SELF
        ]

        return state, unreplied

    def mark_replied(self, chat_name: str, target_msg: ChatMessage, reply_text: str):
        """标记单条消息已回复。"""
        state = self.chats.get(chat_name)
        if not state:
            return
        now = time.time()

        # 用 is 匹配（target_msg 就是 state.messages 中的对象引用）
        # 如果 is 匹配不到，再用 text+sender 兜底
        marked = False
        for msg in state.messages:
            if msg is target_msg or (msg.text == target_msg.text and msg.sender == target_msg.sender):
                msg.replied = True
                msg.reply_text = reply_text
                msg.reply_time = now
                marked = True
                # 不 break，继续标记所有匹配的消息（OCR 不稳定可能导致同一条消息存了多份）
        if marked:
            self._dirty.add(chat_name)

    def _merge_tick_weflow(self, chat_name: str, messages: List[ChatMessage]) -> List[ChatMessage]:
        """WeFlow 精确去重：基于 localId + Bot 消息 text 去重。

        复杂度：O(n)，只需要一次集合查找。
        对 Bot 自己发送的消息基于 text 去重，但保留最新的 reply_time，
        避免 Bot 手动注入的回复被跳过后，历史上下文里看不到 bot 自己的消息。
        """
        state = self.chats[chat_name]
        seen_ids = {
            getattr(m, "local_id", None)
            for m in state.messages
            if hasattr(m, "local_id") and getattr(m, "local_id", None) is not None
        }
        # Bot 自己发的消息：保存对象引用，以便更新 reply_time
        bot_entries = {
            m.text: m
            for m in state.messages
            if m.sender_type == SenderType.SELF
        }
        new_messages = []
        for m in messages:
            lid = getattr(m, "local_id", None)
            if lid is not None and lid in seen_ids:
                continue
            # Bot 自己发的消息，如果 text 已存在，更新 reply_time 而不是跳过
            if m.sender_type == SenderType.SELF and m.text in bot_entries:
                existing = bot_entries[m.text]
                m_reply_time = getattr(m, "reply_time", None)
                existing_reply_time = getattr(existing, "reply_time", None)
                if m_reply_time and (
                    not existing_reply_time
                    or m_reply_time > existing_reply_time
                ):
                    existing.reply_time = m_reply_time
                continue
            new_messages.append(m)
            if lid is not None:
                seen_ids.add(lid)
            if m.sender_type == SenderType.SELF:
                bot_entries[m.text] = m
        if new_messages:
            _logger.debug(
                "[GlobalStore] WeFlow 去重: %s 新消息 %d 条 (历史 %d 条)",
                chat_name,
                len(new_messages),
                len(state.messages),
            )
        return new_messages

    def inject_history(
        self,
        chat_name: str,
        messages: List[ChatMessage],
        mode: str = "weflow",
        is_group: bool = False,
    ) -> int:
        """批量注入历史消息（WeFlow 全量初始化时使用）。

        与 merge_tick 不同：
        - 不裁剪历史（允许超过 max_messages）
        - 不去重（假设输入已去重）
        - 标记 replied=True（历史消息不需要回复）

        Returns:
            注入的消息数量
        """
        # 按 create_time 正序排序，防止 WeFlow 返回倒序消息导致
        # merge_tick 裁剪时误删较新的历史（含 bot 自己的回复）
        if messages:
            messages = sorted(messages, key=lambda m: getattr(m, "create_time", 0) or 0)

        if chat_name not in self.chats:
            self.chats[chat_name] = ChatState(
                chat_id=f"chat_{len(self.chats)}",
                chat_name=chat_name,
                is_group=is_group,
            )

        state = self.chats[chat_name]
        # 如果传入的 is_group 与当前状态不同，更新状态
        if state.is_group != is_group:
            state.is_group = is_group
            self._dirty.add(chat_name)
        count = 0

        if mode in ("weflow", "hybrid") and messages and getattr(messages[0], "local_id", None) is not None:
            # WeFlow 模式：基于 localId 去重后注入
            seen_ids = {
                getattr(m, "local_id", None)
                for m in state.messages
                if hasattr(m, "local_id") and getattr(m, "local_id", None) is not None
            }
            for msg in messages:
                lid = getattr(msg, "local_id", None)
                if lid is not None and lid in seen_ids:
                    continue
                new_msg = replace(msg, chat_name=chat_name, replied=True)
                state.messages.append(new_msg)
                if lid is not None:
                    seen_ids.add(lid)
                count += 1
        else:
            # OCR 模式：基于原有去重键注入
            for msg in messages:
                mid = _msg_id(chat_name, msg, is_group)
                if mid in state._msg_ids:
                    continue
                new_msg = replace(msg, chat_name=chat_name, replied=True)
                state.messages.append(new_msg)
                state._msg_ids.add(_msg_id(chat_name, new_msg, is_group))
                count += 1

        if count:
            self._dirty.add(chat_name)
            _logger.debug(
                "[GlobalStore] 注入历史: %s +%d 条 (总计 %d 条)",
                chat_name,
                count,
                len(state.messages),
            )
        return count

    def get_unreplied(self, chat_name: str) -> List[ChatMessage]:
        """获取某聊天中所有未回复的消息（按时间顺序）"""
        state = self.chats.get(chat_name)
        if not state:
            return []
        return [
            m for m in state.messages
            if not m.replied and m.sender_type != SenderType.SELF
        ]

    def last_reply_time(self, chat_name: str) -> Optional[float]:
        """最后回复时间（从消息中推导）"""
        state = self.chats.get(chat_name)
        if not state:
            return None
        replied_times = [
            m.reply_time for m in state.messages
            if m.replied and m.reply_time
        ]
        return max(replied_times) if replied_times else None

    def reply_count(self, chat_name: str) -> int:
        """回复次数（从消息中推导）"""
        state = self.chats.get(chat_name)
        if not state:
            return 0
        return sum(1 for m in state.messages if m.replied)

    def _load(self):
        """从磁盘加载状态。优先加载分片格式，回退旧格式。"""
        # 1. 优先加载分片格式
        index_file = self._state_file.parent / "chats" / "index.json"
        if index_file.exists():
            self._load_sharded(index_file)
            return

        # 2. 回退旧格式（单 JSON）
        if not self._state_file.exists():
            return
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for chat_name, chat_data in data.items():
                state = ChatState(
                    chat_id=chat_data.get("chat_id", ""),
                    chat_name=chat_data.get("chat_name", chat_name),
                    is_group=chat_data.get("is_group", False),
                )
                for m in chat_data.get("messages", []):
                    msg = self._dict_to_msg(m, chat_name)
                    state.messages.append(msg)
                    state._msg_ids.add(_msg_id(chat_name, msg, state.is_group))
                self.chats[chat_name] = state
            _logger.debug("[GlobalStore] 加载旧格式: %d 个聊天", len(self.chats))
        except (json.JSONDecodeError, FileNotFoundError, PermissionError, OSError) as e:
            _logger.warning(f"加载状态失败: {type(e).__name__}: {e}")
        except Exception as e:
            _logger.error(f"加载状态发生未预期错误: {type(e).__name__}: {e}\n{traceback.format_exc()}")

    @staticmethod
    def _dict_to_msg(m: dict, chat_name: str) -> ChatMessage:
        """把字典反序列化为 ChatMessage。"""
        return ChatMessage(
            text=m.get("text", ""),
            sender=m.get("sender", ""),
            sender_type=SenderType(m.get("sender_type", "other")),
            chat_name=m.get("chat_name", chat_name),
            is_at_me=m.get("is_at_me", False),
            replied=m.get("replied", False),
            reply_text=m.get("reply_text", ""),
            reply_time=m.get("reply_time"),
            message_type=m.get("message_type", "text"),
            image_description=m.get("image_description", ""),
            image_text=m.get("image_text", ""),
            is_image_duplicate=m.get("is_image_duplicate", False),
            account=m.get("account", ""),
            local_id=m.get("local_id"),
            server_id=m.get("server_id"),
            create_time=m.get("create_time"),
            raw_type=m.get("raw_type"),
            sender_wxid=m.get("sender_wxid"),
        )

    def _load_sharded(self, index_file: Path):
        """加载分片格式。"""
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                index = json.load(f)
            chats_dir = index_file.parent
            for chat_name, meta in index.get("chats", {}).items():
                file_path = meta["file"]
                if file_path.startswith("chats/"):
                    file_path = file_path[6:]
                shard_file = chats_dir / file_path
                if not shard_file.exists():
                    _logger.warning("[GlobalStore] 分片文件缺失: %s", shard_file)
                    continue
                with open(shard_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                state = ChatState(
                    chat_id=data.get("chat_id", meta.get("chat_id", "")),
                    chat_name=data.get("chat_name", chat_name),
                    is_group=data.get("is_group", False),
                )
                for m in data.get("messages", []):
                    msg = self._dict_to_msg(m, chat_name)
                    state.messages.append(msg)
                    state._msg_ids.add(_msg_id(chat_name, msg, state.is_group))
                self.chats[chat_name] = state
            _logger.debug("[GlobalStore] 加载分片格式: %d 个聊天", len(self.chats))
        except Exception as e:
            _logger.error(f"加载分片失败: {type(e).__name__}: {e}\n{traceback.format_exc()}")

    @staticmethod
    def _msg_to_dict(m: ChatMessage) -> dict:
        """把 ChatMessage 序列化为字典。"""
        return {
            "text": m.text,
            "sender": m.sender,
            "sender_type": m.sender_type.value,
            "chat_name": m.chat_name,
            "is_at_me": m.is_at_me,
            "replied": m.replied,
            "reply_text": m.reply_text,
            "reply_time": m.reply_time,
            "message_type": m.message_type,
            "image_description": m.image_description,
            "image_text": m.image_text,
            "is_image_duplicate": m.is_image_duplicate,
            "account": m.account,
            "local_id": m.local_id,
            "server_id": m.server_id,
            "create_time": m.create_time,
            "raw_type": m.raw_type,
            "sender_wxid": m.sender_wxid,
        }

    def save(self):
        """保存状态到磁盘（分片格式，增量保存）。"""
        with self._lock:
            try:
                chats_dir = self._state_file.parent / "chats"
                chats_dir.mkdir(parents=True, exist_ok=True)

                dirty = list(self._dirty)
                if not dirty:
                    # 没有任何变化，只保存索引（确保索引是最新的）
                    pass
                else:
                    # 1. 只保存有变化的聊天
                    for chat_name in dirty:
                        state = self.chats.get(chat_name)
                        if state is None:
                            continue
                        safe_name = _safe_filename(chat_name)
                        shard_file = chats_dir / f"{safe_name}.json"
                        data = {
                            "chat_id": state.chat_id,
                            "chat_name": state.chat_name,
                            "is_group": state.is_group,
                            "messages": [self._msg_to_dict(m) for m in state.messages],
                        }
                        tmp = shard_file.with_suffix(".tmp")
                        with open(tmp, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        os.replace(tmp, shard_file)
                    self._dirty.clear()
                    _logger.debug(f"💾 增量保存 {len(dirty)} 个聊天状态")

                # 2. 保存轻量索引（始终保存，保证一致性）
                index = {
                    "version": 2,
                    "format": "sharded",
                    "chats": {
                        name: {
                            "chat_id": s.chat_id,
                            "chat_name": s.chat_name,
                            "is_group": s.is_group,
                            "msg_count": len(s.messages),
                            "file": f"chats/{_safe_filename(name)}.json",
                        }
                        for name, s in self.chats.items()
                    },
                }
                index_file = chats_dir / "index.json"
                tmp = index_file.with_suffix(".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(index, f, ensure_ascii=False, indent=2)
                os.replace(tmp, index_file)

                # 3. 旧文件重命名备份（保留一次）
                if self._state_file.exists():
                    bak = self._state_file.with_suffix(".json.bak")
                    if not bak.exists():
                        os.replace(self._state_file, bak)

            except (PermissionError, OSError) as e:
                _logger.warning(f"GlobalStore save failed (IO): {type(e).__name__}: {e}")
            except Exception as e:
                _logger.error(f"GlobalStore save failed unexpectedly: {type(e).__name__}: {e}\n{traceback.format_exc()}")

    def save_screenshot(self, image_path: str, session_id: Optional[str] = None) -> str:
        """保存截图到 data/screenshots/ 目录。"""
        import shutil
        from datetime import datetime
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"wechat_{session_id}_{timestamp}.png"
        filepath = self._screenshots_dir / filename
        shutil.copy2(image_path, filepath)
        return str(filepath)
