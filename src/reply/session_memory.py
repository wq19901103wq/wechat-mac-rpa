#!/usr/bin/env python3
"""Session Memory - 跨 tick 短期记忆（工具缓存、对话上下文）."""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CachedToolResult:
    """单个工具结果的缓存项."""
    tool_name: str
    query: str          # 查询摘要，用于去重
    result: str         # 结果内容（已截断）
    timestamp: float    # 缓存时间戳
    ttl_seconds: int    # 存活时间

    @property
    def expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl_seconds

    @property
    def age_seconds(self) -> int:
        return int(time.time() - self.timestamp)

    @property
    def remain_seconds(self) -> int:
        return max(0, self.ttl_seconds - self.age_seconds)

    def format_line(self) -> str:
        ts = time.strftime("%H:%M", time.localtime(self.timestamp))
        if self.expired:
            status = "已过期"
        else:
            status = f"剩{self.remain_seconds // 60}分钟"
        # 结果截断到 300 字，避免 prompt 过长
        result_short = self.result[:300]
        if len(self.result) > 300:
            result_short += "..."
        return f'- {self.tool_name}("{self.query}") [{ts} 缓存，{status}] → {result_short}'


@dataclass
class SessionSnapshot:
    """单次聊天的完整上下文快照."""
    chat_name: str
    is_group: bool = False
    last_active: float = field(default_factory=time.time)
    tool_cache: List[CachedToolResult] = field(default_factory=list)

    def add_tool_result(self, tool_name: str, query: str, result: str, ttl: int):
        # 去重：同 tool + 同 query 覆盖旧结果
        for item in self.tool_cache:
            if item.tool_name == tool_name and item.query == query:
                item.result = result
                item.timestamp = time.time()
                item.ttl_seconds = ttl
                self.last_active = time.time()
                return
        self.tool_cache.append(CachedToolResult(
            tool_name=tool_name,
            query=query,
            result=result,
            timestamp=time.time(),
            ttl_seconds=ttl,
        ))
        self.last_active = time.time()

    def get_valid_cache(self) -> List[CachedToolResult]:
        """返回未过期的缓存（按时间倒序）."""
        valid = [c for c in self.tool_cache if not c.expired]
        return sorted(valid, key=lambda x: x.timestamp, reverse=True)

    def get_all_cache(self) -> List[CachedToolResult]:
        """返回所有缓存含过期的（按时间倒序）."""
        return sorted(self.tool_cache, key=lambda x: x.timestamp, reverse=True)

    def cleanup_expired(self):
        """清理过期超过 2 倍 TTL 的缓存（保留近期过期记录供参考）."""
        now = time.time()
        self.tool_cache = [
            c for c in self.tool_cache
            if not c.expired or (now - c.timestamp) < c.ttl_seconds * 2
        ]


class SessionMemory:
    """管理所有聊天的短期记忆，按 chat_name 隔离."""

    # 各工具的默认 TTL（秒）
    DEFAULT_TTL = {
        "web_search": 300,       # 5 分钟
        "stock_query": 60,       # 1 分钟
        "get_weather": 1800,     # 30 分钟
        "search_memory": 600,    # 10 分钟
        "get_current_time": 0,   # 不缓存
    }

    def __init__(self):
        self._sessions: Dict[str, SessionSnapshot] = {}

    def get_or_create(self, chat_name: str, is_group: bool = False) -> SessionSnapshot:
        if chat_name not in self._sessions:
            self._sessions[chat_name] = SessionSnapshot(chat_name=chat_name, is_group=is_group)
        else:
            # 如果传入的 is_group 与当前不同，更新它
            if self._sessions[chat_name].is_group != is_group:
                self._sessions[chat_name].is_group = is_group
            self._sessions[chat_name].last_active = time.time()
        return self._sessions[chat_name]

    def add_tool_result(self, chat_name: str, tool_name: str, query: str, result: str):
        """保存工具执行结果到 session 缓存."""
        ttl = self.DEFAULT_TTL.get(tool_name, 300)
        if ttl <= 0:
            return
        session = self.get_or_create(chat_name)
        session.add_tool_result(tool_name, query, result, ttl)

    def get_cache_lines(self, chat_name: str, include_expired: bool = False) -> List[str]:
        """获取格式化的缓存行列表（用于 prompt 注入）."""
        session = self._sessions.get(chat_name)
        if not session:
            return []
        session.cleanup_expired()
        if include_expired:
            cache = session.get_all_cache()
        else:
            cache = session.get_valid_cache()
        return [c.format_line() for c in cache]

    def cleanup_stale_sessions(self, max_idle_seconds: int = 3600):
        """清理超过 max_idle_seconds 未活跃的 session."""
        now = time.time()
        stale = [
            name for name, s in self._sessions.items()
            if now - s.last_active > max_idle_seconds
        ]
        for name in stale:
            del self._sessions[name]


def _extract_query_key(tool_name: str, tool_args: str) -> str:
    """从工具参数 JSON 中提取 query key（用于去重）."""
    try:
        args = json.loads(tool_args)
    except Exception:
        return tool_args[:50]

    mapping = {
        "web_search": "query",
        "stock_query": "stock_code",
        "search_memory": "query",
        "get_weather": "city",
    }
    key = mapping.get(tool_name)
    if key and key in args:
        return str(args[key])
    # fallback：拼接所有参数值
    return " ".join(str(v) for v in args.values())[:50]
