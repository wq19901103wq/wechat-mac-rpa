#!/usr/bin/env python3
"""SessionMemory 单元测试"""

import time

import pytest

from src.reply.session_memory import CachedToolResult, SessionMemory, SessionSnapshot, _extract_query_key


class TestCachedToolResult:
    def test_creation(self):
        c = CachedToolResult("web_search", "上海天气", "晴 25度", time.time(), 300)
        assert c.tool_name == "web_search"
        assert c.query == "上海天气"

    def test_expired(self):
        past = time.time() - 400
        c = CachedToolResult("web_search", "q", "r", past, 300)
        assert c.expired

    def test_not_expired(self):
        now = time.time()
        c = CachedToolResult("web_search", "q", "r", now, 300)
        assert not c.expired

    def test_age_and_remain(self):
        past = time.time() - 100
        c = CachedToolResult("web_search", "q", "r", past, 300)
        assert c.age_seconds == pytest.approx(100, abs=1)
        assert c.remain_seconds == pytest.approx(200, abs=1)

    def test_format_line(self):
        ts = time.time()
        c = CachedToolResult("web_search", "上海天气", "晴", ts, 300)
        line = c.format_line()
        assert "web_search" in line
        assert "上海天气" in line
        assert "缓存" in line

    def test_format_line_expired(self):
        ts = time.time() - 400
        c = CachedToolResult("web_search", "q", "r", ts, 300)
        line = c.format_line()
        assert "已过期" in line

    def test_format_line_truncation(self):
        ts = time.time()
        c = CachedToolResult("web_search", "q", "x" * 500, ts, 300)
        line = c.format_line()
        assert "..." in line
        assert len(line) < 450


class TestSessionSnapshot:
    def test_add_tool_result(self):
        s = SessionSnapshot(chat_name="test")
        s.add_tool_result("web_search", "q1", "r1", 300)
        assert len(s.tool_cache) == 1
        assert s.tool_cache[0].tool_name == "web_search"

    def test_add_tool_result_dedup(self):
        s = SessionSnapshot(chat_name="test")
        s.add_tool_result("web_search", "q1", "r1", 300)
        s.add_tool_result("web_search", "q1", "r2", 300)
        assert len(s.tool_cache) == 1
        assert s.tool_cache[0].result == "r2"

    def test_get_valid_cache(self):
        s = SessionSnapshot(chat_name="test")
        s.add_tool_result("web_search", "q1", "r1", 300)
        s.add_tool_result("stock_query", "q2", "r2", 0)  # 0 TTL = expired immediately
        valid = s.get_valid_cache()
        assert len(valid) == 1
        assert valid[0].tool_name == "web_search"

    def test_cleanup_expired(self):
        s = SessionSnapshot(chat_name="test")
        s.add_tool_result("web_search", "q1", "r1", 300)
        # 手动修改时间为过期 3 倍 TTL
        s.tool_cache[0].timestamp = time.time() - 1000
        s.cleanup_expired()
        assert len(s.tool_cache) == 0

    def test_cleanup_keeps_recently_expired(self):
        s = SessionSnapshot(chat_name="test")
        s.add_tool_result("web_search", "q1", "r1", 300)
        # 过期 1.5 倍 TTL（应保留）
        s.tool_cache[0].timestamp = time.time() - 450
        s.cleanup_expired()
        assert len(s.tool_cache) == 1

    def test_add_reply_and_get_recent(self):
        s = SessionSnapshot(chat_name="test")
        for i in range(12):
            s.add_reply(f"reply{i}")
        assert len(s.bot_replies) == 10  # 只保留最近 10 条
        recent = s.get_recent_replies(3)
        assert recent == ["reply9", "reply10", "reply11"]


class TestSessionMemory:
    def test_get_or_create(self):
        mem = SessionMemory()
        s = mem.get_or_create("群A")
        assert s.chat_name == "群A"
        s2 = mem.get_or_create("群A")
        assert s2 is s

    def test_add_tool_result(self):
        mem = SessionMemory()
        mem.add_tool_result("群A", "web_search", "上海天气", "晴")
        lines = mem.get_cache_lines("群A")
        assert len(lines) == 1
        assert "web_search" in lines[0]

    def test_add_tool_result_zero_ttl_not_cached(self):
        mem = SessionMemory()
        mem.add_tool_result("群A", "get_current_time", "", "10:00")
        lines = mem.get_cache_lines("群A")
        assert len(lines) == 0

    def test_add_reply(self):
        mem = SessionMemory()
        mem.add_reply("群A", "hello")
        s = mem.get_or_create("群A")
        assert s.bot_replies == ["hello"]

    def test_get_cache_lines_include_expired(self):
        mem = SessionMemory()
        mem.add_tool_result("群A", "web_search", "q", "r")
        # 手动过期
        mem._sessions["群A"].tool_cache[0].timestamp = time.time() - 400
        mem._sessions["群A"].tool_cache[0].ttl_seconds = 300
        lines = mem.get_cache_lines("群A", include_expired=True)
        assert len(lines) == 1
        assert "已过期" in lines[0]

    def test_cleanup_stale_sessions(self):
        mem = SessionMemory()
        mem.get_or_create("群A")
        mem._sessions["群A"].last_active = time.time() - 4000
        mem.cleanup_stale_sessions(max_idle_seconds=3600)
        assert "群A" not in mem._sessions

    def test_default_ttl_values(self):
        assert SessionMemory.DEFAULT_TTL["web_search"] == 300
        assert SessionMemory.DEFAULT_TTL["stock_query"] == 60
        assert SessionMemory.DEFAULT_TTL["get_weather"] == 1800
        assert SessionMemory.DEFAULT_TTL["get_current_time"] == 0


class TestExtractQueryKey:
    def test_web_search(self):
        assert _extract_query_key("web_search", '{"query": "上海"}') == "上海"

    def test_stock_query(self):
        assert _extract_query_key("stock_query", '{"stock_code": "600519"}') == "600519"

    def test_invalid_json_fallback(self):
        assert _extract_query_key("web_search", "not json") == "not json"

    def test_unknown_tool(self):
        assert _extract_query_key("custom_tool", '{"a": "1", "b": "2"}') == "1 2"
