#!/usr/bin/env python3
"""Tests for src/utils/text_utils."""

from src.utils.text_utils import _compress_text, _truncate_text


class TestTruncateText:
    def test_short_text_unchanged(self):
        """短文本不应被截断。"""
        assert _truncate_text("hello", 100) == "hello"

    def test_exact_length_unchanged(self):
        """长度刚好等于阈值时不截断。"""
        text = "a" * 100
        assert _truncate_text(text, 100) == text

    def test_long_text_truncated(self):
        """超长文本被截断并追加 suffix。"""
        text = "a" * 200
        result = _truncate_text(text, 100)
        assert result.startswith("a" * 100)
        assert "truncated" in result

    def test_custom_suffix(self):
        """自定义 suffix 生效。"""
        text = "a" * 200
        result = _truncate_text(text, 100, suffix="[END]")
        assert result == "a" * 100 + "[END]"

    def test_empty_string(self):
        """空字符串返回空。"""
        assert _truncate_text("", 100) == ""


class TestCompressText:
    def test_short_text_unchanged(self):
        """短文本不应被压缩。"""
        assert _compress_text("hello", 100) == "hello"

    def test_compresses_long_text(self):
        """长文本保留头尾，中间省略。"""
        text = "a" * 50 + "b" * 50 + "c" * 50
        result = _compress_text(text, 100)
        assert result.startswith("a" * 20)  # 40% head
        assert result.endswith("c" * 20)   # 40% tail
        assert "中间省略" in result

    def test_empty_string(self):
        """空字符串返回空。"""
        assert _compress_text("", 100) == ""
