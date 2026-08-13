#!/usr/bin/env python3
"""json_extractor 单元测试。"""

from src.utils.json_extractor import extract_json


class TestExtractJson:
    def test_ascii_quotes_inside_value(self):
        """值内含中文引号的合法 JSON → 解析成功，不被 normalize 破坏。

        回归 case：{"replies": ["他说"好的""]} 中中文引号在值中，
        第一次原样解析就应成功，不触发 normalize。
        """
        text = '{"replies": ["他说“好的”"]}'
        result = extract_json(text)
        assert result == {"replies": ["他说“好的”"]}

    def test_cjk_quotes_as_delimiters(self):
        """中文引号当 JSON 分隔符 → 标准化后解析成功。

        Case 4 场景：LLM 输出使用中文引号替代 ASCII 引号作为键/值分隔符。
        """
        text = '{"replies": ["警惕资本主义打牌"]}'
        result = extract_json(text)
        assert result == {"replies": ["警惕资本主义打牌"]}

    def test_markdown_code_block(self):
        """markdown 代码块包裹的 JSON → 解析成功。"""
        text = '```json\n{"replies": ["hello"]}\n```'
        result = extract_json(text)
        assert result == {"replies": ["hello"]}

    def test_invalid_text_returns_none(self):
        """非法文本 → 返回 None。"""
        assert extract_json("") is None
        assert extract_json("这不是 JSON") is None
        assert extract_json("{}") == {}  # 合法空对象
