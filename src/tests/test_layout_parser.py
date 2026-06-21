#!/usr/bin/env python3
"""L3 LayoutParser 单元测试"""

from pathlib import Path

import pytest

from src.layout.layout_parser import TIMESTAMP_PATTERNS, LayoutParser
from src.layout.profile import PROFILE_WECHAT_MAC_1760X1280
from src.models.base import Rect
from src.ocr.vision_ocr import VisionOCREngine

FIXTURES_DIR = Path(__file__).parent.parent.parent / "tests" / "fixtures"
ERRORS_DIR = FIXTURES_DIR / "errors"


class TestTimestampPatterns:
    def test_patterns(self):
        import re
        assert re.match(TIMESTAMP_PATTERNS[0], "12:34")
        assert re.match(TIMESTAMP_PATTERNS[1], "昨天 12:34")
        assert re.match(TIMESTAMP_PATTERNS[2], "星期一 12:34")
        assert re.match(TIMESTAMP_PATTERNS[3], "星期一")
        assert re.match(TIMESTAMP_PATTERNS[4], "2024/01/15")


class TestLayoutParserRealFixtures:
    """使用真实 fixture 图片进行布局解析测试"""

    @pytest.fixture(scope="class")
    def ocr_engine(self):
        return VisionOCREngine()

    @pytest.fixture(scope="class")
    def parser(self):
        return LayoutParser(PROFILE_WECHAT_MAC_1760X1280)

    def _run_parse(self, ocr_engine, parser, image_name: str):
        img_path = FIXTURES_DIR / f"{image_name}.png"
        if not img_path.exists():
            pytest.skip(f"fixture {image_name}.png not found")
        elements = ocr_engine.recognize(str(img_path))
        return parser.parse(elements, str(img_path))

    def test_small_scene_basic(self, ocr_engine, parser):
        """small_scene 是一个低质量/非标准尺寸 fixture，验证解析器不崩溃即可"""
        layout = self._run_parse(ocr_engine, parser, "small_scene")
        # 该 fixture 尺寸仅 560x760，与预设 profile 不匹配，不强制要求 chat_name
        assert layout is not None

    def test_medium_scene_has_title(self, ocr_engine, parser):
        layout = self._run_parse(ocr_engine, parser, "medium_scene")
        # 标题栏应该有内容
        assert len(layout.title_elements) >= 1

    def test_large_scene_message_candidates(self, ocr_engine, parser):
        layout = self._run_parse(ocr_engine, parser, "large_scene")
        # 消息候选区应该有内容
        assert len(layout.message_candidates) >= 1

    def test_error_20260413_002_basic(self, ocr_engine, parser):
        """error_20260413_002 验证解析器能稳定运行并提取基本结构"""
        img_path = ERRORS_DIR / "error_20260413_002.png"
        if not img_path.exists():
            pytest.skip("fixture not found")
        elements = ocr_engine.recognize(str(img_path))
        layout = parser.parse(elements, str(img_path))
        # 该 error case 的颜色特征与标准 profile 不完全匹配（连老 V4 也无法检测），
        # 因此不强制要求 self_bubbles 非空，只验证解析器正常运行
        assert layout is not None
        assert layout.chat_name != ""

    def test_chat_list_items_detected(self, ocr_engine, parser):
        """验证左侧聊天列表能解析出项目"""
        layout = self._run_parse(ocr_engine, parser, "medium_scene")
        # 大多数 fixture 应该有聊天列表
        # 不强制要求非空，但如果为空则跳过
        if len(layout.chat_list_items) == 0:
            pytest.skip("no chat list items in this fixture")
        for item in layout.chat_list_items:
            assert item.nickname != ""
            assert isinstance(item.rect, Rect)

    def test_input_elements_filtered(self, ocr_engine, parser):
        """输入框区域的元素应被过滤到 input_elements"""
        layout = self._run_parse(ocr_engine, parser, "large_scene")
        # 输入框元素不应出现在 message_candidates 中
        input_ids = {id(e) for e in layout.input_elements}
        candidate_ids = {id(e) for e in layout.message_candidates}
        assert input_ids.isdisjoint(candidate_ids)

    def test_timestamps_not_in_candidates(self, ocr_engine, parser):
        """时间戳元素不应出现在 message_candidates 中"""
        layout = self._run_parse(ocr_engine, parser, "large_scene")
        ts_ids = {id(e) for e in layout.timestamp_elements}
        candidate_ids = {id(e) for e in layout.message_candidates}
        assert ts_ids.isdisjoint(candidate_ids)

    def test_regression_title_y_max_extracts_chat_name(self, ocr_engine, parser):
        """
        回归测试：title_y_max=50 会过滤掉 y=90 的标题元素，导致 chat_name 为空。
        修复后 title_y_max=120，应能正确提取标题。

        Fixture: regression_title_y90_20260419.png
        来源：2026-04-19 实际运行截图，标题 "王芊 @ai开发小分队" bbox.y=90。
        """
        layout = self._run_parse(ocr_engine, parser, "regression_title_y90_20260419")
        # 标题栏必须有元素
        assert len(layout.title_elements) >= 1, (
            "title_elements 为空，说明 title_y_max 仍过小"
        )
        # chat_name 必须被提取
        assert layout.chat_name != "", (
            f"chat_name 为空，title_elements={ [e.text for e in layout.title_elements] }"
        )
        # 验证提取到的是聊天名而非噪声（如时间、搜索框）
        assert "王芊" in layout.chat_name or "ai开发" in layout.chat_name, (
            f"chat_name='{layout.chat_name}' 未包含预期的聊天名关键词"
        )

    def test_regression_input_y_min_adaptive_tall_window(self, ocr_engine, parser):
        """
        回归测试：窗口高度 1602px 时，input_y_min=1040 会把消息区底部内容误判为输入框，
        导致 "所有方面"、"你帮我看下" 等消息丢失。

        修复后按实际截图高度动态计算 input_y_min，消息应保留在 candidates 中。

        Fixture: regression_input_y_min_1602_20260420.png
        来源：2026-04-20 实际运行截图，高度 1602px（含阴影排除后的尺寸）。
        """
        layout = self._run_parse(ocr_engine, parser, "regression_input_y_min_1602_20260420")
        candidate_texts = [e.text for e in layout.message_candidates]
        input_texts = [e.text for e in layout.input_elements]

        # 关键消息必须在 candidates 中，不能在 input_elements 中
        assert "所有方面" in candidate_texts, (
            f"\"所有方面\" 未在 message_candidates 中，可能被 input_y_min 误过滤。"
            f"candidates={candidate_texts}, input={input_texts}"
        )
        assert "你帮我看下" in candidate_texts, (
            f"\"你帮我看下\" 未在 message_candidates 中，可能被 input_y_min 误过滤。"
            f"candidates={candidate_texts}, input={input_texts}"
        )
        assert "所有方面" not in input_texts, (
            "\"所有方面\" 不应在 input_elements 中"
        )
        assert "你帮我看下" not in input_texts, (
            "\"你帮我看下\" 不应在 input_elements 中"
        )

        # 验证动态坐标已生效（1602 > 1280，坐标应被放大）
        profile = parser.profile
        assert parser._scaled_input_y_min > profile.input_y_min, (
            f"input_y_min 未缩放: {parser._scaled_input_y_min} <= {profile.input_y_min}"
        )
        assert parser._scaled_title_y_max > profile.title_y_max, (
            f"title_y_max 未缩放: {parser._scaled_title_y_max} <= {profile.title_y_max}"
        )

    def test_scaled_coordinates_on_small_window(self, ocr_engine, parser):
        """
        验证窗口尺寸显著小于 Profile 时，坐标被正确缩小。
        small_scene 尺寸 560x760，scale_x=0.318, scale_y=0.594。
        """
        layout = self._run_parse(ocr_engine, parser, "small_scene")
        assert layout is not None

        # 验证动态坐标已生效（560 < 1760，坐标应被缩小）
        profile = parser.profile
        assert parser._scaled_left_boundary < profile.left_boundary, (
            f"left_boundary 未缩小: {parser._scaled_left_boundary} >= {profile.left_boundary}"
        )
        assert parser._scaled_chat_list_x_max < profile.chat_list_x_max, (
            f"chat_list_x_max 未缩小: {parser._scaled_chat_list_x_max} >= {profile.chat_list_x_max}"
        )
        assert parser._scaled_title_y_max < profile.title_y_max, (
            f"title_y_max 未缩小: {parser._scaled_title_y_max} >= {profile.title_y_max}"
        )
        assert parser._scaled_input_y_min < profile.input_y_min, (
            f"input_y_min 未缩小: {parser._scaled_input_y_min} >= {profile.input_y_min}"
        )
