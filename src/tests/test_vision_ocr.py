#!/usr/bin/env python3
"""VisionOCREngine 单元测试"""

import os

import pytest

from src.models.base import OCRTextElement, Point, Rect
from src.ocr.vision_ocr import VisionOCREngine

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "fixtures")
SMALL_SCENE_PATH = os.path.join(FIXTURES_DIR, "small_scene.png")


class TestVisionOCREngine:
    def test_recognize_real_fixture(self):
        """使用真实 fixture 图片测试 OCR 识别"""
        engine = VisionOCREngine()
        results = engine.recognize(SMALL_SCENE_PATH)

        # 必须返回非空列表
        assert isinstance(results, list)
        assert len(results) > 0

        # 每个结果必须是 OCRTextElement
        for elem in results:
            assert isinstance(elem, OCRTextElement)
            assert isinstance(elem.text, str)
            assert len(elem.text) > 0
            assert isinstance(elem.bbox, Rect)
            assert isinstance(elem.center, Point)
            assert elem.bbox.width >= 0
            assert elem.bbox.height >= 0
            assert 0.0 <= elem.confidence <= 1.0

        # 必须按 center.y 升序排列
        y_coords = [elem.center.y for elem in results]
        assert y_coords == sorted(y_coords)

    def test_recognize_nonexistent_path(self):
        """测试传入不存在的图片路径时抛出 FileNotFoundError"""
        engine = VisionOCREngine()
        with pytest.raises(FileNotFoundError):
            engine.recognize("/this/path/does/not/exist.png")
