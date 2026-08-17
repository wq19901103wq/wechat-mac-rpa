#!/usr/bin/env python3
"""
L2 OCR 模块 - VisionOCREngine

使用 macOS Vision 框架进行文字识别。
纯 OCR 提取，不做任何业务过滤或布局判断。
"""

import logging
import os
from typing import List

from PIL import Image

from src.models.base import OCRTextElement, Point, Rect

_logger = logging.getLogger("src.vision_ocr")


class VisionOCREngine:
    """基于 macOS Vision 框架的 OCR 引擎"""

    def __init__(self, language: str = "zh-Hans"):
        self.language = language
        self._last_image_width: int = 0
        self._last_image_height: int = 0

    def recognize(self, image_path: str) -> List[OCRTextElement]:
        """
        识别图片中的所有文本。

        Args:
            image_path: 图片文件路径

        Returns:
            OCRTextElement 列表，按 center.y 升序排列

        Raises:
            FileNotFoundError: 图片路径不存在
        """
        import sys

        if sys.platform == "win32":
            return self._recognize_windows(image_path)

        import time

        # macOS：Vision 框架（lazy import，避免 Windows 上 import 失败）
        import Quartz
        import Vision
        from Foundation import NSArray, NSURL

        t0 = time.time()
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 获取图片尺寸用于坐标转换
        img = Image.open(image_path).convert("RGB")
        image_width, image_height = img.size
        self._last_image_width = image_width
        self._last_image_height = image_height

        # 创建 Vision 请求
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLanguages_(NSArray.arrayWithObject_(self.language))
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)

        # 加载图片
        image_url = NSURL.fileURLWithPath_(image_path)
        image_source = Quartz.CGImageSourceCreateWithURL(image_url, None)
        if image_source is None:
            _logger.warning(f"无法从 URL 创建图片源: {image_path}")
            return []

        cg_image = Quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)
        if cg_image is None:
            _logger.warning(f"无法从图片源创建 CGImage: {image_path}")
            return []

        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, None
        )

        success, error = handler.performRequests_error_([request], None)
        if not success:
            _logger.warning(f"Vision OCR 请求失败: {error}")
            return []

        elements: List[OCRTextElement] = []
        for observation in request.results():
            text = str(observation.text())
            confidence = float(observation.confidence())
            bbox = observation.boundingBox()

            # Vision 使用左下角原点、归一化坐标；转换为左上角原点像素坐标
            vx = float(bbox.origin.x)
            vy = float(bbox.origin.y)
            vw = float(bbox.size.width)
            vh = float(bbox.size.height)

            x = int(vx * image_width)
            y = int((1.0 - vy - vh) * image_height)
            width = int(vw * image_width)
            height = int(vh * image_height)
            cx = int((vx + vw / 2) * image_width)
            cy = int((1.0 - vy - vh / 2) * image_height)

            elements.append(
                OCRElement(
                    text=text.strip(),
                    bbox=Rect(x=x, y=y, width=width, height=height),
                    center=Point(x=cx, y=cy),
                    confidence=confidence,
                )
            )

        # 按 center.y 升序排列（从上到下）
        elements.sort(key=lambda e: e.center.y)
        t_ms = (time.time() - t0) * 1000
        _logger.info(f"[Perf][OCR] recognize: {t_ms:.0f}ms, elements={len(elements)}")
        return elements

    def _recognize_windows(self, image_path: str) -> List[OCRTextElement]:
        """Windows 兜底 OCR：优先 pytesseract（需安装 Tesseract + chi_sim 语言包）。

        返回空列表表示 OCR 不可用（感知层可降级），不抛异常。
        """
        try:
            import pytesseract
            from PIL import Image as PILImage
        except ImportError:
            _logger.warning("Windows OCR 需要 pytesseract，未安装")
            return []
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = PILImage.open(image_path).convert("RGB")
        self._last_image_width, self._last_image_height = img.size
        try:
            data = pytesseract.image_to_data(
                img, lang="chi_sim+eng", output_type=pytesseract.Output.DICT
            )
        except Exception as e:
            _logger.warning("pytesseract OCR 失败（可能未安装 Tesseract 或语言包）: %s", e)
            return []

        # 按 block/par/line 合并词为文本行
        lines: dict[tuple, dict] = {}
        for i in range(len(data["text"])):
            if not data["text"][i].strip():
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            entry = lines.setdefault(
                key,
                {"texts": [], "conf": [], "left": [], "top": [], "width": [], "height": []},
            )
            entry["texts"].append(data["text"][i])
            entry["conf"].append(data["conf"][i])
            entry["left"].append(data["left"][i])
            entry["top"].append(data["top"][i])
            entry["width"].append(data["width"][i])
            entry["height"].append(data["height"][i])

        elements: List[OCRTextElement] = []
        for entry in lines.values():
            text = "".join(entry["texts"]).strip()
            if not text:
                continue
            x = min(entry["left"])
            y = min(entry["top"])
            width = max(lt + wd for lt, wd in zip(entry["left"], entry["width"])) - x
            height = max(t + ht for t, ht in zip(entry["top"], entry["height"])) - y
            valid_conf = [c for c in entry["conf"] if c >= 0]
            confidence = sum(valid_conf) / max(len(valid_conf), 1) / 100.0
            elements.append(
                OCRElement(
                    text=text,
                    bbox=Rect(x=x, y=y, width=width, height=height),
                    center=Point(x=x + width // 2, y=y + height // 2),
                    confidence=confidence,
                )
            )
        elements.sort(key=lambda el: el.center.y)
        _logger.info("Windows OCR(pytesseract) 识别 %d 个元素", len(elements))
        return elements

    @property
    def image_width(self) -> int:
        return getattr(self, "_last_image_width", 0)

    @property
    def image_height(self) -> int:
        return getattr(self, "_last_image_height", 0)


class OCRElement(OCRTextElement):
    """Backward compatibility wrapper for old OCR element API."""

    @property
    def x(self) -> int:
        return self.bbox.x

    @property
    def y(self) -> int:
        return self.bbox.y

    @property
    def cx(self) -> int:
        return self.center.x

    @property
    def cy(self) -> int:
        return self.center.y

    @property
    def width(self) -> int:
        return self.bbox.width

    @property
    def height(self) -> int:
        return self.bbox.height

    @property
    def normalized_x(self) -> float:
        """归一化 x 坐标 (0-1)"""
        return self.center.x / 1760  # 假设标准宽度

    @property
    def normalized_y(self) -> float:
        """归一化 y 坐标 (0-1)"""
        return self.center.y / 1280  # 假设标准高度


# Backward compatibility alias
VisionOCR = VisionOCREngine
