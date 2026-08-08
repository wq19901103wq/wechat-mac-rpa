#!/usr/bin/env python3
"""L3.5 Vision Pipeline - 感知管道

将 Capture → OCR → Layout → Extract 的完整视觉链路封装为单一接口。
对 Bot 层完全隐藏视觉实现细节。
"""

import logging
from typing import Optional

from src.capture.window_capture import WeChatNotReadyError, WindowCapture
from src.layout.layout_parser import LayoutParser
from src.layout.profile import LayoutProfile
from src.message.extractor import MessageExtractor
from src.models.base import PerceptionResult
from src.ocr.vision_ocr import VisionOCREngine
from src.utils.debug_logger import DebugLogger

_logger = logging.getLogger("src.vision_pipeline")


class VisionPipeline:
    def __init__(self, profile: LayoutProfile):
        self.capture = WindowCapture()
        self.ocr = VisionOCREngine()
        self.layout = LayoutParser(profile)
        self.extractor = MessageExtractor(profile)
        self.debug_logger = DebugLogger()

    def perceive(self) -> Optional[PerceptionResult]:
        """
        执行完整视觉链路：截图 → OCR → 布局分组 → 消息提取。

        Returns:
            PerceptionResult: 包含结构化消息列表、聊天名、截图路径
            None: 当 Capture 失败（如未找到窗口）时返回 None，由 Bot 层跳过本轮
        """
        try:
            capture_result = self.capture.capture()
        except WeChatNotReadyError as e:
            msg = str(e)
            if "手机上确认" in msg:
                _logger.warning(msg)
            elif "扫码" in msg:
                _logger.warning(msg)
            else:
                _logger.warning(
                    "未能获取微信窗口画面，可能原因：微信未启动、窗口被最小化、或需要扫码登录"
                )
            raise
        except Exception:
            _logger.warning(
                "未能获取微信窗口画面，可能原因：微信未启动、窗口被最小化、或需要扫码登录"
            )
            return None

        image_path = capture_result.image_path
        elements = self.ocr.recognize(image_path)
        layout = self.layout.parse(elements, image_path)
        messages = self.extractor.extract(layout)

        # 收集调试信息
        debug = self.debug_logger.start_tick(0, image_path)
        self.debug_logger.log_ocr(elements)
        left_elements = [e for e in elements if e.bbox.x < self.layout.profile.left_boundary]
        self.debug_logger.log_layout_chat_list(
            left_elements=left_elements,
            groups=self.layout.debug_info.get("chat_list", {}).get("groups", []),
            nicknames=[i.nickname for i in layout.chat_list_items],
            unread=[i.unread_count for i in layout.chat_list_items],
        )
        self.debug_logger.log_layout_message_area(
            candidates=layout.message_candidates,
            self_bubbles=layout.self_bubbles,
        )
        # debug_info 中的元素已是 dict 格式，直接传递
        self.debug_logger.log_layout_full(
            right_elements=self.layout.debug_info.get("right_elements", []),
            title_elements=self.layout.debug_info.get("title_elements", []),
            input_elements=self.layout.debug_info.get("input_elements", []),
            timestamp_elements=self.layout.debug_info.get("timestamp_elements", []),
        )
        self.debug_logger.log_extraction(
            clusters=self.extractor.debug_info.get("clusters", []),
            messages=messages,
        )

        return PerceptionResult(
            chat_name=layout.chat_name,
            messages=messages,
            chat_list_items=layout.chat_list_items,
            screenshot_path=image_path,
            window_rect=capture_result.window_rect,
            scale_factor=capture_result.scale_factor,
            debug_info=debug.__dict__,
            is_service_account_list=layout.is_service_account_list,
        )
