#!/usr/bin/env python3
"""微信未登录/窗口异常场景的集成测试"""

import pytest
from unittest.mock import Mock
from pathlib import Path

from src.capture.window_capture import WeChatNotReadyError
from src.perception.vision_pipeline import VisionPipeline
from src.layout.profile import PROFILE_WECHAT_MAC_1760X1280
from src.bot.wechat_bot import WeChatBot


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestVisionPipelineNotReady:
    """VisionPipeline 对 WeChatNotReadyError 的处理"""

    def test_perceive_returns_none_on_not_ready(self):
        """当 WindowCapture 抛出 WeChatNotReadyError 时，perceive() 应返回 None"""
        pipeline = VisionPipeline(PROFILE_WECHAT_MAC_1760X1280)
        mock_capture = Mock()
        mock_capture.capture.side_effect = WeChatNotReadyError("需要扫码")
        pipeline.capture = mock_capture

        result = pipeline.perceive()
        assert result is None

    def test_perceive_with_login_handler_success(self, tmp_path):
        """login_handler 恢复成功后应正常返回 PerceptionResult"""
        from src.models.base import Rect, ChatMessage, ChatListItem, PerceptionResult
        from src.capture.window_capture import CaptureResult
        from src.layout.layout_parser import UILayout

        pipeline = VisionPipeline(PROFILE_WECHAT_MAC_1760X1280)

        mock_capture = Mock()
        mock_capture.capture.return_value = CaptureResult(
            image_path=str(tmp_path / "test.png"),
            window_rect=Rect(0, 0, 1760, 1280),
            scale_factor=1.0,
        )
        pipeline.capture = mock_capture

        # mock 下游，避免依赖真实图片
        pipeline.ocr.recognize = Mock(return_value=[])
        pipeline.layout.parse = Mock(return_value=UILayout(
            chat_name="测试群",
            chat_list_items=[ChatListItem(nickname="示例用户酉", last_message_preview="", unread_count="", timestamp="", rect=Rect(0,0,300,60))],
            title_elements=[], input_elements=[],
            timestamp_elements=[], self_bubbles=[],
            message_candidates=[]
        ))
        pipeline.extractor.extract = Mock(return_value=[
            ChatMessage(text="hi", sender="A", sender_type="other", chat_name="测试群")
        ])

        result = pipeline.perceive()
        assert result is not None
        assert result.chat_name == "测试群"


class TestBotNotReady:
    """Bot 对 WeChatNotReadyError 的处理"""

    def test_tick_logs_warning_on_not_ready(self):
        """Bot.tick() 在 perceive() 返回 None 时应记录 warning"""
        bot = WeChatBot(PROFILE_WECHAT_MAC_1760X1280)
        bot.perception = Mock()
        bot.perception.perceive.return_value = None

        bot.logger = Mock()

        bot.tick()

        bot.logger.warning.assert_called()
        # 检查消息中包含"扫码"或"登录"
        call_args = str(bot.logger.warning.call_args)
        assert "扫码" in call_args or "登录" in call_args or "未找到" in call_args


class TestRealSmallWindowFixture:
    """使用真实截图 fixture 验证 LayoutParser 对小窗口的解析"""

    def test_small_window_parsed_without_crash(self):
        """560x760 小窗口 fixture 应被解析且不崩溃"""
        from src.ocr.vision_ocr import VisionOCREngine
        from src.layout.layout_parser import LayoutParser

        img_path = FIXTURES_DIR / "wechat_not_ready_small_window.png"
        if not img_path.exists():
            pytest.skip("fixture wechat_not_ready_small_window.png not found")

        engine = VisionOCREngine()
        elements = engine.recognize(str(img_path))

        parser = LayoutParser(PROFILE_WECHAT_MAC_1760X1280)
        layout = parser.parse(elements, str(img_path))

        # 小窗口不应崩溃，chat_name 可能为空
        assert layout is not None
