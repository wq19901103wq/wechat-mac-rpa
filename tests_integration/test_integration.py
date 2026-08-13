#!/usr/bin/env python3
"""新架构集成测试

使用真实 fixture 验证 VisionPipeline -> Bot 的完整链路。
不依赖真实微信窗口，使用 mock 的 WindowCapture 注入 fixture 图片。
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perception.vision_pipeline import VisionPipeline
from src.layout.profile import PROFILE_WECHAT_MAC_1760X1280
from src.bot.wechat_bot import WeChatBot
from src.models.base import ChatMessage, SenderType


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture_json(name: str):
    path = FIXTURES_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


class TestVisionPipelineIntegration:
    """VisionPipeline 对真实 fixture 的端到端测试"""

    @pytest.fixture
    def pipeline(self):
        return VisionPipeline(PROFILE_WECHAT_MAC_1760X1280)

    def _mock_capture(self, pipeline, image_path: str):
        """将 WindowCapture mock 为返回指定图片"""
        from src.models.base import Rect
        from src.capture.window_capture import CaptureResult
        mock_capture = Mock()
        mock_capture.capture.return_value = CaptureResult(
            image_path=image_path,
            window_rect=Rect(0, 0, 1760, 1280),
            scale_factor=1.0
        )
        pipeline.capture = mock_capture

    def test_medium_scene(self, pipeline):
        """medium_scene fixture 应能解析出聊天名称和消息"""
        img_path = str(FIXTURES_DIR / "medium_scene.png")
        if not Path(img_path).exists():
            pytest.skip("medium_scene.png not found")

        self._mock_capture(pipeline, img_path)
        result = pipeline.perceive()

        assert result is not None
        assert result.chat_name != ""
        # 该 fixture 通常包含消息
        assert len(result.messages) >= 1

    def test_large_scene(self, pipeline):
        """large_scene fixture 应能解析出聊天列表和消息"""
        img_path = str(FIXTURES_DIR / "large_scene.png")
        if not Path(img_path).exists():
            pytest.skip("large_scene.png not found")

        self._mock_capture(pipeline, img_path)
        result = pipeline.perceive()

        assert result is not None
        assert result.chat_name != ""
        assert len(result.messages) >= 1

    def test_private_w1han(self, pipeline):
        """private_w1han fixture 已移除：包含真实私人聊天隐私数据"""
        pytest.skip("private_w1han fixture 已移除（隐私数据清理）")


class TestBotIntegration:
    """Bot 对 mock 感知结果的集成测试"""

    def test_bot_tick_with_new_message(self, tmp_path):
        bot = WeChatBot(PROFILE_WECHAT_MAC_1760X1280)

        msg = ChatMessage(
            text="在吗", sender="A",
            sender_type=SenderType.OTHER, chat_name="小王"
        )
        mock_result = Mock()
        mock_result.chat_name = "测试群"
        mock_result.messages = [msg]
        mock_result.chat_list_items = []
        mock_result.screenshot_path = str(tmp_path / "test.png")

        bot.perception = Mock()
        bot.perception.perceive.return_value = mock_result

        bot.generator = Mock()
        bot.generator.generate.return_value = ["在的"]

        bot.sender = Mock()
        bot.sender.send.return_value = Mock(success=True)

        bot.tick()

        bot.sender.send.assert_called_once_with("在的")

    def test_bot_tick_no_perception(self):
        bot = WeChatBot(PROFILE_WECHAT_MAC_1760X1280)
        bot.perception = Mock()
        bot.perception.perceive.return_value = None

        bot.sender = Mock()
        bot.tick()

        bot.sender.send.assert_not_called()
