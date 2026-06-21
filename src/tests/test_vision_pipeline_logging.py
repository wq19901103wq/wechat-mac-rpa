#!/usr/bin/env python3
"""VisionPipeline 异常日志测试"""

import logging
from unittest.mock import Mock

from src.capture.window_capture import WeChatNotReadyError
from src.layout.profile import PROFILE_WECHAT_MAC_1760X1280
from src.perception.vision_pipeline import VisionPipeline


class TestVisionPipelineLogging:
    def test_perceive_logs_warning_on_wechat_not_ready(self, caplog):
        """WeChatNotReadyError 时应记录包含'扫码'的 warning 日志"""
        caplog.set_level(logging.WARNING, logger="src.vision_pipeline")

        pipeline = VisionPipeline(PROFILE_WECHAT_MAC_1760X1280)
        mock_capture = Mock()
        mock_capture.capture.side_effect = WeChatNotReadyError("需要扫码登录")
        pipeline.capture = mock_capture

        result = pipeline.perceive()
        assert result is None
        assert any("扫码" in rec.message for rec in caplog.records)
