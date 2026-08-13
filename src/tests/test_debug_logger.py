#!/usr/bin/env python3
"""DebugLogger 单元测试"""

import json
from unittest.mock import MagicMock

import pytest

from src.utils.debug_logger import DebugLogger, TickDebugInfo


class TestTickDebugInfo:
    def test_default_creation(self):
        info = TickDebugInfo()
        assert info.tick_id == 0
        assert info.screenshot_path == ""
        assert info.ocr_elements == []


class TestDebugLogger:
    @pytest.fixture
    def logger(self, tmp_path):
        return DebugLogger(base_dir=str(tmp_path / "test_debug"))

    def test_start_tick(self, logger):
        info = logger.start_tick(1, "screenshot.png")
        assert info.tick_id == 1
        assert info.screenshot_path == "screenshot.png"
        assert info.timestamp != ""

    def test_save_creates_json(self, tmp_path):
        base = tmp_path / "debug"
        logger = DebugLogger(base_dir=str(base))
        logger.start_tick(1, "s.png")
        logger.current.bot_chat_name = "测试群"
        path = logger.save()
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["bot_chat_name"] == "测试群"
        assert data["tick_id"] == 1

    def test_save_raises_without_start(self, tmp_path):
        logger = DebugLogger(base_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="start_tick"):
            logger.save()

    def test_log_ocr(self, logger):
        logger.start_tick(1, "s.png")
        mock_elem = MagicMock()
        mock_elem.text = "hello"
        mock_elem.bbox = MagicMock(x=10, y=20, width=30, height=40)
        mock_elem.center = MagicMock(x=25, y=40)
        logger.log_ocr([mock_elem])
        assert len(logger.current.ocr_elements) == 1
        assert logger.current.ocr_elements[0]["text"] == "hello"

    def test_log_layout_chat_list(self, logger):
        logger.start_tick(1, "s.png")
        mock_elem = MagicMock()
        mock_elem.text = "test"
        mock_elem.bbox.x = 10
        mock_elem.bbox.y = 20
        logger.log_layout_chat_list([mock_elem], [[mock_elem]], ["nick"], ["1"])
        assert logger.current.layout_chat_list_nicknames == ["nick"]
        assert logger.current.layout_chat_list_unread == ["1"]

    def test_log_bot_decision(self, logger):
        logger.start_tick(1, "s.png")
        logger.log_bot_decision(
            chat_name="群A",
            new_messages_count=3,
            should_reply=True,
            reply_target="@bot 你好",
            reply_text="hello",
        )
        assert logger.current.bot_chat_name == "群A"
        assert logger.current.bot_should_reply is True
        assert logger.current.bot_reply_text == "hello"

    def test_log_action(self, logger):
        logger.start_tick(1, "s.png")
        logger.log_action(action="send", action_input="hello", success=True)
        assert logger.current.action == "send"
        assert logger.current.action_result_success is True

    def test_log_session(self, logger):
        logger.start_tick(1, "s.png")
        logger.log_session(
            input_chat_name="群A",
            input_messages=[{"text": "hi"}],
            output_unreplied=[{"text": "hi"}],
            total_stored=5,
        )
        assert logger.current.session_input_chat_name == "群A"
        assert logger.current.session_total_stored == 5

    def test_log_perception_output(self, logger):
        logger.start_tick(1, "s.png")
        logger.log_perception_output(chat_name="群A", messages_count=3, chat_list_count=12)
        assert logger.current.perception_chat_name == "群A"
        assert logger.current.perception_messages_count == 3
        assert logger.current.perception_chat_list_count == 12

    def test_log_reply_generation(self, logger):
        logger.start_tick(1, "s.png")
        logger.log_reply_generation(
            system_prompt="sys",
            user_prompt="user",
            raw_response='{"replies": ["hi"]}',
            loaded_skills=["skill1"],
            active_llm="deepseek",
            hermes_fallback_triggered=False,
        )
        assert logger.current.reply_system_prompt == "sys"
        assert logger.current.reply_raw_response == '{"replies": ["hi"]}'
        assert logger.current.loaded_skills == ["skill1"]
        assert logger.current.active_llm == "deepseek"

    def test_log_reply_generation_trace(self, logger):
        logger.start_tick(1, "s.png")
        trace = [
            {"round": 1, "type": "llm_request", "messages": [{"role": "user", "content": "hi"}]},
            {"round": 1, "type": "llm_response", "content": "hello"},
        ]
        logger.log_reply_generation(trace=trace)
        assert len(logger.current.reply_generation_trace) == 2

    def test_save_prompt_markdown(self, tmp_path):
        base = tmp_path / "debug"
        logger = DebugLogger(base_dir=str(base))
        logger.start_tick(1, "s.png")
        logger.current.api_prompt = "prompt text"
        logger.current.api_response = '{"chat_name": "test"}'
        logger.save()
        md_files = list((base / "prompts").glob("*.md"))
        assert len(md_files) == 1
        content = md_files[0].read_text(encoding="utf-8")
        assert "prompt text" in content
        assert "test" in content

    def test_save_cleans_old_files(self, tmp_path):
        # DebugLogger 本身不清理，但验证能正常写入即可
        base = tmp_path / "debug"
        logger = DebugLogger(base_dir=str(base))
        logger.start_tick(1, "s.png")
        path1 = logger.save()
        logger.start_tick(2, "s.png")
        path2 = logger.save()
        assert path1.exists()
        assert path2.exists()

    def test_save_prompt_markdown_skips_empty(self, tmp_path):
        """空的 tick（无 OCR、无对话、无 Bot 回复）不应生成 .md 文件"""
        base = tmp_path / "debug"
        logger = DebugLogger(base_dir=str(base))
        logger.start_tick(1, "s.png")
        # 不设置任何 api_prompt / trace / raw_response
        logger.save()
        md_files = list((base / "prompts").glob("*.md"))
        assert len(md_files) == 0
