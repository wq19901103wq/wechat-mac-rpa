#!/usr/bin/env python3
"""L4 BotLogger 单元测试"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pytest

from src.logging.bot_logger import BotLogger


class TestBotLogger:
    """测试 BotLogger 的结构化日志输出"""

    @pytest.fixture
    def tmp_logs_dir(self, tmp_path):
        return tmp_path / "logs"

    @pytest.fixture
    def logger(self, tmp_logs_dir):
        bot_logger = BotLogger(logs_dir=str(tmp_logs_dir), max_bytes=1024, backup_count=2)
        yield bot_logger
        bot_logger.close()

    def test_init_creates_directories(self, tmp_logs_dir, logger):
        assert tmp_logs_dir.exists()
        expected_log = tmp_logs_dir / f"runtime_{datetime.now().strftime('%Y%m%d')}.log"
        assert expected_log.parent.exists()

    def test_standard_log_methods(self, tmp_logs_dir, logger, caplog):
        with caplog.at_level(logging.DEBUG, logger="src.runtime"):
            logger.debug("debug msg")
            logger.info("info msg")
            logger.warning("warning msg")
            logger.error("error msg")
            logger.critical("critical msg")

        assert "debug msg" in caplog.text
        assert "info msg" in caplog.text
        assert "warning msg" in caplog.text
        assert "error msg" in caplog.text
        assert "critical msg" in caplog.text

    def test_error_with_exc_info(self, tmp_logs_dir, logger, caplog):
        with caplog.at_level(logging.ERROR, logger="src.runtime"):
            try:
                raise ValueError("boom")
            except Exception:
                logger.error("something went wrong", exc_info=True)
        assert "something went wrong" in caplog.text
        assert "ValueError" in caplog.text

    def test_log_tick_start(self, tmp_logs_dir, logger):
        logger.log_tick_start(tick_id=42, interval=5.0)
        logger.close()

        records = _read_execution_jsonl(tmp_logs_dir)
        assert len(records) == 1
        assert records[0]["event"] == "tick_start"
        assert records[0]["tick_id"] == 42
        assert records[0]["interval"] == 5.0

    def test_log_capture_success(self, tmp_logs_dir, logger):
        logger.log_capture(tick_id=1, success=True, window_info={"x": 10, "y": 20, "width": 100, "height": 200})
        logger.close()

        records = _read_execution_jsonl(tmp_logs_dir)
        assert records[0]["event"] == "capture"
        assert records[0]["success"] is True
        assert records[0]["window"]["width"] == 100

    def test_log_capture_failure(self, tmp_logs_dir, logger):
        logger.log_capture(tick_id=1, success=False, error="window_not_found")
        logger.close()

        records = _read_execution_jsonl(tmp_logs_dir)
        assert records[0]["event"] == "capture"
        assert records[0]["success"] is False
        assert records[0]["error"] == "window_not_found"
        assert "window" in records[0]

    def test_log_ocr(self, tmp_logs_dir, logger):
        logger.log_ocr(tick_id=2, element_count=23, duration_ms=456.7, sample_texts=["群名", "昵称", "消息"])
        logger.close()

        records = _read_execution_jsonl(tmp_logs_dir)
        assert records[0]["event"] == "ocr"
        assert records[0]["element_count"] == 23
        assert records[0]["duration_ms"] == pytest.approx(456.7, abs=0.01)
        assert records[0]["sample_texts"] == ["群名", "昵称", "消息"]

    def test_log_layout(self, tmp_logs_dir, logger):
        logger.log_layout(
            tick_id=3,
            chat_name="测试群",
            title_elem_count=1,
            input_elem_count=2,
            timestamp_elem_count=3,
            self_bubble_count=2,
            message_candidate_count=8,
        )
        logger.close()

        records = _read_execution_jsonl(tmp_logs_dir)
        assert records[0]["event"] == "layout"
        assert records[0]["chat_name"] == "测试群"
        assert records[0]["title_elem_count"] == 1
        assert records[0]["input_elem_count"] == 2
        assert records[0]["timestamp_elem_count"] == 3
        assert records[0]["self_bubble_count"] == 2
        assert records[0]["message_candidate_count"] == 8

    def test_log_messages(self, tmp_logs_dir, logger):
        details = [
            {"text": "在吗", "sender": "示例用户酉", "sender_type": "other", "is_at_me": False},
            {"text": "在的", "sender": "Bot", "sender_type": "self", "is_at_me": False},
        ]
        logger.log_messages(tick_id=4, total_messages=6, new_messages=1, message_details=details)
        logger.close()

        records = _read_execution_jsonl(tmp_logs_dir)
        assert records[0]["event"] == "messages"
        assert records[0]["total_messages"] == 6
        assert records[0]["new_messages"] == 1
        assert len(records[0]["message_details"]) == 2

    def test_log_decision_with_reply(self, tmp_logs_dir, logger):
        logger.log_decision(
            tick_id=5,
            should_reply=True,
            reason="私聊新消息",
            latest_text="在吗",
            reply_text="在的，有什么可以帮你的？",
            extra={"cooldown_remaining": 0},
        )
        logger.close()

        records = _read_execution_jsonl(tmp_logs_dir)
        assert records[0]["event"] == "decision"
        assert records[0]["should_reply"] is True
        assert records[0]["reason"] == "私聊新消息"
        assert records[0]["latest_text"] == "在吗"
        assert records[0]["reply_text"] == "在的，有什么可以帮你的？"
        assert records[0]["extra"]["cooldown_remaining"] == 0

    def test_log_decision_without_reply(self, tmp_logs_dir, logger):
        logger.log_decision(
            tick_id=5,
            should_reply=False,
            reason="cooldown",
            latest_text="在吗",
        )
        logger.close()

        records = _read_execution_jsonl(tmp_logs_dir)
        assert records[0]["should_reply"] is False
        assert "reply_text" not in records[0] or records[0]["reply_text"] is None

    def test_log_send(self, tmp_logs_dir, logger):
        text = "在的，有什么可以帮你的？"
        logger.log_send(tick_id=6, success=True, text=text)
        logger.close()

        records = _read_execution_jsonl(tmp_logs_dir)
        assert records[0]["event"] == "send"
        assert records[0]["success"] is True
        assert records[0]["text_length"] == len(text)
        assert records[0]["text_preview"] == text[:200]
        assert records[0]["error"] is None

    def test_log_send_failure(self, tmp_logs_dir, logger):
        logger.log_send(tick_id=6, success=False, text="hello", error="applescript_failed")
        logger.close()

        records = _read_execution_jsonl(tmp_logs_dir)
        assert records[0]["success"] is False
        assert records[0]["error"] == "applescript_failed"

    def test_log_exception(self, tmp_logs_dir, logger):
        try:
            raise RuntimeError("vision failed")
        except Exception as exc:
            logger.log_exception(tick_id=7, phase="ocr", exc=exc)
        logger.close()

        records = _read_execution_jsonl(tmp_logs_dir)
        assert records[0]["event"] == "exception"
        assert records[0]["tick_id"] == 7
        assert records[0]["phase"] == "ocr"
        assert records[0]["exception_type"] == "RuntimeError"
        assert "vision failed" in records[0]["exception_msg"]
        assert "Traceback" in records[0]["traceback"]

    def test_log_stats(self, tmp_logs_dir, logger):
        logger.log_stats(tick_id=8, stats={"avg_ocr_ms": 123.4, "replies_today": 5})
        logger.close()

        records = _read_execution_jsonl(tmp_logs_dir)
        assert records[0]["event"] == "stats"
        assert records[0]["tick_id"] == 8
        assert records[0]["stats"]["avg_ocr_ms"] == pytest.approx(123.4, abs=0.01)

    def test_multiple_events_append(self, tmp_logs_dir, logger):
        logger.log_tick_start(tick_id=1, interval=5.0)
        logger.log_capture(tick_id=1, success=True, window_info={"x": 0, "y": 0, "width": 100, "height": 100})
        logger.close()

        records = _read_execution_jsonl(tmp_logs_dir)
        assert len(records) == 2
        assert records[0]["event"] == "tick_start"
        assert records[1]["event"] == "capture"

    def test_runtime_log_file_created(self, tmp_logs_dir, logger):
        logger.info("test runtime log")
        logger.close()

        log_files = list(tmp_logs_dir.glob("runtime_*.log"))
        assert len(log_files) == 1
        content = log_files[0].read_text(encoding="utf-8")
        assert "test runtime log" in content

    def test_runtime_log_switches_file_after_date_change(
        self, tmp_logs_dir, logger, monkeypatch
    ):
        monkeypatch.setattr("src.logging.bot_logger._date_stamp", lambda: "20990102")

        logger.info("next day")
        logger.close()

        assert (tmp_logs_dir / "runtime_20990102.log").exists()

    def test_execution_log_rotates(self, tmp_logs_dir):
        logger = BotLogger(
            logs_dir=str(tmp_logs_dir),
            execution_max_bytes=200,
            execution_backup_count=2,
        )
        for i in range(20):
            logger.log_tick_start(tick_id=i, interval=5.0)
        logger.close()

        assert (tmp_logs_dir / "execution.jsonl.1").exists()
        assert (tmp_logs_dir / "execution.jsonl").exists()

    def test_close_is_idempotent(self, logger):
        logger.close()
        logger.close()  # should not raise


def _read_execution_jsonl(logs_dir: Path) -> list:
    path = logs_dir / "execution.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if line:
            records.append(json.loads(line))
    return records
