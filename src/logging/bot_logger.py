#!/usr/bin/env python3
"""
Bot Logger - 结构化运行时日志系统

提供两级日志输出：
1. 人类可读的运行日志 (runtime_YYYYMMDD.log) - 带 emoji 分级
2. 机器可解析的执行流水 (execution.jsonl) - 每行一个 JSON，记录完整决策链路

AI 使用指南：
- 排查逻辑问题时，优先看 execution.jsonl（结构化、可 grep）
- 排查异常报错时，优先看 runtime_YYYYMMDD.log（带堆栈）
- 新增日志时，尽量使用便捷方法：log_tick_start / log_ocr / log_layout / log_decision / log_send
"""


import json
import logging
import threading
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional


def _date_stamp() -> str:
    return datetime.now().strftime("%Y%m%d")


class _DailyRotatingFileHandler(RotatingFileHandler):
    """按日期切换文件，并在单日内继续按大小轮转。"""

    def __init__(self, logs_dir: Path, max_bytes: int, backup_count: int):
        self._logs_dir = logs_dir
        self._current_date = _date_stamp()
        super().__init__(
            logs_dir / f"runtime_{self._current_date}.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )

    def emit(self, record: logging.LogRecord) -> None:
        current_date = _date_stamp()
        if current_date != self._current_date:
            if self.stream:
                self.stream.close()
                self.stream = None  # type: ignore[assignment]
            self._current_date = current_date
            self.baseFilename = str(
                (self._logs_dir / f"runtime_{current_date}.log").resolve()
            )
        super().emit(record)


class _EmojiFormatter(logging.Formatter):
    """带 emoji 的控制台格式器"""
    LEVEL_EMOJI = {
        "DEBUG": "🔍",
        "INFO": "ℹ️ ",
        "WARNING": "⚠️ ",
        "ERROR": "❌",
        "CRITICAL": "🚨",
    }

    def format(self, record: logging.LogRecord) -> str:
        emoji = self.LEVEL_EMOJI.get(record.levelname, "📝")
        record.emoji = emoji
        return f"[{self.formatTime(record)}] {emoji} [{record.levelname}] {record.getMessage()}"


class BotLogger:
    """
    微信 RPA 专用日志器。

    Attributes:
        logs_dir: 日志根目录
        runtime_logger: 人类可读的运行日志
        execution_fp: execution.jsonl 文件句柄（逐行 JSON）
    """

    def __init__(
        self,
        logs_dir: Optional[str] = None,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 3,
        execution_max_bytes: int = 50 * 1024 * 1024,
        execution_backup_count: int = 3,
    ):
        if logs_dir is None:
            logs_dir = str(Path(__file__).parent.parent.parent / "data" / "logs")
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        file_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s"
        )

        # --- 1. 人类可读的运行日志 ---
        self.runtime_logger = logging.getLogger("src.runtime")
        self.runtime_logger.setLevel(logging.DEBUG)
        # 避免重复添加 handler（重复实例化时）
        self.runtime_logger.handlers = []

        # Console handler（带 emoji，仅 src.runtime）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(_EmojiFormatter())
        self.runtime_logger.addHandler(console_handler)

        # --- 2. 根 logger 统一文件输出（捕获所有 src.* 子模块）---
        src_root = logging.getLogger("src")
        src_root.setLevel(logging.DEBUG)
        src_root.handlers = []
        file_handler = _DailyRotatingFileHandler(
            self.logs_dir,
            max_bytes=max_bytes,
            backup_count=backup_count,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_fmt)
        src_root.addHandler(file_handler)

        # --- 2. 机器可解析的执行流水 ---
        self.execution_path = self.logs_dir / "execution.jsonl"
        self._execution_handler = RotatingFileHandler(
            self.execution_path,
            maxBytes=execution_max_bytes,
            backupCount=execution_backup_count,
            encoding="utf-8",
        )
        self._execution_handler.setFormatter(logging.Formatter("%(message)s"))

        self.runtime_logger.info(f"BotLogger 初始化完成 | logs_dir={self.logs_dir}")

    def close(self):
        """关闭 execution.jsonl handler。"""
        if self._execution_handler:
            self._execution_handler.close()

    # ------------------------------------------------------------------
    # 便捷方法：运行日志
    # ------------------------------------------------------------------
    def debug(self, msg: str, *args, **kwargs):
        self.runtime_logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self.runtime_logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self.runtime_logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, exc_info=False, **kwargs):
        self.runtime_logger.error(msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self.runtime_logger.critical(msg, *args, **kwargs)

    # ------------------------------------------------------------------
    # 便捷方法：执行流水（写入 execution.jsonl）
    # ------------------------------------------------------------------
    def _append_execution(self, event_type: str, payload: Dict[str, Any]):
        """追加一条结构化执行记录"""
        record = {
            "ts": datetime.now().isoformat(),
            "event": event_type,
            **payload,
        }
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            log_record = logging.LogRecord(
                name="src.execution",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=line,
                args=(),
                exc_info=None,
            )
            self._execution_handler.handle(log_record)
        except Exception as e:
            self.runtime_logger.error(f"写入 execution.jsonl 失败: {e}")

    def log_tick_start(self, tick_id: int, interval: float):
        self.info(f"🔄 Tick #{tick_id} 开始 (interval={interval}s)")
        self._append_execution("tick_start", {"tick_id": tick_id, "interval": interval})

    def log_capture(self, tick_id: int, success: bool, window_info: Optional[Dict] = None, error: Optional[str] = None):
        if success:
            self.debug(f"📸 截图成功: {window_info}")
        else:
            self.warning(f"❌ 截图失败: {error}")
        self._append_execution("capture", {
            "tick_id": tick_id,
            "success": success,
            "window": window_info,
            "error": error,
        })

    def log_ocr(self, tick_id: int, element_count: int, duration_ms: float, sample_texts: List[str]):
        self.debug(f"🔍 OCR 完成: {element_count} 个元素, 耗时 {duration_ms:.1f}ms, 样例: {sample_texts[:5]}")
        self._append_execution("ocr", {
            "tick_id": tick_id,
            "element_count": element_count,
            "duration_ms": round(duration_ms, 2),
            "sample_texts": sample_texts[:10],
        })

    def log_layout(self, tick_id: int, chat_name: str, title_elem_count: int,
                   input_elem_count: int, timestamp_elem_count: int,
                   self_bubble_count: int, message_candidate_count: int):
        self.debug(
            f"📐 Layout: chat={chat_name}, title={title_elem_count}, "
            f"input={input_elem_count}, timestamps={timestamp_elem_count}, "
            f"self_bubbles={self_bubble_count}, candidates={message_candidate_count}"
        )
        self._append_execution("layout", {
            "tick_id": tick_id,
            "chat_name": chat_name,
            "title_elem_count": title_elem_count,
            "input_elem_count": input_elem_count,
            "timestamp_elem_count": timestamp_elem_count,
            "self_bubble_count": self_bubble_count,
            "message_candidate_count": message_candidate_count,
        })

    def log_messages(self, tick_id: int, total_messages: int, new_messages: int,
                     message_details: List[Dict[str, Any]]):
        self.info(f"💬 消息统计: 共 {total_messages} 条, 新增 {new_messages} 条")
        for m in message_details[-3:]:
            icon = "🤖" if m.get("sender_type") == "self" else "👤"
            self.debug(f"   {icon} [{m.get('sender')}] {m.get('text', '')[:50]}")
        self._append_execution("messages", {
            "tick_id": tick_id,
            "total_messages": total_messages,
            "new_messages": new_messages,
            "message_details": message_details,
        })

    def log_decision(self, tick_id: int, should_reply: bool, reason: str,
                     latest_text: str, reply_text: Optional[str] = None,
                     extra: Optional[Dict] = None):
        if should_reply:
            self.info(f"✅ 决策: 需要回复 | 原因={reason} | 回复={reply_text}")
        else:
            self.info(f"⏭️  决策: 跳过回复 | 原因={reason} | 消息={latest_text[:50]}")
        payload = {
            "tick_id": tick_id,
            "should_reply": should_reply,
            "reason": reason,
            "latest_text": latest_text,
            "reply_text": reply_text,
        }
        if extra:
            payload["extra"] = extra
        self._append_execution("decision", payload)

    def log_send(self, tick_id: int, success: bool, text: str, error: Optional[str] = None):
        text_str = str(text) if text is not None else ""
        if success:
            self.info(f"📤 发送成功 ({len(text_str)}字): {text_str[:60]}")
        else:
            self.error(f"❌ 发送失败: {error}")
        self._append_execution("send", {
            "tick_id": tick_id,
            "success": success,
            "text_length": len(text_str),
            "text_preview": text_str[:200],
            "error": error,
        })

    def log_exception(self, tick_id: int, phase: str, exc: Exception):
        tb = traceback.format_exc()
        self.error(f"💥 Tick #{tick_id} 在 [{phase}] 阶段异常: {exc}\n{tb}", exc_info=True)
        self._append_execution("exception", {
            "tick_id": tick_id,
            "phase": phase,
            "exception_type": type(exc).__name__,
            "exception_msg": str(exc),
            "traceback": tb,
        })

    def log_stats(self, tick_id: int, stats: Dict[str, Any]):
        self.info(f"📊 统计: {stats}")
        self._append_execution("stats", {"tick_id": tick_id, "stats": stats})


# 全局单例（方便 import 即用）
_default_logger: Optional[BotLogger] = None
_default_logger_lock = threading.Lock()


def get_logger(logs_dir: Optional[str] = None) -> BotLogger:
    global _default_logger
    if _default_logger is None:
        with _default_logger_lock:
            if _default_logger is None:
                _default_logger = BotLogger(logs_dir=logs_dir)
    return _default_logger
