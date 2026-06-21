#!/usr/bin/env python3
"""
Case Database — SQLite 存储所有 badcase 的完整信息（原始 prompt、对话、评分、工具调用）

替代原有的 JSON 文件散落存储，提供统一查询和趋势分析。

用法:
    from src.badcase.case_db import CaseDB
    db = CaseDB()
    db.insert_case(draft_dict)         # 入库一个 case
    cases = db.query_recent(days=7)    # 查询最近 7 天的 case
    db.export_daily_metrics(date)      # 导出某天的指标快照
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

_logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "cases.db"


# =============================================================================
# Schema
# =============================================================================

SCHEMA_SQL = """
-- wechat-twin v1.1 schema

CREATE TABLE IF NOT EXISTS tick_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL DEFAULT '',
    tick_id INTEGER NOT NULL,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    skip_reason TEXT,
    chat_name TEXT, is_group INTEGER DEFAULT 0, screenshot_path TEXT,
    messages_count INTEGER, new_messages_count INTEGER DEFAULT 0,
    system_prompt TEXT, user_prompt TEXT, raw_response TEXT, tool_calls_json TEXT, tool_results_json TEXT DEFAULT '[]',
    session_input_messages_json TEXT, session_output_unreplied_json TEXT,
    judge_raw_response TEXT,
    should_reply INTEGER DEFAULT 0,
    replies_sent_json TEXT, send_success INTEGER DEFAULT 0, send_duration_ms INTEGER,
    judge_score REAL, judge_is_badcase INTEGER, judge_dimensions_json TEXT,
    human_is_badcase INTEGER, human_badcase_type TEXT, human_notes TEXT,
    tokens_estimated INTEGER DEFAULT 0, duration_ms INTEGER,
    human_labeled_at TEXT, judge_badcase_type TEXT, judge_reason TEXT
);

CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    draft_id TEXT UNIQUE NOT NULL, tick_id INTEGER, chat_name TEXT,
    source TEXT DEFAULT 'committed',
    status TEXT DEFAULT 'pending', badcase_type TEXT, severity TEXT,
    confidence REAL, overall_score REAL, is_badcase INTEGER DEFAULT 0,
    auto_commit INTEGER DEFAULT 0,
    screenshot_path TEXT, judge_reason TEXT, expected_behavior TEXT,
    committed_at TEXT, committed_by TEXT,
    dismissed_at TEXT, dismiss_reason TEXT,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    git_commit TEXT
);

CREATE TABLE IF NOT EXISTS case_prompts (
    case_id INTEGER PRIMARY KEY REFERENCES cases(id),
    system_prompt TEXT, user_prompt TEXT,
    tools_context TEXT, memory_injected TEXT
);

CREATE TABLE IF NOT EXISTS case_tool_calls (
    case_id INTEGER REFERENCES cases(id),
    call_order INTEGER,
    tool_name TEXT,
    arguments TEXT,
    UNIQUE(case_id, call_order)
);

CREATE TABLE IF NOT EXISTS case_llm_messages (
    case_id INTEGER REFERENCES cases(id),
    message_order INTEGER,
    role TEXT,
    content_preview TEXT,
    UNIQUE(case_id, message_order)
);

CREATE TABLE IF NOT EXISTS case_dimensions (
    case_id INTEGER REFERENCES cases(id), dimension_name TEXT, score REAL, comment TEXT,
    UNIQUE(case_id, dimension_name)
);

CREATE TABLE IF NOT EXISTS case_conversations (
    case_id INTEGER REFERENCES cases(id), turn_order INTEGER,
    role TEXT, sender TEXT, text TEXT, UNIQUE(case_id, turn_order)
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    n_samples INTEGER DEFAULT 0,
    control_badcase_rate REAL, exp_badcase_rate REAL,
    control_avg_score REAL, exp_avg_score REAL,
    summary TEXT,
    dimension_diffs_json TEXT,
    is_improvement INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS experiment_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER REFERENCES experiments(id),
    tick_id INTEGER NOT NULL,
    config_name TEXT NOT NULL,
    bot_reply TEXT,
    judge_is_badcase INTEGER DEFAULT 0,
    judge_score REAL DEFAULT 0,
    judge_dimensions_json TEXT,
    judge_reason TEXT,
    UNIQUE(experiment_id, tick_id, config_name)
);
CREATE TABLE IF NOT EXISTS bench_tool_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_name TEXT UNIQUE NOT NULL,
    user_message TEXT NOT NULL, should_call_memory INTEGER NOT NULL DEFAULT 0,
    category TEXT NOT NULL, notes TEXT, enabled INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS bench_reply_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_name TEXT UNIQUE NOT NULL,
    category TEXT NOT NULL, is_group INTEGER DEFAULT 0,
    unreplied_json TEXT NOT NULL, all_messages_json TEXT NOT NULL,
    required_keywords_json TEXT, forbidden_keywords_json TEXT,
    notes TEXT, enabled INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS bench_search_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_name TEXT UNIQUE NOT NULL,
    query TEXT NOT NULL, expected_docs_json TEXT NOT NULL,
    category TEXT NOT NULL, notes TEXT, enabled INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS bench_adversarial_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT UNIQUE NOT NULL,
    query TEXT NOT NULL, sender TEXT, chat_type TEXT DEFAULT 'single',
    ground_truth TEXT, context_json TEXT, category TEXT, enabled INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS daily_metrics (
    date TEXT NOT NULL, benchmark_name TEXT NOT NULL, metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL, git_commit TEXT,
    UNIQUE(date, benchmark_name, metric_name)
);

CREATE INDEX IF NOT EXISTS idx_tick_log_created ON tick_log(created_at);
CREATE INDEX IF NOT EXISTS idx_tick_log_chat ON tick_log(chat_name);
CREATE INDEX IF NOT EXISTS idx_tick_log_judge ON tick_log(judge_score);
CREATE INDEX IF NOT EXISTS idx_cases_created ON cases(created_at);
CREATE INDEX IF NOT EXISTS idx_daily_metrics_date ON daily_metrics(date);

CREATE TABLE IF NOT EXISTS code_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_key TEXT UNIQUE NOT NULL,
    severity TEXT NOT NULL DEFAULT 'P2',
    status TEXT DEFAULT 'pending',
    notes TEXT,
    ai_proposal TEXT,
    updated_at TEXT DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS code_audit_round (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_key TEXT NOT NULL,
    round_num INTEGER NOT NULL,
    user_notes TEXT,
    ai_proposal TEXT,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    UNIQUE(issue_key, round_num)
);
"""


# =============================================================================
# Database class
# =============================================================================

class CaseDB:
    """Badcase 数据库 — 线程安全的 SQLite 封装。"""

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接（调用方负责关闭）。"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.executescript(SCHEMA_SQL)
            conn.commit()
            # 迁移：添加 session_id 列 + 移除 tick_id UNIQUE 约束
            try:
                cur = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='tick_log'")
                ddl = cur.fetchone()[0]
                needs_migration = ('session_id' not in ddl) or ('UNIQUE' in ddl and 'tick_id' in ddl)
                if needs_migration:
                    conn.executescript("""
                        CREATE TABLE IF NOT EXISTS tick_log_new (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL DEFAULT '',
                            tick_id INTEGER NOT NULL,
                            created_at TEXT DEFAULT (datetime('now','localtime')),
                            skip_reason TEXT,
                            chat_name TEXT, is_group INTEGER DEFAULT 0, screenshot_path TEXT,
                            messages_count INTEGER, new_messages_count INTEGER DEFAULT 0,
                            system_prompt TEXT, user_prompt TEXT, raw_response TEXT, tool_calls_json TEXT, tool_results_json TEXT DEFAULT '[]',
                            session_input_messages_json TEXT, session_output_unreplied_json TEXT,
                            judge_raw_response TEXT,
                            should_reply INTEGER DEFAULT 0,
                            replies_sent_json TEXT, send_success INTEGER DEFAULT 0, send_duration_ms INTEGER,
                            judge_score REAL, judge_is_badcase INTEGER, judge_dimensions_json TEXT,
                            human_is_badcase INTEGER, human_badcase_type TEXT, human_notes TEXT,
                            tokens_estimated INTEGER DEFAULT 0, duration_ms INTEGER,
                            human_labeled_at TEXT, judge_badcase_type TEXT, judge_reason TEXT
                        );
                        INSERT INTO tick_log_new SELECT id, '', tick_id, created_at, skip_reason,
                            chat_name, is_group, screenshot_path, messages_count, new_messages_count,
                            system_prompt, user_prompt, raw_response, tool_calls_json,
                            should_reply, replies_sent_json, send_success, send_duration_ms,
                            judge_score, judge_is_badcase, judge_dimensions_json,
                            human_is_badcase, human_badcase_type, human_notes,
                            tokens_estimated, duration_ms, human_labeled_at, judge_badcase_type, judge_reason
                            FROM tick_log;
                        DROP TABLE tick_log;
                        ALTER TABLE tick_log_new RENAME TO tick_log;
                    """)
                    conn.commit()
            except Exception as e:
                _logger.warning("[CaseDB] schema 迁移失败: %s", e)

            # 迁移：添加 session_input_messages_json 和 session_output_unreplied_json 列
            try:
                cur = conn.execute("PRAGMA table_info(tick_log)")
                columns = {row[1] for row in cur.fetchall()}
                if "session_input_messages_json" not in columns:
                    conn.execute("ALTER TABLE tick_log ADD COLUMN session_input_messages_json TEXT")
                if "session_output_unreplied_json" not in columns:
                    conn.execute("ALTER TABLE tick_log ADD COLUMN session_output_unreplied_json TEXT")
                if "judge_raw_response" not in columns:
                    conn.execute("ALTER TABLE tick_log ADD COLUMN judge_raw_response TEXT")
                conn.commit()
            except Exception as e:
                _logger.warning("[CaseDB] 添加 messages_json 列失败: %s", e)

            # 迁移：补全 cases 表缺失列
            try:
                cur = conn.execute("PRAGMA table_info(cases)")
                columns = {row[1] for row in cur.fetchall()}
                for col, dtype in [
                    ("source", "TEXT DEFAULT 'committed'"),
                    ("auto_commit", "INTEGER DEFAULT 0"),
                    ("expected_behavior", "TEXT"),
                    ("committed_at", "TEXT"),
                    ("committed_by", "TEXT"),
                    ("dismissed_at", "TEXT"),
                    ("dismiss_reason", "TEXT"),
                    ("git_commit", "TEXT"),
                ]:
                    if col not in columns:
                        conn.execute(f"ALTER TABLE cases ADD COLUMN {col} {dtype}")
                conn.commit()
            except Exception as e:
                _logger.warning("[CaseDB] cases 表迁移失败: %s", e)

            # 迁移：补全 case_prompts 表缺失列
            try:
                cur = conn.execute("PRAGMA table_info(case_prompts)")
                columns = {row[1] for row in cur.fetchall()}
                for col, dtype in [("tools_context", "TEXT"), ("memory_injected", "TEXT")]:
                    if col not in columns:
                        conn.execute(f"ALTER TABLE case_prompts ADD COLUMN {col} {dtype}")
                conn.commit()
            except Exception as e:
                _logger.warning("[CaseDB] case_prompts 表迁移失败: %s", e)

            # 迁移：创建 case_tool_calls / case_llm_messages 表
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS case_tool_calls (
                        case_id INTEGER REFERENCES cases(id),
                        call_order INTEGER,
                        tool_name TEXT,
                        arguments TEXT,
                        UNIQUE(case_id, call_order)
                    );
                    CREATE TABLE IF NOT EXISTS case_llm_messages (
                        case_id INTEGER REFERENCES cases(id),
                        message_order INTEGER,
                        role TEXT,
                        content_preview TEXT,
                        UNIQUE(case_id, message_order)
                    );
                """)
                conn.commit()
            except Exception as e:
                _logger.warning("[CaseDB] 创建 case_tool_calls / case_llm_messages 表失败: %s", e)

            conn.close()

    # ── INSERT ──

    def insert_case(self, draft: dict) -> int:
        """插入一个完整 case（来自 JudgeWorker 的 draft）。返回 case_id。"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                draft_id = draft.get("draft_id", "")
                if not draft_id:
                    return 0

                judge = draft.get("judge_result", {})

                # Upsert cases
                # Extract screenshot path
                assets = draft.get("assets", {})
                screenshot = assets.get("screenshot_path", "") or draft.get("screenshot_path", "")

                conn.execute("""
                    INSERT INTO cases (draft_id, tick_id, chat_name, source, status,
                        badcase_type, severity, confidence, overall_score,
                        is_badcase, auto_commit, judge_reason, expected_behavior,
                        screenshot_path, committed_at, committed_by, dismissed_at, dismiss_reason,
                        created_at, git_commit)
                    VALUES (?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?)
                    ON CONFLICT(draft_id) DO UPDATE SET
                        status=excluded.status, severity=excluded.severity,
                        overall_score=excluded.overall_score,
                        screenshot_path=COALESCE(excluded.screenshot_path, cases.screenshot_path),
                        committed_at=excluded.committed_at, committed_by=excluded.committed_by,
                        dismissed_at=excluded.dismissed_at, dismiss_reason=excluded.dismiss_reason
                """, (
                    draft_id,
                    draft.get("tick_id"),
                    draft.get("chat_name", ""),
                    draft.get("source", "committed"),
                    draft.get("status", "pending"),
                    judge.get("badcase_type", ""),
                    judge.get("severity", ""),
                    judge.get("confidence", 0),
                    judge.get("overall_score", 0),
                    1 if judge.get("is_badcase") else 0,
                    1 if judge.get("auto_commit") else 0,
                    judge.get("reason", ""),
                    judge.get("expected_behavior", ""),
                    screenshot,
                    draft.get("committed_at"),
                    draft.get("committed_by"),
                    draft.get("dismissed_at"),
                    draft.get("dismiss_reason"),
                    draft.get("timestamp", datetime.now().isoformat()),
                    draft.get("git_commit", ""),
                ))
                conn.commit()

                # Get case_id
                row = conn.execute("SELECT id FROM cases WHERE draft_id = ?", (draft_id,)).fetchone()
                if not row:
                    conn.close()
                    return 0
                case_id = row[0]

                # Delete old sub-records
                allowed_child_tables = {"case_conversations", "case_dimensions", "case_tool_calls"}
                for table in ("case_conversations", "case_dimensions", "case_tool_calls"):
                    if table not in allowed_child_tables:
                        continue
                    sql = "DELETE FROM "
                    sql += table
                    sql += " WHERE case_id = ?"
                    conn.execute(sql, (case_id,))

                # Insert conversations
                conv = draft.get("conversation", [])
                for i, m in enumerate(conv):
                    conn.execute(
                        "INSERT OR REPLACE INTO case_conversations (case_id, turn_order, role, sender, text) VALUES (?, ?, ?, ?, ?)",
                        (case_id, i, m.get("role", "user"), m.get("sender", ""), m.get("text", "")[:5000])
                    )

                # Insert prompts
                sp = draft.get("full_system_prompt", "")
                up = draft.get("full_user_prompt", "")
                tc = draft.get("full_tools_context", "")
                mi = draft.get("memory_injected", "")
                if sp or up:
                    conn.execute(
                        "INSERT OR REPLACE INTO case_prompts (case_id, system_prompt, user_prompt, tools_context, memory_injected) VALUES (?, ?, ?, ?, ?)",
                        (case_id, sp[:30000], up[:30000], tc[:10000], mi[:10000])
                    )

                # Insert dimensions
                dims = judge.get("dimensions", {})
                for name, dd in dims.items():
                    conn.execute(
                        "INSERT OR REPLACE INTO case_dimensions (case_id, dimension_name, score, comment) VALUES (?, ?, ?, ?)",
                        (case_id, name, dd.get("score", 0), dd.get("comment", "")[:1000])
                    )

                # Insert tool calls
                tool_calls = draft.get("tool_calls", [])
                for i, tc_item in enumerate(tool_calls):
                    fn = tc_item.get("function", {}) if isinstance(tc_item, dict) else {}
                    name = fn.get("name", "") if isinstance(fn, dict) else getattr(tc_item, "tool_name", "")
                    args = fn.get("arguments", "") if isinstance(fn, dict) else str(tc_item)
                    conn.execute(
                        "INSERT INTO case_tool_calls (case_id, call_order, tool_name, arguments) VALUES (?, ?, ?, ?)",
                        (case_id, i, name, str(args)[:2000])
                    )

                # Insert LLM messages
                llm_msgs = draft.get("full_llm_messages", [])
                for i, msg in enumerate(llm_msgs):
                    content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                    conn.execute(
                        "INSERT INTO case_llm_messages (case_id, message_order, role, content_preview) VALUES (?, ?, ?, ?)",
                        (case_id, i, msg.get("role", "") if isinstance(msg, dict) else "", str(content)[:3000])
                    )

                conn.commit()
                return case_id
            finally:
                conn.close()

    def update_status(self, draft_id: str, status: str, **kwargs):
        """更新 case 状态。"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                allowed_columns = {
                    "committed_at": "committed_at = ?",
                    "committed_by": "committed_by = ?",
                    "dismissed_at": "dismissed_at = ?",
                    "dismiss_reason": "dismiss_reason = ?",
                    "severity": "severity = ?",
                    "overall_score": "overall_score = ?",
                }
                updates = ["status = ?"]
                params = [status]
                for k, v in kwargs.items():
                    if k in allowed_columns:
                        updates.append(allowed_columns[k])
                        params.append(v)
                params.append(draft_id)
                sql = "UPDATE cases SET "
                sql += ", ".join(updates)
                sql += " WHERE draft_id = ?"
                conn.execute(sql, params)
                conn.commit()
            finally:
                conn.close()

    # ── QUERY ──

    def query_recent(self, days: int = 7, status: Optional[str] = None) -> list[dict]:
        """查询最近 N 天的 case。"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            try:
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                sql = "SELECT * FROM cases WHERE created_at >= ?"
                params = [cutoff]
                if status:
                    sql += " AND status = ?"
                    params.append(status)
                sql += " ORDER BY created_at DESC"
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_case_detail(self, draft_id: str) -> dict | None:
        """获取一个 case 的完整详情（含对话、prompt、评分）。"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            try:
                case = conn.execute("SELECT * FROM cases WHERE draft_id = ?", (draft_id,)).fetchone()
                if not case:
                    return None
                result = dict(case)

                # Conversations
                convs = conn.execute(
                    "SELECT * FROM case_conversations WHERE case_id = ? ORDER BY turn_order", (result["id"],)
                ).fetchall()
                result["conversation"] = [dict(c) for c in convs]

                # Prompts
                prompts = conn.execute(
                    "SELECT * FROM case_prompts WHERE case_id = ?", (result["id"],)
                ).fetchone()
                result["prompts"] = dict(prompts) if prompts else {}

                # Dimensions
                dims = conn.execute(
                    "SELECT * FROM case_dimensions WHERE case_id = ?", (result["id"],)
                ).fetchall()
                result["dimensions"] = {d["dimension_name"]: {"score": d["score"], "comment": d["comment"]} for d in dims}

                # Tools
                tools = conn.execute(
                    "SELECT * FROM case_tool_calls WHERE case_id = ? ORDER BY call_order", (result["id"],)
                ).fetchall()
                result["tool_calls"] = [dict(t) for t in tools]

                return result
            finally:
                conn.close()

    def get_stats(self, days: int = 7) -> dict:
        """获取统计概览。"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                total = conn.execute("SELECT COUNT(*) FROM cases WHERE created_at >= ?", (cutoff,)).fetchone()[0]
                by_type = {}
                for row in conn.execute(
                    "SELECT badcase_type, COUNT(*) as cnt FROM cases WHERE created_at >= ? AND badcase_type != '' GROUP BY badcase_type", (cutoff,)
                ):
                    by_type[row[0]] = row[1]
                by_status = {}
                for row in conn.execute(
                    "SELECT status, COUNT(*) as cnt FROM cases WHERE created_at >= ? GROUP BY status", (cutoff,)
                ):
                    by_status[row[0]] = row[1]
                avg_score_row = conn.execute(
                    "SELECT AVG(overall_score) FROM cases WHERE created_at >= ? AND overall_score > 0", (cutoff,)
                ).fetchone()
                return {
                    "total_cases": total,
                    "by_type": by_type,
                    "by_status": by_status,
                    "avg_score": round(avg_score_row[0], 1) if avg_score_row[0] else 0,
                }
            finally:
                conn.close()

    # ── METRICS ──

    def insert_daily_metrics(self, date: str, benchmarks: dict, git_commit: str = ""):
        """插入每日 benchmark 指标。"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                for bname, bdata in benchmarks.items():
                    if "error" in bdata:
                        continue
                    for key, val in bdata.items():
                        if isinstance(val, (int, float)) and key not in ("tp", "fp", "fn", "tn", "passed", "total", "skipped", "case_count"):
                            conn.execute(
                                "INSERT OR REPLACE INTO daily_metrics (date, benchmark_name, metric_name, metric_value, git_commit) VALUES (?, ?, ?, ?, ?)",
                                (date, bname, key, val, git_commit)
                            )
                conn.commit()
            finally:
                conn.close()

    def get_metric_trend(self, benchmark_name: str, metric_name: str, days: int = 30) -> list[dict]:
        """获取某个指标的历史趋势。"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            try:
                cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                rows = conn.execute(
                    "SELECT date, metric_value FROM daily_metrics WHERE benchmark_name = ? AND metric_name = ? AND date >= ? ORDER BY date",
                    (benchmark_name, metric_name, cutoff)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # ── MIGRATION ──

    def migrate_from_json(self, drafts_dir: Optional[str] = None):
        """从 data/review_drafts/ 迁移已有的 JSON draft 到数据库。"""
        drafts_path = Path(drafts_dir) if drafts_dir else PROJECT_ROOT / "data" / "review_drafts"

        for status_dir in ("committed", "pending", "dismissed"):
            d = drafts_path / status_dir
            if not d.exists():
                continue
            for f in d.glob("*.json"):
                if "mock" in f.name or "test" in f.name:
                    continue
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    data["status"] = status_dir
                    data["source"] = status_dir
                    if not data.get("draft_id"):
                        data["draft_id"] = f.stem
                    case_id = self.insert_case(data)
                    if case_id:
                        print(f"  ✓ {f.stem} → case #{case_id}")
                except Exception as e:
                    print(f"  ✗ {f.stem}: {e}")

    def load_benchmark_cases(self, bench_type: str) -> list[dict]:
        """从 DB 加载 benchmark case。bench_type: 'tool' | 'reply' | 'search' | 'adversarial'"""
        sql_map = {
            "tool": "SELECT * FROM bench_tool_cases WHERE enabled=1",
            "reply": "SELECT * FROM bench_reply_cases WHERE enabled=1",
            "search": "SELECT * FROM bench_search_cases WHERE enabled=1",
            "adversarial": "SELECT * FROM bench_adversarial_cases WHERE enabled=1",
        }
        sql = sql_map.get(bench_type)
        if not sql:
            return []
        conn = self._get_conn()
        try:
            return [dict(r) for r in conn.execute(sql).fetchall()]
        finally:
            conn.close()


# =============================================================================
# Singleton
# =============================================================================

_db_instance: Optional[CaseDB] = None
_db_lock = threading.Lock()


def get_db() -> CaseDB:
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                _db_instance = CaseDB()
    return _db_instance
