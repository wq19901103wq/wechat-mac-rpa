#!/usr/bin/env python3
"""迁移 P0/P2/P4 硬编码 case → database。"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.badcase.case_db import CaseDB


def migrate_tool_decision(db: CaseDB):
    from src.tests.test_tool_decision_benchmark import BENCHMARK_CASES as _OLD_TOOL_CASES
    conn = db._get_conn()
    for c in _OLD_TOOL_CASES:
        conn.execute("""INSERT OR REPLACE INTO benchmark_tool_cases
            (case_name, user_message, should_call_memory, category, notes, evaluation_mode)
            VALUES (?, ?, ?, ?, ?, ?)""", (
            c.case_name, c.user_message,
            1 if c.should_call_memory else 0,
            c.category, c.notes, "binary"
        ))
    conn.commit()
    print(f"  P0 Tool Decision: {len(_OLD_TOOL_CASES)} cases")


def migrate_reply_quality(db: CaseDB):
    from src.tests.test_reply_quality_benchmark import BENCHMARK_CASES as _OLD_REPLY_CASES
    conn = db._get_conn()
    for c in _OLD_REPLY_CASES:
        unreplied = []
        for m in c.unreplied:
            unreplied.append({"sender": m.sender, "text": m.text})
        all_msgs = []
        for m in c.all_messages:
            all_msgs.append({
                "sender": m.sender, "text": m.text,
                "sender_type": "self" if m.sender_type.value == "self" else "other",
            })
        rubric_name = ""
        if hasattr(c, 'rubric') and c.rubric and hasattr(c.rubric, 'instructions'):
            rubric_name = c.rubric.instructions[:50]

        conn.execute("""INSERT OR REPLACE INTO benchmark_reply_cases
            (case_name, category, is_group, unreplied_json, all_messages_json,
             required_keywords_json, required_hits, forbidden_keywords_json,
             min_replies, max_replies, rubric_name, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            c.case_name, c.category,
            1 if c.is_group else 0,
            json.dumps(unreplied, ensure_ascii=False),
            json.dumps(all_msgs, ensure_ascii=False),
            json.dumps(c.required_keywords, ensure_ascii=False) if c.required_keywords else "[]",
            c.required_hits,
            json.dumps(c.forbidden_keywords, ensure_ascii=False) if c.forbidden_keywords else "[]",
            c.min_replies, c.max_replies, rubric_name, c.notes,
        ))
    conn.commit()
    conn.close()
    print(f"  P2 Reply Quality: {len(_OLD_REPLY_CASES)} cases")


def migrate_memory_search(db: CaseDB):
    from src.tests.test_memory_search_benchmark import BENCHMARK_CASES as _OLD_SEARCH_CASES
    conn = db._get_conn()
    for c in _OLD_SEARCH_CASES:
        conn.execute("""INSERT OR REPLACE INTO benchmark_search_cases
            (case_name, query, expected_docs_json, unexpected_docs_json,
             required_fragments_json, category, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)""", (
            c.case_name, c.query,
            json.dumps(c.expected_docs, ensure_ascii=False),
            json.dumps(c.unexpected_docs, ensure_ascii=False) if c.unexpected_docs else "[]",
            json.dumps(c.required_fragments, ensure_ascii=False) if c.required_fragments else "[]",
            c.category, c.notes,
        ))
    conn.commit()
    conn.close()
    print(f"  P4 Memory Search: {len(_OLD_SEARCH_CASES)} cases")


def main():
    db = CaseDB()
    print("Migration → database:")
    migrate_tool_decision(db)
    migrate_reply_quality(db)
    migrate_memory_search(db)
    print(f"\nDone. Verify: sqlite3 data/cases.db 'SELECT COUNT(*) FROM benchmark_tool_cases'")


if __name__ == "__main__":
    main()
