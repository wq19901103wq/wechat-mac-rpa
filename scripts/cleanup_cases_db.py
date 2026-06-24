#!/usr/bin/env python3
"""清理 cases.db tick_log 表（NFR-5）。

问题：tick_log.session_input_messages_json 平均 470KB/行，5394 行吃掉 2.5GB，
且无保留策略，从 2026-05-23 只增不删。

策略：
- 删除 created_at 早于保留期（默认 7 天）的 tick_log 行
- VACUUM 回收磁盘空间
- 同时清理空的 case_db.sqlite（0 字节残留，与 cases.db 并存的配置不一致产物）

用法：
    python3 scripts/cleanup_cases_db.py                  # dry-run，只统计
    python3 scripts/cleanup_cases_db.py --apply          # 执行删除 + VACUUM
    python3 scripts/cleanup_cases_db.py --apply --days 14
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "cases.db"
EMPTY_DB_PATH = PROJECT_ROOT / "data" / "case_db.sqlite"


def main() -> int:
    ap = argparse.ArgumentParser(description="清理 cases.db tick_log")
    ap.add_argument("--apply", action="store_true", help="执行删除+VACUUM（默认 dry-run）")
    ap.add_argument("--days", type=int, default=7, help="保留天数（默认 7）")
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f"❌ 数据库不存在: {DB_PATH}", file=sys.stderr)
        return 1

    size_before = DB_PATH.stat().st_size
    print(f"数据库: {DB_PATH}")
    print(f"当前大小: {size_before / 1024 / 1024:.1f} MB")
    print(f"保留策略: {args.days} 天{'（dry-run，不实际删除）' if not args.apply else '（将执行）'}\n")

    conn = sqlite3.connect(str(DB_PATH))

    # 统计
    total = conn.execute("SELECT COUNT(*) FROM tick_log").fetchone()[0]
    cutoff_row = conn.execute(
        "SELECT COUNT(*) FROM tick_log WHERE created_at < datetime('now', ?)",
        (f"-{args.days} days",),
    ).fetchone()[0]
    keep = total - cutoff_row
    print(f"tick_log 总行数: {total}")
    print(f"将删除（早于 {args.days} 天）: {cutoff_row} 行")
    print(f"保留: {keep} 行\n")

    # 时间范围
    mn, mx = conn.execute("SELECT MIN(created_at), MAX(created_at) FROM tick_log").fetchone()
    print(f"时间范围: {mn} ~ {mx}")

    if not args.apply:
        conn.close()
        print("\n（dry-run。加 --apply 执行删除 + VACUUM）")
        return 0

    # 执行删除
    conn.execute(
        "DELETE FROM tick_log WHERE created_at < datetime('now', ?)",
        (f"-{args.days} days",),
    )
    conn.commit()
    print("\n✅ 删除完成，开始 VACUUM（可能需要数十秒）...")
    conn.execute("VACUUM")
    conn.close()

    size_after = DB_PATH.stat().st_size
    print("✅ VACUUM 完成")
    print(f"大小: {size_before/1024/1024:.1f} MB → {size_after/1024/1024:.1f} MB "
          f"(省 {(size_before-size_after)/1024/1024:.1f} MB)")

    # 清理空的 case_db.sqlite
    if EMPTY_DB_PATH.exists() and EMPTY_DB_PATH.stat().st_size == 0:
        os.remove(EMPTY_DB_PATH)
        print(f"✅ 删除空残留文件: {EMPTY_DB_PATH.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
