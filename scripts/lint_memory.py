#!/usr/bin/env python3
"""记忆库健康检查 + 修复（LLMWiki 的 Lint 操作）。

对标 Karpathy LLMWiki 的 Lint：定期扫描记忆库，发现矛盾、膨胀、重复、孤立，
保持 wiki 作为"编译摘要"的结构清晰（详见 docs/02-architecture/specs/MEMORY_SPEC.md FR-7/8）。

用法：
    python3 scripts/lint_memory.py                  # 只检查，打印报告
    python3 scripts/lint_memory.py --truncate       # 截断膨胀 wiki（先备份）
    python3 scripts/lint_memory.py --json           # 输出原始 JSON
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.engine import MemoryEngine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="记忆库 Lint 健康检查")
    ap.add_argument("--truncate", action="store_true", help="截断膨胀 wiki（自动备份到 .lint-bak）")
    ap.add_argument("--json", action="store_true", help="输出 JSON 报告")
    ap.add_argument(
        "--max-chars",
        type=int,
        default=4000,
        help="wiki 长度上限（与 engine._enforce_wiki_limits / prompt 约束一致）",
    )
    args = ap.parse_args()

    engine = MemoryEngine.__new__(MemoryEngine)
    engine.wiki_dir = PROJECT_ROOT / "data" / "memory" / "wiki"
    engine._aliases = {}
    # 加载真实 aliases 用于冲突检测
    aliases_path = PROJECT_ROOT / "data" / "memory" / "overrides" / "aliases.json"
    if aliases_path.exists():
        data = json.loads(aliases_path.read_text(encoding="utf-8"))
        for user, cfg in data.get("users", {}).items():
            engine._aliases[user] = cfg.get("aliases", [])

    report = engine.lint_memory(args.max_chars)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print("=== 记忆库 Lint 报告 ===\n")
    print(f"别名冲突: {len(report['conflicts'])}")
    print(f"膨胀 wiki (> {args.max_chars} 字): {len(report['bloated'])}")
    print(f"重复群/用户(归一化后同名): {len(report['duplicates'])}")
    print(f"广告群: {len(report['ad_groups'])}\n")

    if report["conflicts"]:
        print("--- 别名冲突（同一别名指向多个主名，需人工裁决）---")
        for c in report["conflicts"][:20]:
            print(f"  {c['alias']!r} -> {c['mains']}")
        if len(report["conflicts"]) > 20:
            print(f"  ...另有 {len(report['conflicts'])-20} 条")
        print()

    if report["duplicates"]:
        print("--- 重复群/用户（归一化后同名，建议合并）---")
        for d in report["duplicates"][:20]:
            print(f"  [{d['normalized']}] {d['names']}")
        print()

    if report["ad_groups"]:
        print("--- 广告群（建议删除 wiki，加黑名单）---")
        for g in report["ad_groups"]:
            print(f"  {g}")
        print()

    if report["bloated"]:
        print(f"--- 膨胀 wiki ({len(report['bloated'])} 个) ---")
        for b in sorted(report["bloated"], key=lambda x: -x["chars"])[:15]:
            print(f"  {b['chars']:>6} 字  {b['path']}")
        if len(report["bloated"]) > 15:
            print(f"  ...另有 {len(report['bloated'])-15} 个")

    if args.truncate and report["bloated"]:
        print("\n=== 执行截断 ===")
        bak_dir = PROJECT_ROOT / "data" / "memory" / "wiki" / ".lint-bak"
        bak_dir.mkdir(parents=True, exist_ok=True)
        # 先备份所有膨胀文件（原始内容），再截断
        wiki_root = PROJECT_ROOT / "data" / "memory" / "wiki"
        for b in report["bloated"]:
            src = wiki_root / b["path"]
            bak = bak_dir / src.name
            if not bak.exists():
                shutil.copy2(src, bak)
        actions = engine.lint_truncate_bloated(args.max_chars, apply=True)
        for a in actions:
            print(f"  {a['before']} -> {a['after']}  {a['path']}")
        print(f"\n截断 {len(actions)} 个文件，原始备份在 {bak_dir}")
    elif report["bloated"] and not args.truncate:
        print("\n（dry-run。加 --truncate 执行截断，自动备份到 data/memory/wiki/.lint-bak）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
