#!/usr/bin/env python3
"""清洗 aliases.json 中的脏别名（一次性迁移）。

问题背景：历史 aliases.json 里混入大量非别名噪声——
- 多个别名被顿号连成一整串："示例别名庚、示例别名辛"（导致 "示例别名辛" 召回失败）
- 角色词 / 系统占位符："Bot"、"对话中"、"匿名"、"群主"、"记录者"
- 房号描述："4-1-703"、"6幢5号501"
- 整句描述："被群友@时使用..."、"未发现其他显著别名"

本脚本用 engine 的统一校验逻辑（_split_alias_string + _is_valid_alias）重洗：
- 顿号/斜杠串拆成单条
- 丢弃黑名单 / 房号 / 句子 / 过长条目
- 跨用户去重（别名不能是其他人的主名）
- 去重保序

用法：
    python3 scripts/sanitize_aliases.py            # dry-run，只打印 diff
    python3 scripts/sanitize_aliases.py --apply    # 写回（自动备份 .bak）
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# 让脚本能直接 import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.engine import MemoryEngine, _split_alias_string  # noqa: E402


def sanitize(data: dict, engine: MemoryEngine):
    """原地清洗 data['users']，返回 (before_total, after_total, per_user_diff)。"""
    users = data.get("users", {})
    # 先把所有主名收集起来，供跨用户去重
    all_mains = set(users.keys())

    before_total = 0
    after_total = 0
    diffs = []  # (main, dropped, kept_new)

    for main, cfg in users.items():
        raw = cfg.get("aliases", [])
        before_total += len(raw)
        existing_mains = all_mains - {main}

        kept = []
        seen = set()
        dropped = []
        for alias in raw:
            for a in _split_alias_string(alias):
                if a in seen:
                    continue
                if engine._is_valid_alias(a, main, existing_mains):
                    seen.add(a)
                    kept.append(a)
                else:
                    if a not in seen:
                        dropped.append(a)
                        seen.add(a)  # 避免重复记录同一丢弃项
        after_total += len(kept)
        cfg["aliases"] = kept
        if dropped or len(kept) != len(raw):
            diffs.append((main, dropped, kept))

    return before_total, after_total, diffs


def main():
    ap = argparse.ArgumentParser(description="清洗 aliases.json 脏别名")
    ap.add_argument("--apply", action="store_true", help="写回文件（默认只 dry-run）")
    ap.add_argument("--path", default="data/memory/overrides/aliases.json",
                    help="aliases.json 路径")
    args = ap.parse_args()

    path = (PROJECT_ROOT / args.path).resolve()
    if not path.exists():
        print(f"❌ 文件不存在: {path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(path.read_text(encoding="utf-8"))
    engine = MemoryEngine.__new__(MemoryEngine)  # 不触发 __init__ 的后台线程
    engine._aliases = {}

    before, after, diffs = sanitize(data, engine)

    print(f"文件: {path}")
    print(f"用户数: {len(data.get('users', {}))}")
    print(f"别名总数: {before} → {after}  (丢弃 {before - after} 条)\n")

    if not diffs:
        print("✅ 无需清洗，数据已干净。")
        return

    print(f"变更用户 {len(diffs)} 个，明细（最多展示每用户前 10 条丢弃项）：\n")
    for main, dropped, kept in diffs:
        print(f"## {main}")
        print(f"  保留 ({len(kept)}): {kept}")
        show = dropped[:10]
        more = f" …另有 {len(dropped)-10} 条" if len(dropped) > 10 else ""
        print(f"  丢弃 ({len(dropped)}): {show}{more}\n")

    if not args.apply:
        print("（dry-run，未写回。加 --apply 执行写回。）")
        return

    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"✅ 已写回 {path}，备份在 {bak}")


if __name__ == "__main__":
    main()
