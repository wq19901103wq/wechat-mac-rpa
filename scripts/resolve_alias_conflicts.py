#!/usr/bin/env python3
"""裁决并修复 aliases.json 中的别名冲突（一次性迁移）。

冲突来源：LLM 别名发现把同一个别名归给了多个主名，或把 OCR 噪声/群名当成人名建主名。
裁决依据：人工查看各主名 wiki 的"别名"段落与上下文（lint 报告 + wiki 内容核对）。

处理三类问题：
1. 纯垃圾别名：单字符、OCR 噪声词（[待验证]/~/至/续/内容未显示/Wang/Zhang...）→ 从所有主名删除
2. 真冲突别名：按裁决表归一方，另一方移除
3. 幽灵主名：aliases.json 有条目但 wiki/ 目录无对应 .md 的（OCR 把群名/时间戳当人名）→ 整条删除

用法：
    python3 scripts/resolve_alias_conflicts.py            # dry-run
    python3 scripts/resolve_alias_conflicts.py --apply    # 写回（自动备份）
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 1. 纯垃圾别名：直接从所有主名删除 ──
GARBAGE_ALIASES = {
    "[待验证]", "~", "至", "续", "含车内带娃", "内容未显示", "啧啧[旺柴]",
    "Wang", "Zhang", "小a",  # 小a:"不爱说话"AI 是项目名非人
}

# ── 2. 真冲突裁决：alias -> 归属主名（其他主名移除该别名）──
# 依据：各主名 wiki 的"别名"段落。归属方都是 wiki 里明确标注该别名、证据更全的主名。
CONFLICT_VERDICT = {
    "g神": "王芊",            # 王芊 wiki 多处明确"别名:G神"；张波移除
    "小海哥": "王海",          # 王海 wiki 明确"别名:小海哥"；王芊只是引用表哥
    "老王": "王旭东 住别墅但是爱吃干脆面",  # 王旭东 wiki 明确"老王(群友称呼)"；钱文英俊是引用
    "G少爷": "Ghost-大脖子-魏一博",   # Ghost wiki 明确"别名:G少爷"；张波移除
    "G总": "G总-王志腾-有三个亿比特币",  # 主名即 G总；芝麻标[待验证]移除
    "白姐": "白",             # 白 wiki 明确"别名:白姐"；"白:"是 OCR 脏主名
}

# ── 3. 保守不裁决：两边都删该别名（无法从 wiki 确定归属，宁可少不可错）──
DROP_FROM_BOTH = {
    "钦妈", "Jay", "云玺", "Fei", "李哥",
    # "白:" 主名整个删除（OCR 带冒号脏主名）
}

# OCR 脏主名特征：整条从 aliases 删除（无 wiki 或主名明显是群名/时间戳/带标点）
GARBAGE_MAIN_PATTERNS = [
    lambda m: m.endswith(":"),
    lambda m: m.endswith("。"),
    lambda m: m.startswith("2026-") or m.startswith("2025-") or m.startswith("2024-"),
    lambda m: "团购" in m and len(m) > 4,
    lambda m: m in {"联系供应商", "配送与分发", "母亲节专享接龙", "后记",
                    "微信群机器人接入热潮", "CowAgent 功能讨论", "典型问题与解决方案",
                    "GPT 代充与号池服务", "梯子/VPN/机场讨论", "模型与编码能力讨论",
                    "育儿讨论", "带娃梗", "42-202-补位ing"},
]


def is_garbage_main(main: str) -> bool:
    return any(p(main) for p in GARBAGE_MAIN_PATTERNS)


def main() -> int:
    ap = argparse.ArgumentParser(description="裁决别名冲突")
    ap.add_argument("--apply", action="store_true", help="写回（默认 dry-run）")
    ap.add_argument("--path", default="data/memory/overrides/aliases.json")
    args = ap.parse_args()

    path = (PROJECT_ROOT / args.path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    users = data.get("users", {})
    wiki_users = PROJECT_ROOT / "data" / "memory" / "wiki" / "users"

    removed_aliases = []   # (main, alias, reason)
    removed_mains = []     # (main, reason)

    for main, cfg in list(users.items()):
        aliases = cfg.get("aliases", [])
        new_aliases = []
        for a in aliases:
            if a in GARBAGE_ALIASES:
                removed_aliases.append((main, a, "垃圾别名"))
                continue
            if a in DROP_FROM_BOTH:
                removed_aliases.append((main, a, "冲突无法裁决，两边删"))
                continue
            if a in CONFLICT_VERDICT:
                if CONFLICT_VERDICT[a] == main:
                    new_aliases.append(a)  # 归属方保留
                else:
                    removed_aliases.append((main, a, f"冲突裁决归 {CONFLICT_VERDICT[a]}"))
                continue
            new_aliases.append(a)
        cfg["aliases"] = new_aliases

    # 删除幽灵/脏主名
    for main in list(users.keys()):
        wiki_path = wiki_users / f"{main}.md"
        if is_garbage_main(main):
            removed_mains.append((main, "脏主名(OCR群名/时间戳/标点)"))
            del users[main]
        elif not wiki_path.exists():
            # 无 wiki 文件的幽灵主名：仅当别名也为空或全是垃圾时删除
            remaining = users[main].get("aliases", [])
            if not remaining:
                removed_mains.append((main, "幽灵主名(无wiki无别名)"))
                del users[main]

    print("=== 别名冲突裁决报告 ===\n")
    print(f"删除垃圾/冲突别名: {len(removed_aliases)} 条")
    print(f"删除脏/幽灵主名: {len(removed_mains)} 个\n")

    if removed_aliases:
        print("--- 删除的别名 ---")
        from collections import defaultdict
        by_main = defaultdict(list)
        for m, a, r in removed_aliases:
            by_main[m].append((a, r))
        for m, items in by_main.items():
            print(f"  {m}:")
            for a, r in items:
                print(f"    - {a!r}  ({r})")

    if removed_mains:
        print("\n--- 删除的主名 ---")
        for m, r in removed_mains:
            print(f"  {m!r}  ({r})")

    # 裁决保留情况
    print("\n--- 冲突裁决归属 ---")
    for alias, kept in CONFLICT_VERDICT.items():
        print(f"  {alias!r} → 归 {kept!r}")

    if not args.apply:
        print("\n（dry-run，未写回。加 --apply 执行）")
        return 0

    bak = path.with_suffix(path.suffix + ".conflict-bak")
    shutil.copy2(path, bak)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n✅ 已写回 {path}，备份 {bak}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
