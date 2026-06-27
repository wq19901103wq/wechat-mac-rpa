#!/usr/bin/env python3
"""Wiki 事实错误的外科手术式清洗（只删/改被证伪的行，不整篇重生成）。

消费 detect_wiki_contradictions.py 的 contradictions.json + audit_wiki.py 的各 wiki
audit json，产出 EditPlan，默认 dry-run，--apply 才写盘（带备份）。

三层操作（精度从高到低）：
  GARBAGE（自动 apply）：
    - A5 别名行垃圾词：用 contradictions.json 给出的 cleaned_line 整行替换
    - A1+A2 昵称错配 STRIP_TOKEN：A1 内部冲突 + A2 跨 wiki 判定 suggested_owner，
      从"非建议归属"的那一行剥掉错配昵称 token（不动关系词）
  VERDICT（自动 apply）：
    - audit unverified 事实：若 wiki_excerpt 能唯一匹配到一行，追加 [待验证]
  REVIEW（写 review.json，不 apply）：
    - A3 Bot 来源行、audit contradicted、A2 平票、无法唯一匹配的 unverified

设计原则（防误删/防引入新错）：
  - LLM 不自由改写整行；自动 apply 仅 STRIP_TOKEN / REPLACE_LINE(别名) / APPEND_TAG
  - 行定位用子串匹配 + 唯一性校验（匹配 0 或 >1 行则跳过进 review）
  - 每条修改带证据指纹（A2 投票明细 / A1 冲突 / audit verdict）
  - apply 前后 ## 标题数不变；备份到 .clean-bak-YYYYMMDD/

用法：
    python3 scripts/clean_wiki_errors.py --pilot              # dry-run 试点
    python3 scripts/clean_wiki_errors.py --pilot --apply      # 写盘
    python3 scripts/clean_wiki_errors.py --user 王芊          # 单个
    python3 scripts/clean_wiki_errors.py --pilot --json       # 输出 EditPlan JSON
"""

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = PROJECT_ROOT / "data" / "memory" / "wiki"
USERS_DIR = WIKI_DIR / "users"
GROUPS_DIR = WIKI_DIR / "groups"
AUDIT_DIR = PROJECT_ROOT / "data" / "memory" / "wiki_audit"
CONTRA_PATH = AUDIT_DIR / "contradictions.json"
REVIEW_PATH = AUDIT_DIR / "review.json"
DECISIONS_PATH = AUDIT_DIR / "decisions.json"
PILOT_LIST = PROJECT_ROOT / "data" / "memory" / "pilot_list.json"

# 昵称 token 及其标记词的剥离模式：，群昵称"X" / ，昵称"X" / 网名"X" 等（按 nick 动态构建）
def _nick_strip_patterns(nick: str) -> List[re.Pattern]:
    n = re.escape(nick)
    return [
        re.compile(r"[，,]\s*(?:群昵称|昵称|网名|微信名|微信号|群名)\s*[：:]?\s*[“\"]" + n + r"[”\"]\s*"),
        re.compile(r"\s*(?:群昵称|昵称|网名|微信名|微信号|群名)\s*[：:]?\s*[“\"]" + n + r"[”\"]\s*[，,]\s*"),
        re.compile(r"[（(]\s*(?:群昵称|昵称|网名|微信名)?\s*[“\"]?" + n + r"[”\"]?\s*[）)]"),
    ]


def _strip_nick_from_line(line: str, nick: str) -> str:
    """从一行剥掉指定昵称 token 及其标记词，清理孤立标点。"""
    out = line
    for pat in _nick_strip_patterns(nick):
        out = pat.sub("", out, count=1)
    # 兜底：若仍含该昵称引号串，移除
    out = out.replace(f"“{nick}”", "").replace(f"\"{nick}\"", "")
    # 清理孤立逗号/顿号
    out = re.sub(r"[，,]{2,}", "，", out)
    out = re.sub(r"[，,]\s*([）)）])", r"\1", out)
    return out


def _count_headers(wiki: str) -> int:
    return sum(1 for ln in wiki.split("\n") if ln.startswith("## "))


def _find_unique_line(lines: List[str], needle: str) -> Optional[int]:
    """子串匹配，返回唯一命中的行号（0-based）；0 或 >1 命中返回 None。"""
    hits = [i for i, ln in enumerate(lines) if needle in ln]
    if len(hits) == 1:
        return hits[0]
    return None


def _build_edit_plan(
    name: str, is_group: bool, wiki: str, contra: Dict[str, Any], audit: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """为一个 wiki 构建 EditPlan。"""
    uid = f"{name}@group" if is_group else name
    cw = contra.get("per_wiki", {}).get(uid, {})
    lines = wiki.split("\n")
    edits: List[Dict[str, Any]] = []
    review: List[Dict[str, Any]] = []

    # ── GARBAGE: A5 别名行替换 ──
    for g in cw.get("alias_garbage", []):
        old_line = g["line"]
        new_line = g["cleaned_line"]
        if old_line == new_line:
            continue
        idx = _find_unique_line(lines, old_line[:60])  # 用前 60 字定位（行可能很长）
        if idx is None:
            review.append({"layer": "GARBAGE", "op": "REPLACE_LINE", "reason": "A5 别名垃圾词",
                           "line": old_line[:120], "issue": "行未唯一匹配，需人工核对",
                           "proposed": new_line[:120]})
            continue
        edits.append({
            "layer": "GARBAGE", "op": "REPLACE_LINE", "line_idx": idx,
            "old": lines[idx], "new": new_line,
            "evidence": f"A5 别名垃圾词: {g['garbage_tokens']}",
            "reason": "别名行含黑名单/描述性脏 token，程序化过滤",
        })

    # ── GARBAGE: A1+A2 昵称错配 STRIP_TOKEN ──
    a2_by_nick = {c["nickname"]: c for c in contra.get("cross_nick_conflicts", [])}
    for c in cw.get("internal_contradictions", []):
        nick = c["nickname"]
        a2 = a2_by_nick.get(nick, {})
        suggested = a2.get("suggested_owner")
        if not suggested:
            review.append({"layer": "VERDICT", "op": "STRIP_TOKEN", "reason": f"A1 昵称碰撞 {nick}",
                           "owners": c["owners"], "issue": "A2 未给出建议归属（平票/证据不足），需人工裁决",
                           "lines": c["lines"]})
            continue
        # 从"非建议归属"的行剥掉昵称 token
        for ln_text in c["lines"]:
            if suggested in ln_text:
                continue  # 建议归属行，保留
            # 这行是错配行：定位并 strip
            idx = _find_unique_line(lines, ln_text[:60])
            if idx is None:
                review.append({"layer": "VERDICT", "op": "STRIP_TOKEN", "reason": f"A1 昵称碰撞 {nick}",
                               "issue": "错配行未唯一匹配", "line": ln_text[:120],
                               "suggested_owner": suggested})
                continue
            new_line = _strip_nick_from_line(lines[idx], nick)
            if new_line == lines[idx]:
                review.append({"layer": "VERDICT", "op": "STRIP_TOKEN", "reason": f"A1 昵称碰撞 {nick}",
                               "issue": "strip 后行无变化（昵称壳未识别）", "line": ln_text[:120],
                               "suggested_owner": suggested})
                continue
            edits.append({
                "layer": "GARBAGE", "op": "STRIP_TOKEN", "line_idx": idx,
                "old": lines[idx], "new": new_line, "nickname": nick,
                "evidence": f"A2 跨 wiki 投票: {suggested} (gap={a2.get('gap')}, votes={ {k:v['score'] for k,v in a2.get('votes',{}).items()} })",
                "reason": f"昵称 {nick} 跨 wiki 应归 {suggested}，从本行剥除错配",
            })

    # ── REVIEW: A3 Bot 来源行（不自动改，送审）──
    for b in cw.get("bot_suspect_lines", []):
        review.append({"layer": "REVIEW", "op": "INSPECT", "reason": "A3 Bot 来源行",
                       "section": b.get("section"), "line": b["line"][:120],
                       "quality": b.get("reason"), "suggested": "向用户求证或标 [待验证]"})

    # ── REVIEW: audit 事实（pilot 阶段不自动改，全部送审）──
    # unverified 的行级匹配是模糊的（audit 只给 wiki_excerpt 片段），自动 APPEND 风险高，
    # 故 pilot 一律送 review；contradicted 同理。等 audit 重跑 + 人工确认后再决定是否自动。
    if audit:
        raw_ev = audit.get("raw_evidence", "")
        for f in audit.get("facts", []):
            verdict = f.get("verdict")
            excerpt = f.get("wiki_excerpt", "")
            hal = f.get("hallucinated_evidence", False)
            if verdict == "contradicted":
                review.append({"layer": "REVIEW", "op": "FIX_OR_DELETE", "reason": "audit contradicted",
                               "fact": f.get("fact"), "wiki_excerpt": excerpt,
                               "evidence": f.get("evidence"), "hallucinated_evidence": hal,
                               "raw_evidence": raw_ev[:1500],
                               "suggested": "核对原始证据后修正或删除该行"})
            elif verdict == "unverified":
                review.append({"layer": "REVIEW", "op": "MARK_UNVERIFIED", "reason": "audit unverified",
                               "fact": f.get("fact"), "wiki_excerpt": excerpt,
                               "evidence": f.get("evidence"), "hallucinated_evidence": hal,
                               "raw_evidence": raw_ev[:1500],
                               "suggested": "向用户求证后标 [待验证] 或删除"})

    return {"name": name, "is_group": is_group, "edits": edits, "review": review,
            "header_count": _count_headers(wiki)}


def _apply_edits(wiki: str, plan: Dict[str, Any]) -> str:
    """对 wiki 应用 plan 中的 edits（行级替换）。同一行多 edit 取最后状态。"""
    lines = wiki.split("\n")
    # 按行号聚合，后一个 edit 覆盖前一个（同层内无冲突）
    by_idx: Dict[int, str] = {}
    for e in plan["edits"]:
        by_idx[e["line_idx"]] = e["new"]
    for i, new in by_idx.items():
        lines[i] = new
    return "\n".join(lines)


def _resolve_targets(args) -> List[tuple]:
    if args.user:
        return [(args.user, False)]
    if args.group:
        return [(args.group, True)]
    if args.pilot:
        if not PILOT_LIST.exists():
            print(f"✗ pilot_list.json 不存在: {PILOT_LIST}", file=sys.stderr)
            sys.exit(2)
        data = json.loads(PILOT_LIST.read_text(encoding="utf-8"))
        return [(it["name"], it.get("is_group", False)) for it in data["items"]]
    print("✗ 请指定 --user / --group / --pilot", file=sys.stderr)
    sys.exit(2)


def _apply_decisions(stamp: str) -> int:
    """读 decisions.json，按人工确认动作写回 wiki（带备份）。

    decisions.json: [{id, wiki, is_group, action, new_value?, decided_at}]
      action: delete(删行) / fix(修正为 new_value) / mark(追加 [待验证]) / skip(跳过)
    """
    if not DECISIONS_PATH.exists():
        print(f"✗ {DECISIONS_PATH} 不存在，先在网页 /wiki-review 确认", file=sys.stderr)
        return 1
    decisions = json.loads(DECISIONS_PATH.read_text(encoding="utf-8"))
    # 按 wiki 分组
    by_wiki: Dict[str, List[Dict[str, Any]]] = {}
    for d in decisions:
        if d.get("action") == "skip":
            continue
        by_wiki.setdefault(d["wiki"], []).append(d)

    bak_dir = WIKI_DIR / f".clean-bak-{stamp}"
    bak_dir.mkdir(parents=True, exist_ok=True)
    n_applied = 0
    total = 0
    for wiki_name, items in by_wiki.items():
        is_group = items[0].get("is_group", False)
        d = GROUPS_DIR if is_group else USERS_DIR
        path = d / f"{wiki_name}.md"
        if not path.exists():
            print(f"⚠ 跳过（无 wiki）: {wiki_name}", file=sys.stderr)
            continue
        wiki = path.read_text(encoding="utf-8")
        lines = wiki.split("\n")
        header_before = _count_headers(wiki)
        changed = False
        deleted_idxs: Set[int] = set()
        for dec in items:
            needle = dec.get("line") or dec.get("wiki_excerpt") or dec.get("fact", "")
            if not needle:
                continue
            idx = _find_unique_line(lines, needle[:40])
            action = dec["action"]
            if idx is None:
                print(f"  ⚠ [{wiki_name}] id={dec.get('id')} 未唯一匹配行，跳过: {needle[:40]}")
                continue
            if action == "delete":
                deleted_idxs.add(idx)  # 标记删除（稍后过滤）
                changed = True
                total += 1
            elif action == "fix":
                new_val = dec.get("new_value", "").strip()
                if new_val:
                    lines[idx] = new_val
                    changed = True
                    total += 1
            elif action == "mark":
                if idx not in deleted_idxs and "[待验证]" not in lines[idx]:
                    lines[idx] = lines[idx].rstrip() + " [待验证]"
                    changed = True
                    total += 1
        if not changed:
            continue
        new_wiki = "\n".join(ln for i, ln in enumerate(lines) if i not in deleted_idxs)
        # 结构校验
        if _count_headers(new_wiki) != header_before:
            print(f"  ✗ [{wiki_name}] 结构校验失败（## 标题数 {header_before}→{_count_headers(new_wiki)}），跳过写盘")
            continue
        shutil.copy2(path, bak_dir / path.name)
        path.write_text(new_wiki, encoding="utf-8")
        n_applied += 1
        print(f"  ✅ [{wiki_name}] 应用 {sum(1 for it in items if it['action']!='skip')} 条决策，备份 {bak_dir / path.name}")
    print(f"\n=== 决策写盘汇总 ===")
    print(f"wiki: {n_applied}，应用决策: {total} 条，备份目录: {bak_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Wiki 外科手术清洗")
    ap.add_argument("--user", help="单个 user wiki")
    ap.add_argument("--group", help="单个 group wiki")
    ap.add_argument("--pilot", action="store_true", help="读 pilot_list.json")
    ap.add_argument("--apply", action="store_true", help="写盘（默认 dry-run）")
    ap.add_argument("--apply-decisions", action="store_true",
                    help="读 decisions.json 按人工确认写回 wiki（网页 /wiki-review 确认后用）")
    ap.add_argument("--json", action="store_true", help="输出 EditPlan JSON")
    args = ap.parse_args()

    if not CONTRA_PATH.exists():
        print(f"✗ 请先运行 detect_wiki_contradictions.py 生成 {CONTRA_PATH}", file=sys.stderr)
        return 1
    contra = json.loads(CONTRA_PATH.read_text(encoding="utf-8"))

    if args.apply_decisions:
        return _apply_decisions(time.strftime("%Y%m%d"))

    targets = _resolve_targets(args)
    plans = []
    all_review = []
    stamp = time.strftime("%Y%m%d")
    bak_dir = WIKI_DIR / f".clean-bak-{stamp}"
    n_applied = 0

    for name, is_group in targets:
        d = GROUPS_DIR if is_group else USERS_DIR
        path = d / f"{name}.md"
        if not path.exists():
            print(f"⚠ 跳过（无 wiki）: {name}", file=sys.stderr)
            continue
        wiki = path.read_text(encoding="utf-8")
        audit = None
        audit_path = AUDIT_DIR / f"{name}.json"
        if audit_path.exists():
            try:
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
            except Exception:
                audit = None
        plan = _build_edit_plan(name, is_group, wiki, contra, audit)
        plans.append(plan)
        for r in plan["review"]:
            r["wiki"] = name
            r["is_group"] = is_group
            all_review.append(r)

        if args.json:
            continue
        # dry-run 打印
        print(f"\n=== {name}{' (group)' if is_group else ''} ===")
        if plan["edits"]:
            print(f"  [自动 apply] {len(plan['edits'])} 条编辑:")
            for e in plan["edits"]:
                print(f"    [{e['layer']}/{e['op']}] 行{e['line_idx']}: {e['old'][:60]}")
                print(f"      → {e['new'][:60]}")
                print(f"      证据: {e['evidence'][:80]}")
        if plan["review"]:
            print(f"  [送 review] {len(plan['review'])} 条:")
            for r in plan["review"][:5]:
                print(f"    [{r['layer']}] {r.get('reason')}: {r.get('line', r.get('fact',''))[:60]}")
            if len(plan["review"]) > 5:
                print(f"    ... 还有 {len(plan['review'])-5} 条")

        if args.apply and plan["edits"]:
            new_wiki = _apply_edits(wiki, plan)
            # 结构校验：## 标题数不变
            if _count_headers(new_wiki) != plan["header_count"]:
                print(f"  ✗ 结构校验失败（## 标题数 {plan['header_count']}→{_count_headers(new_wiki)}），跳过写盘")
                continue
            bak_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, bak_dir / path.name)
            path.write_text(new_wiki, encoding="utf-8")
            n_applied += 1
            print(f"  ✅ 已写盘，备份 {bak_dir / path.name}")

    # 写 review.json：为每项分配稳定 id（wiki + 行号哈希，便于网页定位）
    import hashlib as _hl
    for i, r in enumerate(all_review):
        key = f"{r.get('wiki','')}|{r.get('reason','')}|{r.get('line', r.get('wiki_excerpt', r.get('fact','')))[:80]}"
        r["id"] = f"r{i:03d}_{_hl.md5(key.encode(), usedforsecurity=False).hexdigest()[:8]}"
    REVIEW_PATH.write_text(json.dumps(all_review, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(plans, ensure_ascii=False, indent=2))
        return 0

    total_edits = sum(len(p["edits"]) for p in plans)
    total_review = len(all_review)
    print(f"\n=== 汇总 ===")
    print(f"wiki: {len(plans)}，自动编辑: {total_edits}，送 review: {total_review}")
    if args.apply:
        print(f"已写盘: {n_applied} 个 wiki，备份目录: {bak_dir}")
    else:
        print("（dry-run，未写盘。加 --apply 执行）")
    print(f"review 明细: {REVIEW_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
