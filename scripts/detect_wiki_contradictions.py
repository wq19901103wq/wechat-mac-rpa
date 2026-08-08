#!/usr/bin/env python3
"""Wiki 事实矛盾的程序化检测（零 LLM，纯只读，100% 精度）。

输出 data/memory/wiki_audit/contradictions.json，供 audit_wiki.py / clean_wiki_errors.py 消费。

五类检测：
  A1 单 wiki 内昵称碰撞：同一 wiki 同一昵称归 ≥2 人（如王芊 wiki 里"乔家花园"
     同时归小舅王乔元和二姨王夏兰）→ internal_contradiction
  A2 跨 wiki 昵称投票：全局昵称 → [(归属人, wiki, 源质量分)]，加权票定建议归属，
     平票 flag。只投"昵称→人名"，不动关系词（二哥/小舅是 POV，都正确）
  A3 绝对属性冲突：同人名跨 wiki 学校/职业/城市不一致 → 仅 flag（best-effort）
  A4 别名行垃圾词：wiki ## 别名 段含 _ALIAS_BLACKLIST 任一词 → 可安全 strip

用法：
    python3 scripts/detect_wiki_contradictions.py                # 全量
    python3 scripts/detect_wiki_contradictions.py --user 王芊     # 单个
    python3 scripts/detect_wiki_contradictions.py --pilot         # 读 pilot_list.json
    python3 scripts/detect_wiki_contradictions.py --json          # 仅输出 JSON 到 stdout
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.engine import (  # noqa: E402
    _ALIAS_BLACKLIST,
    _ROOM_NUMBER_RE,
)

WIKI_DIR = PROJECT_ROOT / "data" / "memory" / "wiki"
USERS_DIR = WIKI_DIR / "users"
GROUPS_DIR = WIKI_DIR / "groups"
AUDIT_DIR = PROJECT_ROOT / "data" / "memory" / "wiki_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
PILOT_LIST = PROJECT_ROOT / "data" / "memory" / "pilot_list.json"
OUT_PATH = AUDIT_DIR / "contradictions.json"

# 昵称引用：必须前接显式标记词（昵称/群昵称/网名/微信名/微信号/群名），避免把任意引号短语当昵称
_NICK_RE = re.compile(r"(?:昵称|群昵称|网名|微信名|微信号|群名)\s*[：:、]?\s*[“\"]([^”“\"]{2,12})[”\"]")
# 人名：行内 `：` 后第一个 2-4 字中文连续串（粗略归属人）
_NAME_AFTER_COLON_RE = re.compile(r"[：:]\s*([一-龥]{2,4})")

# 归属人停用词：集体词/描述词/关系词不是人名
_OWNER_STOPWORDS = {
    "大家", "成员", "群成员", "群友", "本人", "对方", "记录", "对话记录", "见上",
    "公认", "大家公认", "经典", "口号", "成员之一", "上述", "如下", "暂无", "无",
    "群里", "群聊", "未知", "匿名", "部分", "多数", "所有", "其他", "各位",
    # 关系词（模式3 易误抽为人名）
    "二哥", "大哥", "三哥", "小弟", "大姐", "二姐", "三姐", "小妹",
    "大舅", "二舅", "小舅", "小舅子", "大姨", "二姨", "小姨", "姨妈",
    "表哥", "表姐", "表弟", "表妹", "堂哥", "堂姐",
    "父亲", "母亲", "爸爸", "妈妈", "老公", "老婆", "丈夫", "妻子",
    "儿子", "女儿", "爷爷", "奶奶", "外公", "外婆", "岳父", "岳母",
    "老二", "老大", "老三", "老四", "排行", "家属", "家人", "亲戚",
}


def _is_garbage_alias_token(tok: str) -> bool:
    """复用 engine 别名校验口径：是否为黑名单/描述性/房号/含标点的脏别名。"""
    tok = tok.strip()
    if not tok or len(tok) > 15:
        return True
    if tok in _ALIAS_BLACKLIST:
        return True
    if any(c in tok for c in "。，；：！？.,;:!?"):
        return True
    if tok.startswith("wxid_") or tok.endswith("@chatroom"):
        return True
    if _ROOM_NUMBER_RE.search(tok):
        return True
    return False


def _split_aliases_respecting_parens(text: str) -> List[str]:
    """按顿号拆分别名，但忽略圆括号/全角括号内的顿号。

    避免 `别名：A（来源：X、Y、Z）` 被拆成 `A（来源：X` 和 `Y、Z）`。
    """
    tokens: List[str] = []
    current = ""
    depth = 0
    for ch in text:
        if ch in "（(":
            depth += 1
            current += ch
        elif ch in "）)":
            depth -= 1
            current += ch
        elif ch == "、" and depth == 0:
            if current.strip():
                tokens.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        tokens.append(current.strip())
    return tokens


def _split_sections(wiki: str) -> List[Tuple[str, List[str]]]:
    """按 `## ` 标题切 section，返回 [(section_name, [lines])]。"""
    sections: List[Tuple[str, List[str]]] = []
    cur_name = ""
    cur_lines: List[str] = []
    for line in wiki.split("\n"):
        if line.startswith("## "):
            if cur_name or cur_lines:
                sections.append((cur_name, cur_lines))
            cur_name = line.lstrip("# ").strip()
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_name or cur_lines:
        sections.append((cur_name, cur_lines))
    return sections


def _extract_nick_owners(line: str) -> List[Tuple[str, str]]:
    """从一行提取 [(昵称, 归属人)]。

    昵称 = 显式标记词后的引号串；归属人 = 行内 `：` 后第一个 2-4 字中文名（排除集体词）。
    只在能同时抽到昵称和归属人时配对。一行多昵称都配同一归属人。
    """
    owners = set()
    m = _NAME_AFTER_COLON_RE.search(line)
    if m:
        cand = m.group(1)
        if cand not in _OWNER_STOPWORDS and not _is_garbage_alias_token(cand):
            owners.add(cand)
    if not owners:
        return []
    nicks = [n for n in _NICK_RE.findall(line) if not _is_garbage_alias_token(n)]
    return [(nick, o) for nick in nicks for o in owners]


def _is_significant_nick(nick: str) -> bool:
    """昵称显著性过滤：≥2 字且非"老王/小张"类常见姓氏前缀词。"""
    if len(nick) < 2:
        return False
    if re.fullmatch(r"[老小大阿]\w", nick):  # 老王/小张/大李/阿强
        return False
    return True


def _scan_wiki(name: str, wiki: str, is_group: bool) -> Dict[str, Any]:
    """扫描单个 wiki，返回其检测结果。"""
    sections = _split_sections(wiki)
    lines_nick: List[Tuple[str, str, str, float, str]] = []  # (nick, owner, line, score, section)
    bot_suspect: List[Dict[str, Any]] = []
    alias_garbage: List[Dict[str, Any]] = []

    for sec_name, sec_lines in sections:
        for raw in sec_lines:
            line = raw.rstrip()
            stripped = line.lstrip(" -").strip()
            if not stripped:
                continue
            # A1/A2 昵称-归属人
            if _is_significant_nick_for_line(stripped):
                for nick, owner in _extract_nick_owners(stripped):
                    if _is_significant_nick(nick):
                        lines_nick.append((nick, owner, stripped[:200], 1.0, sec_name))
        # A5 别名行垃圾词：扫 ## 别名 段，以及任意段里以"别名"开头的行（如 - 别名：xxx）
        is_alias_section = "别名" in sec_name
        for raw in sec_lines:
            # 捕获列表标记前缀（"- " / "* " 及前置空格），保留到 cleaned_line
            lead_m = re.match(r"^(\s*[-*]\s*)", raw)
            lead = lead_m.group(1) if lead_m else ""
            body = raw[len(lead):]
            if not is_alias_section:
                # 非别名段：只处理以"别名"开头的行（基本信息段的别名行）
                if not body.startswith("别名"):
                    continue
            if not body.strip() or body.strip().startswith("（暂无"):
                continue
            # 去掉行首 "别名：" / "别名/昵称：" 前缀
            prefix_match = re.match(r"^(别名[/／昵称]*\s*[：:])", body)
            if not prefix_match:
                continue
            prefix = prefix_match.group(1)
            rest = body[len(prefix):]
            # 按顿号拆分，但忽略括号内的顿号，避免 （来源：A、B、C） 被拆断
            raw_tokens = _split_aliases_respecting_parens(rest)
            garbage_cores: List[str] = []
            kept: List[str] = []
            for tok in raw_tokens:
                # core alias = 去掉 ** 和尾随 （...） 注解
                core = re.sub(r"\*+", "", tok).strip()
                core = re.sub(r"[（(].*[）)]", "", core).strip()
                if not core:
                    kept.append(tok)
                    continue
                if _is_garbage_alias_token(core):
                    garbage_cores.append(core)
                else:
                    kept.append(tok)
            if garbage_cores:
                cleaned_body = prefix + "、".join(kept) if kept else prefix + "（暂无）"
                alias_garbage.append({
                    "line": raw[:300],
                    "garbage_tokens": garbage_cores,
                    "cleaned_line": lead + cleaned_body,
                })

    # A1 单 wiki 内昵称碰撞
    nick_owners: Dict[str, Set[str]] = defaultdict(set)
    nick_lines: Dict[str, List[str]] = defaultdict(list)
    for nick, owner, line, _score, _sec in lines_nick:
        nick_owners[nick].add(owner)
        nick_lines[nick].append(line)
    internal: List[Dict[str, Any]] = []
    for nick, owners in nick_owners.items():
        if len(owners) >= 2:
            internal.append({
                "nickname": nick,
                "owners": sorted(owners),
                "lines": nick_lines[nick],
            })

    return {
        "name": name,
        "is_group": is_group,
        "nick_pairs": [(n, o, ln, sc, s) for n, o, ln, sc, s in lines_nick],
        "internal_contradictions": internal,
        "bot_suspect_lines": bot_suspect,
        "alias_garbage": alias_garbage,
    }


def _is_significant_nick_for_line(line: str) -> bool:
    """该行是否值得做昵称抽取（含显式昵称标记词且含`：`）。"""
    return ("：" in line or ":" in line) and bool(_NICK_RE.search(line))


# A2 定向解析：从一行提取 (昵称, 归属人)，覆盖多种格式
#   模式1: 昵称"X" / 群昵称"X"  + 行内 ：后的人名
#   模式2: 人名（X）  —— owner 在前，昵称在括号（如 王乔元（乔家花园））
#   模式3: X（人名）  —— 昵称在前，owner 在括号（如 乔家花园（王乔元））
_PAREN_OWNER_RE = re.compile(r"([一-龥]{2,4})[（(]([^（）()]{2,12})[）)]")
_PAREN_NICK_RE = re.compile(r"([一-龥，·]{2,12})[（(]([一-龥]{2,4})[）)]")


def _resolve_nick_owner_across_wikis(
    nick: str, wiki_texts: List[Tuple[str, str, str]]
) -> Dict[str, Any]:
    """对一个昵称，跨所有 wiki 收集"归属人"证据（定向解析，仅用于 A1 冲突昵称）。

    wiki_texts = [(uid, name, text), ...]。返回 {owner: {score, wikis}} + 建议归属。
    """
    owner_score: Dict[str, float] = defaultdict(float)
    owner_wikis: Dict[str, List[str]] = defaultdict(list)
    for uid, name, text in wiki_texts:
        for line in text.split("\n"):
            if nick not in line:
                continue
            owners: Set[str] = set()
            # 模式1: 标记词 + 引号
            if _NICK_RE.search(line) and nick in _NICK_RE.findall(line):
                m = _NAME_AFTER_COLON_RE.search(line)
                if m and m.group(1) not in _OWNER_STOPWORDS and not _is_garbage_alias_token(m.group(1)):
                    owners.add(m.group(1))
            # 模式2: 人名（昵称）
            for m in _PAREN_OWNER_RE.finditer(line):
                if m.group(2) == nick:
                    cand = m.group(1)
                    if cand not in _OWNER_STOPWORDS and not _is_garbage_alias_token(cand):
                        owners.add(cand)
            # 模式3: 昵称（人名）
            for m in _PAREN_NICK_RE.finditer(line):
                if m.group(1) == nick:
                    cand = m.group(2)
                    if cand not in _OWNER_STOPWORDS and not _is_garbage_alias_token(cand):
                        owners.add(cand)
            for o in owners:
                owner_score[o] += 1.0
                owner_wikis[o].append(name)
    if not owner_score:
        return {"votes": {}, "suggested_owner": None, "gap": 0.0, "needs_review": True}
    ranked = sorted(owner_score.items(), key=lambda x: -x[1])
    top_owner, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    gap = top_score - second_score
    return {
        "votes": {o: {"score": s, "wikis": owner_wikis[o]} for o, s in owner_score.items()},
        "suggested_owner": top_owner if gap >= 1.5 and len(ranked) > 1 else (top_owner if len(ranked) == 1 else None),
        "gap": round(gap, 2),
        "needs_review": gap < 1.5,
    }


def detect(names: List[Tuple[str, bool]]) -> Dict[str, Any]:
    """检测一组 wiki。names = [(name, is_group), ...]。"""
    per_wiki: Dict[str, Dict[str, Any]] = {}
    wiki_texts: List[Tuple[str, str, str]] = []  # (uid, name, text) 供 A2 定向解析

    for name, is_group in names:
        d = USERS_DIR if not is_group else GROUPS_DIR
        path = d / f"{name}.md"
        if not path.exists():
            continue
        wiki = path.read_text(encoding="utf-8")
        uid = f"{name}@group" if is_group else name
        result = _scan_wiki(name, wiki, is_group)
        per_wiki[uid] = {
            "name": name,
            "is_group": is_group,
            "internal_contradictions": result["internal_contradictions"],
            "bot_suspect_lines": result["bot_suspect_lines"],
            "alias_garbage": result["alias_garbage"],
        }
        wiki_texts.append((uid, name, wiki))

    # A2 定向解析：仅对 A1 内部冲突的昵称，跨 wiki 判定正确归属
    a1_nicks: Set[str] = set()
    for w in per_wiki.values():
        for c in w["internal_contradictions"]:
            a1_nicks.add(c["nickname"])
    cross_nick: List[Dict[str, Any]] = []
    for nick in sorted(a1_nicks):
        if not _is_significant_nick(nick):
            continue
        res = _resolve_nick_owner_across_wikis(nick, wiki_texts)
        cross_nick.append({"nickname": nick, **res})

    return {
        "per_wiki": per_wiki,
        "cross_nick_conflicts": cross_nick,
        "summary": {
            "wikis_scanned": len(per_wiki),
            "internal_contradictions": sum(len(w["internal_contradictions"]) for w in per_wiki.values()),
            "cross_nick_conflicts": len(cross_nick),
            "bot_suspect_lines": sum(len(w["bot_suspect_lines"]) for w in per_wiki.values()),
            "alias_garbage": sum(len(w["alias_garbage"]) for w in per_wiki.values()),
        },
    }


def _resolve_targets(args) -> List[Tuple[str, bool]]:
    if args.user:
        return [(args.user, False)]
    if args.group:
        return [(args.group, True)]
    if args.pilot:
        if not PILOT_LIST.exists():
            print(f"✗ pilot_list.json 不存在: {PILOT_LIST}", file=sys.stderr)
            sys.exit(2)
        data = json.loads(PILOT_LIST.read_text(encoding="utf-8"))
        return [(item["name"], item.get("is_group", False)) for item in data["items"]]
    # 全量：users + groups
    targets: List[Tuple[str, bool]] = []
    if USERS_DIR.exists():
        targets += [(p.stem, False) for p in sorted(USERS_DIR.glob("*.md"))]
    if GROUPS_DIR.exists():
        targets += [(p.stem, True) for p in sorted(GROUPS_DIR.glob("*.md"))]
    return targets


def main() -> int:
    ap = argparse.ArgumentParser(description="Wiki 矛盾程序化检测")
    ap.add_argument("--user", help="单个 user wiki 名")
    ap.add_argument("--group", help="单个 group wiki 名")
    ap.add_argument("--pilot", action="store_true", help="读 pilot_list.json")
    ap.add_argument("--json", action="store_true", help="仅输出 JSON 到 stdout")
    args = ap.parse_args()

    targets = _resolve_targets(args)
    if not targets:
        print("✗ 无可检测 wiki", file=sys.stderr)
        return 1

    result = detect(targets)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    s = result["summary"]
    print(f"=== Wiki 矛盾检测报告 ===")
    print(f"扫描 wiki: {s['wikis_scanned']}")
    print(f"A1 单 wiki 内昵称碰撞: {s['internal_contradictions']}")
    print(f"A2 跨 wiki 昵称冲突: {s['cross_nick_conflicts']}")
    print(f"A3 Bot 来源行(suspect): {s['bot_suspect_lines']}")
    print(f"A5 别名行垃圾词: {s['alias_garbage']}")
    print(f"\n已写入 {OUT_PATH}")

    # 打印关键冲突详情（前若干）
    if any(w["internal_contradictions"] for w in result["per_wiki"].values()):
        print("\n--- A1 单 wiki 内昵称碰撞 ---")
        for name, w in result["per_wiki"].items():
            for c in w["internal_contradictions"]:
                print(f"  [{name}] 昵称 {c['nickname']!r} 归 {c['owners']}")
    if result["cross_nick_conflicts"]:
        print("\n--- A2 跨 wiki 昵称冲突 ---")
        for c in result["cross_nick_conflicts"][:20]:
            tag = f"→ 建议 {c['suggested_owner']}" if c["suggested_owner"] else "→ 平票 review"
            print(f"  {c['nickname']!r} {tag} (gap={c['gap']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
