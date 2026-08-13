#!/usr/bin/env python3
"""Wiki 幻觉校验脚本：对照原始对话，检查 wiki 事实是否有支撑。

用法:
    # 校验单个 wiki
    python scripts/audit_wiki.py --user 示例用户甲

    # 校验全部 wiki（后台分批跑）
    python scripts/audit_wiki.py --all --limit 50

    # 只列出有 prompt 备份的 wiki（不调 LLM）
    python scripts/audit_wiki.py --list

校验结果存到 data/memory/wiki_audit/<name>.json，admin 后台 /wiki-audit 页面展示。
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.qwen_client import QwenClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_logger = logging.getLogger("audit_wiki")

WIKI_DIR = PROJECT_ROOT / "data" / "memory" / "wiki"
PROMPT_DIR = WIKI_DIR / "prompts"
AUDIT_DIR = PROJECT_ROOT / "data" / "memory" / "wiki_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# 校验 prompt：让 LLM 对照历史消息，逐条检查 wiki 事实是否有支撑
_AUDIT_PROMPT = """你是一个事实核查员。下面是一段用户 wiki 和从历史聊天记录中检索到的相关片段。
请检查 wiki 中的【关系和身份事实】（姓名、亲属关系、同学/同事关系、学校、职业、城市、别名等），
判断每条事实在历史消息中是否有明确支撑。

判定标准（严格）：
- confirmed：历史消息里有【用户本人或群里其他人】明确说过这个事实
- unverified：消息里没提到，或只是 Bot 推测/用户未回应，无法确认
- contradicted：消息里【用户或他人】说了相反的内容（最严重的幻觉）

关键规则（严格遵守）：
1. **证据标记解读**：历史片段每行带 [谁][何时] 标记。
   - 👤真人说的（is_self=False，如 👤示例用户甲、👤示例用户午）= 最可信，confirmed
   - 🤖Bot 说的（is_self=True）= Bot 以用户身份自述，**等于用户授权说的，可信**（confirmed），
     因为 Bot 是用户的分身/小号
   - 但若 wiki 写的是"Bot推测/未否认"，而历史里 Bot 只是随口一提且用户未参与，判 unverified
2. **零命中 = 可能编造**：若某事实的关键数字/人名在历史片段里完全找不到（证据段标注"无匹配"），
   很可能是 LLM 编造 → 判 unverified（如历史里没人说"首付287万"，wiki 却写了）。
3. **严禁编造证据（最重要）**：evidence 字段必须**逐字复制**上面【历史聊天检索片段】里真实存在的
   某一行原文（含 [谁][何时] 标记）。绝对禁止改写、拼接、概括、或编造不在片段里的内容/出处。
   - 若该事实在片段里找不到任何相关原文 → 判 unverified，evidence 填"检索片段未提及"
   - 若片段里有明确相反的原话 → 判 contradicted，evidence 逐字复制那行原文
   - 宁可判 unverified，绝不编造 contradicted 证据。不确定就 unverified。
4. **contradicted 极严**：只有当片段里【同一话题】有用户/他人明确说了相反内容时才判 contradicted。
   禁止把不同话题/不同时间的无关原话当作反证（如 A 话题的事实不能用 B 话题的"你自己编吧"反证）。
5. **别名整条算 1 条事实**：wiki 的"别名"行（如"别名：A、B、C"）整行作为一条事实
   （"别名集合是否被对话支撑"），不要逐个别名展开成多条。
6. 只检查关系和身份事实，不检查"近期动态"、"说过的话"。
7. 历史片段可能不完整。片段里有支撑就 confirmed，没提到但不一定错就 unverified，
   片段里明确说了相反内容才 contradicted。

【wiki 内容】
{wiki}

【历史聊天检索片段】
{evidence}

【输出格式】
只输出 JSON 数组，每个元素：
{{
  "fact": "事实内容简述",
  "wiki_excerpt": "wiki 中对应的原文片段（只摘录关键事实，不超过50字，不要复制整行）",
  "verdict": "confirmed|unverified|contradicted",
  "evidence": "从上面【历史聊天检索片段】逐字复制的支撑原文行（含[谁][何时]标记）；unverified 填'检索片段未提及'。禁止编造不在片段里的内容"
}}
不要输出 JSON 以外的任何文字。"""


def _safe_parse_json_array(json_str: str) -> Optional[List[Dict]]:
    """容错解析 LLM 输出的 JSON 数组。

    LLM 经常在 evidence 字段里放未转义的引号导致整体 JSON 解析失败。
    策略：用正则逐个提取 {} 对象，每个对象单独解析（失败则跳过）。
    """
    facts = []
    # 找所有 {...} 对象
    depth = 0
    start = -1
    for i, ch in enumerate(json_str):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                obj_str = json_str[start:i + 1]
                try:
                    obj = json.loads(obj_str)
                    facts.append(obj)
                except json.JSONDecodeError:
                    # 尝试修复：把值里的双引号替换成单引号
                    fixed = _fix_inner_quotes(obj_str)
                    try:
                        obj = json.loads(fixed)
                        facts.append(obj)
                    except json.JSONDecodeError:
                        _logger.debug("跳过无法解析的对象: %s", obj_str[:80])
                start = -1
    return facts if facts else None


def _fix_inner_quotes(obj_str: str) -> str:
    """修复 JSON 对象内值字段的未转义双引号。

    策略：用正则匹配 "key": "value" 模式，把 value 内的双引号替换成单引号。
    """
    # 匹配 "key": "..." 模式，value 里可能有未转义引号
    def fix_value(m):
        key = m.group(1)
        # value 是从第一个引号到最后一个引号之间的内容
        raw_val = m.group(2)
        # 把 value 内部的双引号替换成单引号
        fixed_val = raw_val.replace('"', "'")
        return f'"{key}": "{fixed_val}"'

    # 这个正则匹配 "key": "value" 其中 value 不含换行
    return re.sub(r'"(\w+)":\s*"([^"]*(?:"[^"]*)*)"', fix_value, obj_str)


def _load_wiki(name: str) -> str:
    path = WIKI_DIR / "users" / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"wiki 不存在: {path}")
    return path.read_text(encoding="utf-8")


def _extract_relation_keywords(wiki: str) -> List[str]:
    """从 wiki 提取关系/身份关键词，用于检索历史消息。"""
    keywords = set()
    # 提取人名（**粗体** 标记）
    for m in re.finditer(r"\*\*(.+?)\*\*", wiki):
        name = m.group(1).strip()
        if 2 <= len(name) <= 8 and not any(c in name for c in "：（）"):
            keywords.add(name)
    # 提取"关系：人名"模式，提取关系词和人名分别作为检索词
    for m in re.finditer(r"(二姨|小舅|大舅|大姨|二舅|表哥|表姐|母亲|父亲|配偶|老婆|老公|同学|同事)", wiki):
        keywords.add(m.group(1))
    # 提取"：人名（"模式里的人名
    for m in re.finditer(r"[：:]\s*([^\s（(]{2,8})[（(]", wiki):
        name = m.group(1).strip()
        if not name.replace("，", "").isdigit():
            keywords.add(name)
    # 提取群昵称（中文引号或英文引号内）
    for m in re.finditer(r'["\u201c]([^"\u201d]{2,10})["\u201d]', wiki):
        nick = m.group(1).strip()
        if not any(c in nick for c in "，。：；！？"):
            keywords.add(nick)
    return list(keywords)[:15]


def _extract_fact_keywords(wiki: str) -> List[str]:
    """从 wiki 提取用于核查的关键词：人名、数字、昵称、关系词。

    每个关键词单独去历史里搜，判定该事实有没有人说过。
    只取短词（2-8 字），不取整句，避免长句匹配不到。
    """
    kws: List[str] = []
    seen = set()

    def add(s: str) -> None:
        s = s.strip()
        if s and 2 <= len(s) <= 12 and s not in seen:
            seen.add(s)
            kws.append(s)

    # 人名（**粗体** 标记）
    for m in re.finditer(r"\*\*(.+?)\*\*", wiki):
        name = m.group(1).strip()
        if 2 <= len(name) <= 8 and not any(c in name for c in "：（）()， "):
            add(name)
    # 数字事实（金额、年份、尾号等）
    for m in re.finditer(r"\d{4,}|\d+\.?\d*万|\d+\.?\d*元", wiki):
        add(m.group(0))
    # 引号内昵称/群名（仅短串 ≤6 字，长句是原话引用不入选，留给专有词扫描）
    for m in re.finditer(r"[“\"]([^”“\"]{2,6})[”\"]", wiki):
        nick = m.group(1).strip()
        if not any(c in nick for c in "，。：；！？ "):
            add(nick)
    # 关系行里的人名："- XX：YY" 取 XX（关系词）和 YY 开头的人名
    for m in re.finditer(r"^[\s-]*\*{0,2}([一-龥]{2,6})\*{0,2}[：:]", wiki, re.M):
        add(m.group(1))
    # 显著机构/地名/专有词：2-5 字连续中文，去掉来源标注行后再抽
    wiki_clean = re.sub(r"（来源：[^）]*）", "", wiki)  # 去来源标注
    wiki_clean = re.sub(r"[“”\"][^”\"]{2,40}[”\"]", "", wiki_clean)  # 去引号原话
    for m in re.finditer(r"(?<![一-龥])([一-龥]{2,5})(?![一-龥])", wiki_clean):
        w = m.group(1)
        # 排除关系词/常见词
        if w in {"本人", "对方", "成员", "暂无", "无其他", "来源", "动态", "关系", "信息", "备注",
                 "配偶", "父亲", "母亲", "老公", "老婆", "儿子", "女儿", "基本", "别名", "昵称"}:
            continue
        add(w)
    return kws[:30]


def _gather_evidence(name: str, wiki: str) -> str:
    """用全量历史 pkl 检索证据（关键词精确匹配，带 is_self 标记谁说的）。

    每个关键词查 top 命中，合并去重。证据带 [谁][何时] 标记，
    is_self=True（Bot 自述）= 用户授权 Bot 说的，可信；零命中 = 历史里没人说过。
    """
    try:
        from src.memory.history_lookup import HistoryLookup, format_evidence
    except Exception as e:
        _logger.warning("无法导入 history_lookup: %s，回退 prompt 备份", e)
        return _gather_evidence_from_prompt(name)

    lk = HistoryLookup.get()
    if lk is None:
        _logger.warning("[%s] history_lookup 不可用，回退 prompt 备份", name)
        return _gather_evidence_from_prompt(name)

    kws = _extract_fact_keywords(wiki)
    # 主名相关 chat 限定（私聊 + 含主名的群）
    name_chats = [name] if name else None
    # 数字/金额类关键词（需不限 chat 单搜，跨群捞证据）
    num_kws = [k for k in kws if re.search(r"\d", k)]

    all_hits: List[Dict[str, Any]] = []
    seen_text: set = set()

    # 策略 1：每个非数字关键词在主名相关 chat 里搜（chat_name 含主名 = 该用户私聊/相关群）
    for kw in kws[:12]:
        if re.search(r"\d", kw):
            continue  # 数字类走策略 3
        hits = lk.search([kw], chats=name_chats, limit=3) if name else lk.search([kw], limit=3)
        for h in hits:
            key = (h["who"], h["text"][:60])
            if key not in seen_text:
                seen_text.add(key)
                all_hits.append(h)

    # 策略 2：每个非数字关键词单独搜（不限 chat，捞跨群证据）
    for kw in kws[:12]:
        if re.search(r"\d", kw):
            continue
        hits = lk.search([kw], limit=2)
        for h in hits:
            key = (h["who"], h["text"][:60])
            if key not in seen_text:
                seen_text.add(key)
                all_hits.append(h)

    # 策略 3：数字/金额关键词不限 chat 搜（判定"历史里有没有人说过这个数"）
    zero_match: List[str] = []  # 零命中的数字关键词 → 可能 LLM 编造
    for kw in num_kws:
        hits = lk.search([kw], limit=4)
        if not hits:
            zero_match.append(kw)
        for h in hits:
            key = (h["who"], h["text"][:60])
            if key not in seen_text:
                seen_text.add(key)
                all_hits.append(h)

    # 策略 3：prompt 备份（末次更新对话，作为补充）
    prompt_ev = _gather_evidence_from_prompt(name)
    prompt_block = ""
    if prompt_ev and len(prompt_ev) > 50:
        prompt_block = "【prompt 备份·末次更新对话】\n" + prompt_ev[:2000]

    evidence = format_evidence(all_hits[:40])
    if zero_match:
        evidence += "\n\n【零命中关键词】以下数字/金额在全部 78 万条历史消息中无任何匹配，" \
                    "wiki 中对应事实很可能是 LLM 编造：\n" + "、".join(zero_match)
    if prompt_block:
        evidence = evidence + "\n\n" + prompt_block
    if len(evidence) > 30000:
        evidence = evidence[:15000] + "\n... [中间截断] ...\n" + evidence[-15000:]
    return evidence


def _extract_relation_sections(wiki: str) -> List[str]:
    """从 wiki 提取关系相关段落，作为检索 query。

    把"## 家族关系""## 基本信息""## 社会关系"等 section 的内容
    整段提取，让 search_history 的 embedding 做语义匹配。
    """
    sections = []
    current_section = ""
    current_lines: List[str] = []
    relation_headers = ("家族关系", "基本信息", "社会关系", "关系", "亲属", "同学", "同事")
    for line in wiki.split("\n"):
        if line.startswith("##"):
            # 保存前一个 section
            if current_section and any(h in current_section for h in relation_headers) and current_lines:
                sections.append("\n".join(current_lines[:15]))
            current_section = line.lstrip("#").strip()
            current_lines = []
        elif current_section and any(h in current_section for h in relation_headers):
            current_lines.append(line)
    # 最后一个 section
    if current_section and any(h in current_section for h in relation_headers) and current_lines:
        sections.append("\n".join(current_lines[:15]))
    return sections


def _gather_evidence_from_prompt(name: str) -> str:
    """回退方案：从 prompt 备份提取对话。"""
    path = PROMPT_DIR / "users" / f"{name}.md"
    if not path.exists():
        return ""
    prompt = path.read_text(encoding="utf-8")
    marker = "对话内容：\n"
    idx = prompt.find(marker)
    if idx == -1:
        return prompt[len(prompt) // 2:]
    return prompt[idx + len(marker):]


def _load_internal_contradictions(name: str) -> List[Dict[str, Any]]:
    """从 contradictions.json 读取该 wiki 的 A1 内部昵称碰撞（程序化检测结果）。

    audit 不自算内部一致性（同模型族不可靠），直接复用 detect 脚本的 100% 精度结果。
    """
    path = AUDIT_DIR / "contradictions.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    w = data.get("per_wiki", {}).get(name, {})
    return w.get("internal_contradictions", [])


class _LLM:
    """轻量 LLM 抽象：支持 qwen(DashScope) / deepseek / kimi 三后端，统一 chat() 接口。"""

    def __init__(self, backend: str = "deepseek"):
        from openai import OpenAI
        self.backend = backend
        if backend == "qwen":
            if not os.environ.get("DASHSCOPE_API_KEY"):
                raise RuntimeError("DASHSCOPE_API_KEY 未设置")
            self.client = OpenAI(
                api_key=os.environ["DASHSCOPE_API_KEY"],
                base_url=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            )
            self.model = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")
        elif backend == "kimi":
            # 从 ~/.config/kimi-claude/config.env 加载
            cfg = Path.home() / ".config" / "kimi-claude" / "config.env"
            if cfg.exists() and not os.environ.get("KIMI_API_KEY"):
                for line in cfg.read_text(encoding="utf-8").splitlines():
                    if "=" in line and not line.strip().startswith("#"):
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            if not os.environ.get("KIMI_API_KEY"):
                raise RuntimeError("KIMI_API_KEY 未设置")
            self.client = OpenAI(
                api_key=os.environ["KIMI_API_KEY"],
                base_url=os.environ.get("KIMI_API_BASE", "https://api.kimi.com/coding/v1"),
            )
            self.model = os.environ.get("KIMI_MODEL", "kimi-for-coding")
        else:
            q = QwenClient()
            self.client = q.client
            self.model = q.model

    def chat(self, messages, temperature=0.0, max_tokens=4096, timeout=60):
        r = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        return r.choices[0].message.content or ""


def audit_wiki(name: str, llm) -> Dict[str, Any]:
    """校验单个 wiki，返回校验结果 dict。"""
    wiki = _load_wiki(name)

    _logger.info("[%s] 检索历史消息作为校验依据...", name)
    evidence = _gather_evidence(name, wiki)
    if not evidence:
        _logger.warning("[%s] 无法获取历史消息，跳过", name)
        return {"name": name, "status": "no_evidence", "facts": [],
                "internal_contradictions": _load_internal_contradictions(name)}

    _logger.info("[%s] 获取到 %d 字符历史消息，调用 LLM 校验...", name, len(evidence))
    audit_prompt = _AUDIT_PROMPT.format(wiki=wiki[:6000], evidence=evidence)

    try:
        response = llm.chat(
            messages=[{"role": "user", "content": audit_prompt}],
            temperature=0.0,
            max_tokens=8000,
        )
        raw = response if isinstance(response, str) else getattr(response, "content", str(response))
    except Exception as e:
        _logger.error("[%s] LLM 调用失败: %s", name, e)
        return {"name": name, "status": "llm_error", "error": str(e), "facts": [],
                "internal_contradictions": _load_internal_contradictions(name)}

    # 解析 JSON：LLM 可能有 thinking 前缀或 markdown 包裹
    raw = raw.strip()
    # 去掉 thinking 前缀（如果有 <think>...</think> 或纯文本分析）
    json_start = raw.find("[")
    json_end = raw.rfind("]")
    if json_start == -1 or json_end == -1:
        _logger.error("[%s] 未找到 JSON 数组", name)
        return {"name": name, "status": "parse_error", "raw": raw[:800], "facts": [],
                "internal_contradictions": _load_internal_contradictions(name)}

    json_str = raw[json_start:json_end + 1]

    # 修复 LLM 常见的 JSON 格式问题：evidence 字段里的未转义引号
    # 策略：逐字符扫描，对值字段内的双引号转义
    facts = _safe_parse_json_array(json_str)
    if facts is None:
        _logger.error("[%s] JSON 解析失败", name)
        return {"name": name, "status": "parse_error", "raw": json_str[:800], "facts": [],
                "internal_contradictions": _load_internal_contradictions(name)}

    # 幻觉护栏：校验每条 fact 的 evidence 是否真实存在于检索证据里。
    # LLM 可能编造证据出处（如把无关原话接到某事实上）。evidence 里若含 [谁][何时] 标记但
    # 该标记+原话不在 evidence 段中，标 hallucinated=True，verdict 降级 unverified。
    ev_lower = evidence
    for f in facts:
        ev = f.get("evidence", "")
        if ev and ev != "检索片段未提及":
            # 取 evidence 字段里的核心原话（引号内或括号后内容）做子串检查
            # 简化：若 evidence 字段含某 8+ 字连续串不在原始证据段里，判幻觉
            core = re.sub(r"\[.*?\]", "", ev).strip("'""' ")
            if len(core) >= 6 and core not in ev_lower:
                f["hallucinated_evidence"] = True
                if f.get("verdict") == "contradicted":
                    f["verdict"] = "unverified"
                    f["evidence"] = "（原判 contradicted 证据经核对不在检索片段中，降级 unverified）"
            else:
                f["hallucinated_evidence"] = False

    result = {
        "name": name,
        "status": "ok",
        "audited_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wiki_chars": len(wiki),
        "evidence_chars": len(evidence),
        "facts": facts,
        "raw_evidence": evidence[:8000],  # 原始检索片段，供网页展示核对
        "internal_contradictions": _load_internal_contradictions(name),
    }

    confirmed = sum(1 for f in facts if f.get("verdict") == "confirmed")
    unverified = sum(1 for f in facts if f.get("verdict") == "unverified")
    contradicted = sum(1 for f in facts if f.get("verdict") == "contradicted")
    result["summary"] = {
        "total": len(facts),
        "confirmed": confirmed,
        "unverified": unverified,
        "contradicted": contradicted,
    }
    _logger.info("[%s] 校验完成: %d confirmed, %d unverified, %d contradicted",
                 name, confirmed, unverified, contradicted)

    out_path = AUDIT_DIR / f"{name}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def list_auditable() -> List[str]:
    """列出有 prompt 备份的 wiki 名称。"""
    prompt_dir = PROMPT_DIR / "users"
    if not prompt_dir.exists():
        return []
    names = []
    for p in sorted(prompt_dir.glob("*.md")):
        wiki_path = WIKI_DIR / "users" / p.name
        if wiki_path.exists():
            names.append(p.stem)
    return names


def list_audited() -> List[str]:
    """列出已校验的 wiki 名称。"""
    return sorted(p.stem for p in AUDIT_DIR.glob("*.json"))


def main():
    parser = argparse.ArgumentParser(description="Wiki 幻觉校验")
    parser.add_argument("--user", help="校验指定用户 wiki")
    parser.add_argument("--all", action="store_true", help="校验全部（有 prompt 备份的）")
    parser.add_argument("--limit", type=int, default=0, help="--all 时最多校验 N 个（0=不限）")
    parser.add_argument("--list", action="store_true", help="列出可校验的 wiki")
    parser.add_argument("--sleep", type=float, default=0.5, help="每次调用间隔秒数")
    parser.add_argument("--backend", choices=["deepseek", "qwen", "kimi"], default="qwen",
                        help="LLM 后端（默认 qwen/DashScope，deepseek 余额易不足）")
    args = parser.parse_args()

    if args.list:
        names = list_auditable()
        audited = set(list_audited())
        print(f"可校验 wiki: {len(names)} 个")
        for n in names:
            mark = " ✅已校验" if n in audited else ""
            print(f"  {n}{mark}")
        return

    try:
        llm = _LLM(args.backend)
        print(f"LLM 后端: {args.backend} (model={llm.model})")
    except Exception as e:
        print(f"错误: LLM 后端初始化失败: {e}", file=sys.stderr)
        sys.exit(1)

    if args.user:
        result = audit_wiki(args.user, llm)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.all:
        names = list_auditable()
        audited = set(list_audited())
        # 跳过已校验的
        todo = [n for n in names if n not in audited]
        if args.limit > 0:
            todo = todo[:args.limit]
        print(f"待校验: {len(todo)} 个（共 {len(names)} 个，已校验 {len(audited)} 个）")
        for i, name in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {name}")
            try:
                audit_wiki(name, llm)
            except Exception as e:
                _logger.error("[%s] 校验异常: %s", name, e)
            if args.sleep > 0:
                time.sleep(args.sleep)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
