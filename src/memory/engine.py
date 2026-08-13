#!/usr/bin/env python3
"""Memory Engine - LLM Wiki based long-term memory with overrides support."""

import json
import logging
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from httpx import Timeout
except ImportError:
    Timeout: Any = None  # type: ignore[misc,assignment,no-redef]

from src.memory.evidence import (
    format_evidence_conversation,
    strip_unverified_lines,
)
from src.memory.wiki_prompts import RUNTIME_DEFAULT_GROUP_WIKI, RUNTIME_UPDATE_GROUP_PROMPT
from src.utils.chat_utils import _safe_filename

_logger = logging.getLogger("src.memory.engine")

# ── 别名校验（防止把非别名噪声写进 aliases.json）──

# 角色词 / 系统占位符：明显不是真人昵称，一律拒绝
_ALIAS_BLACKLIST = {
    "Bot", "bot", "我", "对方", "对话中", "匿名", "未知昵称", "未知", "群主", "群聊主人",
    "群成员", "记录者", "记录人", "旁白", "示例交流群", "本人", "自己", "他人", "某人",
    "无", "暂无", "未发现", "未发现其他显著别名",
}

# 房号 / 单元号模式：如 "6幢5号501"、"4-1-703"、"1幢10号802"
_ROOM_NUMBER_RE = re.compile(r"\d+\s*[幢栋号楼室单元]\s*\d|\d+-\d+-\d+|\d+幢\d+号")

# 别名长度上限：真实昵称不会太长
_ALIAS_MAX_LEN = 15

# 别名拆分符：顿号 / 斜杠 / 空格（拆完后逐条校验）
_ALIAS_SPLIT_RE = re.compile(r"[、/／\|｜\s]+")


def _split_alias_tokens(text: str) -> List[str]:
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


# 常见 emoji Unicode 码点集合（用于群名归一化剥离）
_EMOJI_CODEPOINTS: frozenset[int] = frozenset(
    [0x24C2]
    + list(range(0x2702, 0x27B0 + 1))
    + list(range(0x1F1E6, 0x1F1FF + 1))
    + list(range(0x1F300, 0x1F5FF + 1))
    + list(range(0x1F600, 0x1F64F + 1))
    + list(range(0x1F680, 0x1F6FF + 1))
    + list(range(0x1F900, 0x1F9FF + 1))
    + list(range(0x1FA00, 0x1FAFF + 1))
)


def _strip_emoji(name: str) -> str:
    """去除常见 emoji 字符。"""
    return "".join(c for c in name if ord(c) not in _EMOJI_CODEPOINTS)


def normalize_chat_name(name: str) -> str:
    """群名/用户名归一化（FR-13）。

    用于 wiki 路径计算与去重，避免 OCR 空格/前缀差异导致的重复群：
    - 去除 emoji
    - 折叠连续空白
    - 去除首尾空白与常见前缀序号（"3D " / "1" 等数字前缀不剥离，避免误并）

    注意：归一化只用于"判断是否重复"和"路径"，不改变显示名。
    """
    if not name:
        return ""
    n = _strip_emoji(name)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _split_alias_string(s: str) -> List[str]:
    """把 LLM 输出的别名串按顿号/斜杠/空格拆成单条，去重保序。

    LLM 经常把多个别名写成 "老王、王总" 或 "Paul、阿杰" 一整串，
    不拆分会导致整串入库、按整串匹配，召回失败。
    """
    parts = _ALIAS_SPLIT_RE.split(s)
    seen: set = set()
    result: List[str] = []
    for p in parts:
        p = p.strip().strip("（）()「」『』“”\"'").strip()
        if not p or p in seen:
            continue
        seen.add(p)
        result.append(p)
    return result


# 默认 wiki 模板
_DEFAULT_USER_WIKI = """# {user_name}

## 基本信息
- 姓名/本名：
- 别名/昵称：
- 性别：
- 出生年份/年龄：
- 籍贯/家乡：
- 现居城市：
- 教育背景（学校、专业、学历）：
- 职业/职位：
- 当前公司：
- 过往工作经历：
- 婚姻状况：
- 家庭成员：
- 联系方式（手机尾号、微信号等）：

## 别名
（暂无）

## 共同群聊
（暂无）

## 与 Bot 的关系
（暂无）

## 与其他人的关系
（暂无）

## MBTI
（暂无）

## 偏好 & 兴趣
（暂无）

## 近期动态
（暂无）

## 说过的话（短期）
（暂无）

## 交互风格
（暂无）
"""

_UPDATE_PROMPT = """请根据以下对话记录，更新用户 {user_name} 的 wiki。

【当前用户】
主名：{user_name}
别名：{user_aliases}

【已确认身份信息（必须遵守，禁止 contradict）】
{identity_context}

【更新规则】
1. 只记录**当前用户本人**的信息，严禁记录其他用户的信息
2. 增量更新——编译而非堆叠（最重要）：
   - wiki 是**编译后的摘要**，不是对话流水账。只记事实和事件，严禁把原文对话逐条搬进 wiki
   - **身份事实**（姓名/职业/关系/偏好/MBTI）：增量保留，新信息覆盖旧信息，冲突标 `[待验证]`，不要臆测删除
   - **近期动态**：滚动窗口，每人每天最多 1-2 条，只记事件摘要（如"2026-06-20 讨论AI能力"），不记原文对话
   - 超过 7 天的"近期动态"**必须删除**或并入"说过的话"
   - "本次对话未提及" ≠ "信息已过期"，但近期动态到期必须清理
3. 标注日期：时间敏感的信息必须带日期（格式：YYYY-MM-DD），日期必须严格来自对话记录开头的时间戳。禁止编造、推测、推断任何日期
4. 信息来源标注：所有事实信息（姓名、职业、城市、日期、关系等）都必须标注信息来源，格式 `（来源：某群/私聊/某人提及/日期）`，没有例外。**关键：Bot 自己的回复（对话中以"我："或"【Bot回复】"开头的行）不是有效事实来源**——只有当前用户本人或群里其他人明确陈述的内容才能记为事实。Bot 的推测/猜测/回应，若用户未明确确认，不得写入 wiki，或标 `[待验证]`
5. 时间戳缺失：无法确定日期时不标注或用 [待验证] 标记
6. 过期处理：超过 7 天的"近期动态"移到"说过的话"或删除
7. 冲突处理：新信息覆盖旧信息
8. 不确定的信息用 [待验证] 标记
9. 多账号标注：如果对话来源包含不同账号标记，标注所属账号
10. 共同群聊：记录当前用户和 Bot 共同所在的群聊（放在 wiki 靠前位置）。**只列群聊名称，严禁列出对话时间、历史记录或时间戳**
11. 关系记录：记录当前用户与 Bot 的社会关系，如家人、大学同学、小公司同事等（需细分，如"大学同学"而非仅"同学"）。不要记录互动频率
12. 与其他人的关系：记录当前用户与其他人的社会关系，不限于群成员，包括对话中出现的所有人
13. **区分陈述和疑问（严格）**：以"吗"、"呢"、"?"结尾的句子是疑问，不是事实陈述，严禁当作事实提取。例如"周宇之前在上海吗？"是疑问，不能提取为"周宇之前在上海"。
14. MBTI 推断：根据用户的说话风格、用词习惯、决策方式等，推断其可能的 MBTI 类型，并简要说明依据
15. 控制长度：个人 wiki 不超过 4000 字（代码层有兜底截断，但仍请主动精简）
16. 保持 Markdown 格式
17. 别名发现（严格）：只记录当前用户本人的其他称呼。严禁记录其他人的名字。格式：`- 别名：xxx`
18. **沉默≠确认（严格）**：用户"未否认"、"未反驳"、"没回应"绝对不能当作确认事实的依据。只有用户或他人明确陈述/肯定的内容才是 confirmed。Bot 自答后用户沉默的，标 `[待验证]` 或不记录
19. **昵称/微信号唯一归属**：同一昵称/群昵称/微信号只能归属一个人。若现有 wiki 已把某昵称归给 A，对话中又出现该昵称指 B，必须核对清楚再记，不确定时标 `[待验证]`，严禁把同一昵称同时分给两人

【现有 wiki】
{current_wiki}

【新对话】
聊天：{chat_name}
时间：{current_time}

对话内容：
{conversation}

【输出】
直接输出更新后的完整 wiki markdown，不要加代码块标记。严禁添加任何开场白、前言、总结或解释性文字。"""

class MemoryEngine:
    """LLM Wiki 记忆引擎：管理用户/群聊/话题的 wiki 文件，支持外挂 overrides。"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        project_root = Path(__file__).parent.parent.parent
        self.wiki_dir = project_root / "data" / "memory" / "wiki"
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        (self.wiki_dir / "users").mkdir(exist_ok=True)
        (self.wiki_dir / "groups").mkdir(exist_ok=True)
        (self.wiki_dir / "topics").mkdir(exist_ok=True)

        # 外挂配置
        self.overrides_dir = project_root / "data" / "memory" / "overrides"
        self.overrides_dir.mkdir(parents=True, exist_ok=True)
        self._aliases: Dict[str, List[str]] = {}      # 用户名 -> [别名列表]
        self._facts: Dict[str, List[dict]] = {}       # 用户名 -> [事实列表]
        self._corrections: Dict[str, List[str]] = {}  # 群名 -> [纠正列表]
        # 实体级权威纠正（最高优先级，泛化加载，无硬编码实体名）
        self._entity_corrections: Dict[str, List[str]] = {}  # 实体名 -> [纠正]
        self._load_overrides()

        # 异步更新队列
        self._update_queue: List[dict] = []
        self._queue_lock = threading.Lock()
        self._queue_condition = threading.Condition(self._queue_lock)
        self._aliases_lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown = False
        self._start_worker()

    # ── Overrides 加载 ──

    def _load_overrides(self) -> None:
        """加载外挂配置（aliases / facts / corrections）。"""
        # aliases
        aliases_path = self.overrides_dir / "aliases.json"
        if aliases_path.exists():
            try:
                data = json.loads(aliases_path.read_text(encoding="utf-8"))
                for user, cfg in data.get("users", {}).items():
                    if not self._is_valid_main_name(user):
                        _logger.warning(f"aliases.json 中非法主名，跳过: {user}")
                        continue
                    self._aliases[user] = cfg.get("aliases", [])
            except Exception as e:
                _logger.warning(f"加载 aliases 失败: {e}")

        # facts
        facts_path = self.overrides_dir / "facts.json"
        if facts_path.exists():
            try:
                data = json.loads(facts_path.read_text(encoding="utf-8"))
                for user, cfg in data.get("users", {}).items():
                    self._facts[user] = cfg.get("facts", [])
            except Exception as e:
                _logger.warning(f"加载 facts 失败: {e}")

        # corrections（群级 + 实体级）
        corrections_path = self.overrides_dir / "corrections.json"
        if corrections_path.exists():
            try:
                data = json.loads(corrections_path.read_text(encoding="utf-8"))
                for group, cfg in data.get("groups", {}).items():
                    self._corrections[group] = cfg.get("corrections", [])
                # 实体级权威纠正：泛化加载，不依赖任何硬编码实体名。
                # 结构：{"entities": {"<实体名>": {"corrections": [...]}}}
                for entity, cfg in data.get("entities", {}).items():
                    corrs = cfg.get("corrections", []) if isinstance(cfg, dict) else []
                    if corrs:
                        self._entity_corrections[entity] = list(corrs)
            except Exception as e:
                _logger.warning(f"加载 corrections 失败: {e}")

    def _resolve_alias(self, user_name: str) -> str:
        """根据别名找到主用户名。"""
        for main_name, aliases in self._aliases.items():
            if user_name == main_name or user_name in aliases:
                return main_name
        return user_name

    def _all_names_for(self, user_name: str) -> List[str]:
        """获取一个用户的所有名字（主名 + 别名）。"""
        resolved = self._resolve_alias(user_name)
        names = [resolved]
        names.extend(self._aliases.get(resolved, []))
        return list(dict.fromkeys(names))  # 去重保序

    def _entity_corrections_for_doc(self, name: str) -> List[str]:
        """按实体名（精确或带群后缀的基底名）查找权威纠正。

        数据驱动：遍历 corrections.json 的 entities 键，不硬编码任何实体名。
        - 精确匹配：name == entity
        - 基底名匹配：name 以 entity 开头，且紧邻字符是群后缀分隔符
          （如 "实体甲@某群" 归属实体 "实体甲"），避免误配 "实体甲X"。
        """
        if name in self._entity_corrections:
            return list(self._entity_corrections[name])
        for entity, corrs in self._entity_corrections.items():
            if name.startswith(entity):
                rest = name[len(entity):]
                if not rest or rest[0] in "@_（( 、，,|/；;":
                    return list(corrs)
        return []

    def _entity_key_for_doc(self, name: str) -> Optional[str]:
        """返回文档名 name 命中的实体纠正键（无则 None）。

        与 _entity_corrections_for_doc 的匹配口径一致（精确名或基底名），
        用于判定某实体纠正是否已被现有 wiki 文档承载，避免 search_keyword
        重复添加 correction-only 文档。泛化遍历，不硬编码任何实体名。
        """
        if name in self._entity_corrections:
            return name
        for entity in self._entity_corrections:
            if name.startswith(entity):
                rest = name[len(entity):]
                if not rest or rest[0] in "@_（( 、，,|/；;":
                    return entity
        return None

    def _entity_corrections_text(self, names: List[str]) -> str:
        """汇总给定名字命中的实体级权威纠正。

        泛化读取 self._entity_corrections（数据驱动，代码不含实体名）。
        返回可直接嵌入 prompt 的约束文本；无命中时返回空字符串。
        """
        lines = []
        seen = set()
        for name in names:
            resolved = self._resolve_alias(name)
            if resolved in seen:
                continue
            seen.add(resolved)
            for corr in self._entity_corrections_for_doc(resolved):
                lines.append(f"- {resolved}：{corr}")
        return "\n".join(lines)

    def _build_identity_context(self, names: List[str]) -> str:
        """根据 aliases + facts + 实体级纠正 构建身份约束文本，防止 LLM invent 关系。"""
        lines = []
        seen = set()
        for name in names:
            resolved = self._resolve_alias(name)
            if resolved in seen:
                continue
            seen.add(resolved)

            all_names = self._all_names_for(resolved)
            if len(all_names) > 1:
                lines.append(f"- {resolved} 的别名/微信名包括：{'、'.join(all_names[1:])}")

            facts = self._facts.get(resolved, [])
            for f in facts:
                relation = f.get("relation", "")
                value = f.get("value", "")
                note = f.get("note", "")
                if relation and value:
                    line = f"- {resolved} 的 {relation} 是 {value}"
                    if note:
                        line += f"（{note}）"
                    lines.append(line)

        # 实体级权威纠正：放在身份约束中，确保更新/生成 wiki 时禁止 contradict
        corr_text = self._entity_corrections_text(names)
        if corr_text:
            lines.append("【权威纠正（必须遵守，禁止 contradict）】")
            lines.append(corr_text)

        if not lines:
            return "（暂无已确认身份信息，请仅根据对话内容推断，不确定的用 [待验证] 标记）"
        return "\n".join(lines)

    # ── 读取接口 ──

    def get_user_memory(self, user_name: str, max_chars: int = 2000) -> str:
        """读取用户 wiki（含别名合并 + 外挂 facts + 实体级纠正），返回压缩后的摘要。

        顺序（高优先级在前）：实体级权威纠正 → 人工 facts → wiki 本体。
        这样即使截断也保留权威纠正。运行时剔除 [待验证] 派生行。
        """
        resolved = self._resolve_alias(user_name)
        all_names = self._all_names_for(resolved)

        # 合并所有别名的 wiki（运行时剔除 [待验证] 派生行）
        wikis = []
        for name in all_names:
            path = self.wiki_dir / "users" / f"{name}.md"
            if path.exists():
                wikis.append(strip_unverified_lines(self._load_wiki(path)))

        # 实体级权威纠正（最高优先级）
        corrections_text = ""
        corr_text = self._entity_corrections_text([resolved])
        if corr_text:
            corrections_text = "## 权威纠正（人工标注，最高优先级）\n" + corr_text

        # 人工 facts
        facts = self._facts.get(resolved, [])
        facts_text = ""
        if facts:
            fact_lines = ["## 补充信息（人工标注）"]
            for f in facts:
                fact_lines.append(f"- {f.get('relation', '')}：{f.get('value', '')}")
                if f.get("note"):
                    fact_lines.append(f"  （{f['note']}）")
            facts_text = "\n".join(fact_lines)

        if not wikis and not facts_text and not corrections_text:
            return ""

        parts = []
        if corrections_text:
            parts.append(corrections_text)
        if facts_text:
            parts.append(facts_text)
        if wikis:
            parts.append("\n\n".join(wikis))
        wiki_text = "\n\n".join(parts)

        return self._compress_wiki(wiki_text, max_chars)

    def get_group_memory(self, group_name: str, max_chars: int = 2000) -> str:
        """读取群聊 wiki（含外挂 corrections），返回压缩后的摘要。

        运行时剔除 [待验证] 派生行，避免不可靠内容进入上下文。
        """
        path = self.wiki_dir / "groups" / f"{group_name}.md"
        wiki = strip_unverified_lines(self._load_wiki(path)) if path.exists() else ""

        # 注入纠正信息
        corrections = self._corrections.get(group_name, [])
        if corrections:
            corr_text = "\n\n## 重要纠正（人工标注）\n" + "\n".join(f"- {c}" for c in corrections)
            if wiki:
                wiki = wiki + "\n" + corr_text
            else:
                wiki = corr_text

        if not wiki:
            return ""
        return self._compress_wiki(wiki, max_chars)

    # ── 更新接口 ──

    def update_user_wiki(self, user_name: str, chat_name: str,
                         messages: List, bot_replies: List[str]) -> None:
        """把更新任务加入队列，后台异步执行。"""
        if not user_name or self.llm_client is None:
            return
        if not self._is_valid_main_name(user_name):
            _logger.warning(f"非法用户名，跳过 wiki 更新: {user_name}")
            return
        resolved = self._resolve_alias(user_name)
        with self._queue_condition:
            self._update_queue.append({
                "type": "user",
                "user_name": resolved,  # 用主用户名更新
                "chat_name": chat_name,
                "messages": messages,
                "bot_replies": bot_replies,
                "timestamp": time.time(),
            })
            self._queue_condition.notify()

    def update_group_wiki(self, group_name: str, chat_name: str,
                          messages: List, bot_replies: List[str]) -> None:
        """把群聊 wiki 更新任务加入队列，后台异步执行。"""
        if not group_name or self.llm_client is None:
            return
        if not self._is_valid_main_name(group_name):
            _logger.warning(f"非法群名，跳过 wiki 更新: {group_name}")
            return
        with self._queue_condition:
            self._update_queue.append({
                "type": "group",
                "group_name": group_name,
                "chat_name": chat_name,
                "messages": messages,
                "bot_replies": bot_replies,
                "timestamp": time.time(),
            })
            self._queue_condition.notify()

    def shutdown(self) -> None:
        """关闭 worker 线程，等待队列清空。"""
        with self._queue_condition:
            self._shutdown = True
            self._queue_condition.notify_all()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join()

    # ── 内部方法 ──

    def _user_wiki_path(self, user_name: str) -> Path:
        return self.wiki_dir / "users" / f"{_safe_filename(user_name)}.md"

    def _group_wiki_path(self, group_name: str) -> Path:
        return self.wiki_dir / "groups" / f"{_safe_filename(group_name)}.md"

    def _load_wiki(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            _logger.warning(f"加载 wiki 失败 {path}: {e}")
            return ""

    # 时效性 section：超长时优先从这些 section 底部（最老的条目）砍
    _VOLATILE_SECTIONS = ("近期动态", "近期话题", "说过的话", "说过的话（短期）", "历史记录")

    def _enforce_wiki_limits(self, wiki: str, max_chars: int = 10000) -> str:
        """代码级长度护栏（NFR-2）。

        LLM 不遵守长度约束时兜底：按 `## ` 切 section，超长时优先压缩
        时效性 section（近期动态/说过的话），身份/关系 section 保留。
        仍超长再整体尾部截断。这是 LLMWiki"编译摘要"原则的强制保障——
        不让 wiki 退化成无限增长的流水账。
        """
        wiki = wiki.strip()
        if len(wiki) <= max_chars:
            return wiki

        # 按 ## 标题切 section（保留标题行）
        parts = re.split(r"(?=^## )", wiki, flags=re.MULTILINE)
        header = parts[0] if parts else ""
        sections = parts[1:] if len(parts) > 1 else []

        def is_volatile(sec: str) -> bool:
            title = sec.split("\n", 1)[0]
            return any(v in title for v in self._VOLATILE_SECTIONS)

        volatile = [s for s in sections if is_volatile(s)]
        stable = [s for s in sections if not is_volatile(s)]

        stable_len = len(header) + sum(len(s) for s in stable)
        budget = max_chars - stable_len  # volatile 可用的字数预算

        if budget < 0:
            # 稳定 section 已超限：volatile 全部丢弃，整体截断
            result = (header + "".join(stable)).strip()
        elif not volatile:
            result = (header + "".join(stable)).strip()
        else:
            # 按比例给每个 volatile section 分配预算，各自只保留尾部（最新的条目）
            total_v = sum(len(s) for s in volatile)
            compressed = []
            for sec in volatile:
                lines = sec.split("\n")
                title_line = lines[0] if lines else ""
                body = lines[1:]
                # 该 section 分到的预算
                share = max(200, int(budget * len(sec) / total_v)) if total_v > 0 else budget
                if len(sec) <= share:
                    compressed.append(sec)
                    continue
                # 从 body 尾部往前取，直到填满 share
                kept = []
                cur = len(title_line) + 1
                for line in reversed(body):
                    if cur + len(line) + 1 > share:
                        break
                    kept.append(line)
                    cur += len(line) + 1
                kept.reverse()
                compressed.append("\n".join([title_line] + kept))
            result = (header + "".join(stable) + "".join(compressed)).strip()

        # 兜底：仍超长则整体尾部截断
        if len(result) > max_chars:
            # 给截断提示留空间
            cap = max_chars - 30
            truncated = result[:cap]
            last_break = max(truncated.rfind("\n## "), truncated.rfind("\n- "), truncated.rfind("\n\n"))
            if last_break > cap * 0.5:
                truncated = truncated[:last_break]
            result = truncated.strip() + "\n（…记忆已截断，超长部分省略）"

        if len(result) < len(wiki):
            _logger.info(f"[WikiLimit] {len(wiki)} → {len(result)} 字 (上限 {max_chars})")
        return result

    def _save_wiki(self, path: Path, content: str) -> None:
        try:
            # 根据 wiki 类型应用不同长度护栏
            if "/users/" in path.as_posix():
                max_chars = 4000
            else:
                max_chars = 10000
            content = self._enforce_wiki_limits(content, max_chars=max_chars)
            content = self._sanitize_wiki_aliases(content, path.stem)
            path.write_text(content, encoding="utf-8")
        except Exception as e:
            _logger.warning(f"保存 wiki 失败 {path}: {e}")

    def _sanitize_wiki_aliases(self, content: str, main_name: str) -> str:
        """落盘前清洗 wiki 别名行（运行期护栏，防止 LLM 再积垃圾 token）。

        对 ## 别名 段及任意"别名："开头的行：按顿号拆分，去掉 ** 和 （来源：…）
        注解后逐条过 _is_valid_alias，剔除黑名单/描述性/房号等脏 token，保留合法别名。
        同时拦截跨人别名：若某别名已属于其他用户，也从本 wiki 中剔除。
        """
        try:
            existing_mains = set(self._aliases.keys())
        except Exception:
            existing_mains = set()

        # 归一化主名；只有已知用户的 wiki 才做跨人归属校验
        resolved_main = self._resolve_alias(main_name)
        check_ownership = resolved_main in self._aliases

        lines = content.split("\n")
        in_alias_section = False
        changed = False
        for i, raw in enumerate(lines):
            if raw.startswith("## "):
                in_alias_section = "别名" in raw
                continue
            # 捕获列表标记前缀（"- " / "* " 及前置空格）
            lead_m = re.match(r"^(\s*[-*]\s*)", raw)
            lead = lead_m.group(1) if lead_m else ""
            body = raw[len(lead):]
            is_alias_line = in_alias_section or body.startswith("别名")
            if not is_alias_line:
                continue
            if not body.strip() or body.strip().startswith("（暂无"):
                continue
            m = re.match(r"^(别名[/／昵称]*\s*[：:])", body)
            if not m:
                continue
            prefix = m.group(1)
            rest = body[len(prefix):]
            # 按顿号拆，但忽略括号内的顿号（保留 （来源：…） 注解附着在别名上）
            tokens = _split_alias_tokens(rest)
            kept: List[str] = []
            for tok in tokens:
                core = re.sub(r"\*+", "", tok).strip()
                core = re.sub(r"[（(].*[）)]", "", core).strip()
                if not core:
                    kept.append(tok)
                    continue
                if not self._is_valid_alias(core, resolved_main, existing_mains):
                    changed = True
                    continue
                if check_ownership and self._alias_owned_by_other(core, resolved_main):
                    _logger.warning(
                        f"清洗 wiki 别名时发现冲突，从『{resolved_main}』的 wiki 中移除『{core}』"
                        f"（已属于其他用户）"
                    )
                    changed = True
                    continue
                kept.append(tok)
            if len(kept) != len(tokens):
                new_body = prefix + "、".join(kept) if kept else prefix + "（暂无）"
                lines[i] = lead + new_body
                changed = True
        return "\n".join(lines) if changed else content

    def _save_prompt(self, path: Path, prompt: str) -> None:
        """保存生成该 wiki 使用的 prompt，方便排查。截断过长的 conversation 部分。"""
        try:
            prompt_dir = path.parent.parent / "prompts" / path.parent.name
            prompt_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = prompt_dir / f"{path.stem}.md"
            max_len = 80000
            if len(prompt) > max_len:
                truncated = f"{prompt[:40000]}\n\n... [中间部分截断，共 {len(prompt)} 字符] ...\n\n{prompt[-40000:]}"
            else:
                truncated = prompt
            content = f"# Prompt for {path.stem}\n\n生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n```\n{truncated}\n```\n"
            prompt_path.write_text(content, encoding="utf-8")
        except Exception as e:
            _logger.debug(f"保存 prompt 失败 {path}: {e}")

    def _save_alias_suggestion(self, path: Path, aliases: List[str], is_group: bool = False) -> None:
        """把 LLM 发现的别名保存为格式化的 JSON 建议文件，方便人工审核后导入 aliases.json。"""
        try:
            sugg_dir = path.parent.parent / "alias_suggestions" / path.parent.name
            sugg_dir.mkdir(parents=True, exist_ok=True)
            sugg_path = sugg_dir / f"{path.stem}.json"
            data = {
                "main_name": path.stem,
                "aliases": aliases,
                "source": "group" if is_group else "user",
                "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            sugg_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as e:
            _logger.debug(f"保存别名建议失败 {path}: {e}")

    def _compress_wiki(self, wiki: str, max_chars: int) -> str:
        """压缩 wiki 到指定长度。

        注意：此函数按 markdown 标题/列表断点截断，策略与通用
        text_utils._compress_text 不同（后者保留头尾），因此保留
        独立实现。未来如需通用文本压缩，优先使用 text_utils。
        """
        wiki = wiki.strip()
        if len(wiki) <= max_chars:
            return wiki
        truncated = wiki[:max_chars]
        last_break = max(truncated.rfind("\n## "), truncated.rfind("\n- "), truncated.rfind("\n\n"))
        if last_break > max_chars * 0.5:
            truncated = truncated[:last_break]
        return truncated.strip() + "\n（…记忆已截断）"

    def _format_conversation(self, messages: List, bot_replies: List[str]) -> str:
        """构建传给 wiki 提取的证据对话文本。

        Bot/self 消息与 bot_replies 一律排除：Bot 自己的话不是有效事实来源。
        按角色（sender_type == self）排除，不按关键词。
        """
        return format_evidence_conversation(messages, bot_replies)

    def _do_update(self, task: dict) -> None:
        """执行单次 wiki 更新。"""
        task_type = task.get("type", "user")

        if task_type == "group":
            self._do_update_group(task)
        else:
            self._do_update_user(task)

    def _try_generate_wiki(self, prompt: str, path: Path, is_group: bool = False) -> str:
        """调用 LLM 生成 wiki，带重试逻辑（处理 400 超长 / 429 配额限制）。失败时抛异常。"""
        last_error = None
        for attempt in range(3):
            current_prompt = prompt
            if attempt > 0:
                # 截断 conversation 后半部分（保留最近的消息）
                marker = "对话内容：\n"
                if marker in current_prompt:
                    parts = current_prompt.split(marker, 1)
                    if len(parts) == 2:
                        header, conv = parts
                        conv_lines = conv.strip().split("\n")
                        truncated = "\n".join(conv_lines[len(conv_lines) // 2:])
                        current_prompt = header + marker + truncated
                        _logger.warning(f"输入超长，截断 conversation 后重试 ({attempt}/2)")

            try:
                response = self.llm_client.chat(
                    messages=[{"role": "user", "content": current_prompt}],
                    temperature=0.3,
                    max_tokens=10000,
                    timeout=Timeout(connect=10.0, read=300.0, write=10.0, pool=5.0) if Timeout is not None else 300,
                )
                new_wiki = response if isinstance(response, str) else getattr(response, "content", str(response))
                new_wiki = new_wiki.strip()
                # 清理 LLM 常见开场白/前缀
                new_wiki = self._strip_llm_prefix(new_wiki)
                if new_wiki and len(new_wiki) > 50:
                    self._save_wiki(path, new_wiki)
                    self._save_prompt(path, current_prompt)
                    return new_wiki
                raise RuntimeError("LLM 返回内容过短或为空")
            except Exception as e:
                last_error = e
                err_str = str(e)
                if "429" in err_str or "quota" in err_str:
                    if attempt < 2:
                        wait = 5 * (attempt + 1)
                        _logger.warning(f"配额限制，等待 {wait}s 后重试")
                        time.sleep(wait)
                        continue
                elif "400" in err_str and "input length" in err_str:
                    if attempt < 2:
                        _logger.warning(f"输入超长，准备截断重试 ({attempt + 1}/2)")
                        continue
                break
        raise RuntimeError(f"LLM 生成 wiki 失败（已重试 3 次）: {last_error}")

    def _strip_llm_prefix(self, text: str) -> str:
        """去掉 LLM 常见的口头开场白（不截断正文，只做前缀删除）。"""
        prefixes = ("好的，", "好的,", "以下是", "根据", "这是", "我来")
        text = text.lstrip()
        for prefix in prefixes:
            if text.startswith(prefix):
                return text[len(prefix):].lstrip()
        return text

    def _do_update_user(self, task: dict) -> None:
        """执行用户 wiki 更新。"""
        user_name = task["user_name"]
        chat_name = task["chat_name"]
        messages = task["messages"]
        bot_replies = task["bot_replies"]

        path = self._user_wiki_path(user_name)
        current_wiki = self._load_wiki(path) if path.exists() else _DEFAULT_USER_WIKI.format(user_name=user_name)

        conversation = self._format_conversation(messages, bot_replies)
        if not conversation.strip():
            return

        now = time.strftime("%Y-%m-%d %H:%M")
        user_aliases = self._aliases.get(user_name, [])
        identity_context = self._build_identity_context([user_name])
        prompt = _UPDATE_PROMPT.format(
            user_name=user_name,
            user_aliases="、".join(user_aliases) if user_aliases else "无",
            identity_context=identity_context,
            current_wiki=current_wiki,
            chat_name=chat_name,
            current_time=now,
            conversation=conversation,
        )

        new_wiki = self._try_generate_wiki(prompt, path, is_group=False)
        if new_wiki:
            try:
                new_aliases = self._extract_aliases_from_user_wiki(new_wiki, user_name)
                if new_aliases:
                    self._merge_aliases(user_name, new_aliases)
                    self._save_alias_suggestion(path, new_aliases, is_group=False)
            except Exception as e:
                _logger.debug(f"解析用户别名失败: {e}")

    def _do_update_group(self, task: dict) -> None:
        """执行群聊 wiki 更新。"""
        group_name = task["group_name"]
        chat_name = task["chat_name"]
        messages = task["messages"]
        bot_replies = task["bot_replies"]

        path = self._group_wiki_path(group_name)
        current_wiki = self._load_wiki(path) if path.exists() else RUNTIME_DEFAULT_GROUP_WIKI.format(group_name=group_name)

        conversation = self._format_conversation(messages, bot_replies)
        if not conversation.strip():
            return

        now = time.strftime("%Y-%m-%d %H:%M")
        # 提取群里涉及的所有 sender，解析真实身份
        involved = set()
        for msg in messages:
            sender = getattr(msg, "sender", "")
            if sender:
                involved.add(sender)
        identity_context = self._build_identity_context(list(involved))
        prompt = RUNTIME_UPDATE_GROUP_PROMPT.format(
            identity_context=identity_context,
            current_wiki=current_wiki,
            chat_name=chat_name,
            current_time=now,
            conversation=conversation,
        )

        new_wiki = self._try_generate_wiki(prompt, path, is_group=True)
        if new_wiki:
            try:
                group_aliases = self._extract_aliases_from_group_wiki(new_wiki)
                for main_name, aliases in group_aliases.items():
                    if aliases:
                        self._merge_aliases(main_name, aliases)
                        self._save_alias_suggestion(path, aliases, is_group=True)
            except Exception as e:
                _logger.debug(f"解析群别名失败: {e}")

    # ── 别名自动发现 ──

    def _is_valid_main_name(self, name: str) -> bool:
        """主名（wiki 文件名 stem / aliases.json 的 key）合法性校验。

        拒绝：空 / 带文件系统非法字符 / 以点开头 / 过长。
        用于防止 OCR 把 "白:" 这类脏文件名当作合法用户进入记忆系统。
        """
        name = name.strip()
        if not name or name.startswith("."):
            return False
        if len(name) > 64:
            return False
        # 文件名安全字符不是主名合法性的必要条件：
        # wiki 路径会调用 _safe_filename 做兜底，因此允许 | / : 等真实昵称/群名存在。
        return True

    def _is_valid_alias(self, alias: str, main_name: str, existing_mains: set) -> bool:
        """单条别名的统一校验。两个提取器共用，保证入库口径一致。

        拒绝：空 / 等于主名 / 是其他人的主名 / 过长 / 含标点 /
              房号模式 / 微信 ID / 角色词黑名单。
        """
        alias = alias.strip()
        if not alias or alias == main_name:
            return False
        if alias in existing_mains:
            return False
        if len(alias) > _ALIAS_MAX_LEN:
            return False
        if alias in _ALIAS_BLACKLIST:
            return False
        if any(c in alias for c in '。，；：！？.,;:!?'):
            return False
        if alias.startswith("wxid_") or alias.endswith("@chatroom"):
            return False
        if _ROOM_NUMBER_RE.search(alias):
            return False
        return True

    def _alias_owned_by_other(self, alias: str, owner: str) -> bool:
        """检查某个别名是否已经被另一个用户占用（owner 本身除外）。"""
        for main_name, aliases in self._aliases.items():
            if main_name != owner and alias in aliases:
                return True
        return False

    def _extract_aliases_from_user_wiki(self, wiki: str, user_name: str) -> List[str]:
        """从用户 wiki 的 ## 别名 段落提取别名。

        关键修复：LLM 常把多个别名写成 "老王、王总" 一整串，必须按顿号/斜杠
        拆分后再逐条校验，否则整串入库会导致按子串（如 "王总"）召回失败。
        """
        aliases: List[str] = []
        marker = "## 别名"
        if marker not in wiki:
            return aliases
        start = wiki.find(marker)
        end = wiki.find("\n## ", start + 1)
        if end < 0:
            end = len(wiki)
        section = wiki[start:end]

        existing_mains = set(self._aliases.keys())

        for line in section.split("\n"):
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                alias_text = line[2:].strip()
                # 去掉前缀 "别名："
                if alias_text.startswith("别名：") or alias_text.startswith("别名:"):
                    alias_text = alias_text[3:].strip()
                # 去掉括号里的来源说明，如 "qian（发现来源：某群）"
                alias_text = alias_text.split("（")[0].split("(")[0].strip()
                # 拆分整串："老王、王总" → ["老王", "王总"]
                for alias in _split_alias_string(alias_text):
                    if self._is_valid_alias(alias, user_name, existing_mains):
                        if self._alias_owned_by_other(alias, user_name):
                            _logger.warning(
                                f"提取用户 wiki 别名时发现冲突，跳过『{alias}』"
                                f"（已属于其他用户，不分配给『{user_name}』）"
                            )
                            continue
                        if alias not in aliases:
                            aliases.append(alias)
        return aliases

    def _extract_aliases_from_group_wiki(self, wiki: str) -> Dict[str, List[str]]:
        """从群聊 wiki 的成员画像中提取别名。
        匹配格式：**成员名（别名1/别名2）** 或 **成员名**：...
        """
        result: Dict[str, List[str]] = {}
        if "## 群成员画像" not in wiki and "## 活跃成员" not in wiki:
            return result
        # 模式: **成员名（别名1/别名2）**
        existing_mains = set(self._aliases.keys())
        # 从 markdown 中提取 **成员名（别名1/别名2）** 格式
        # 遍历所有 **...** 模式，用字符串操作替代正则
        i = 0
        while i < len(wiki):
            open_idx = wiki.find("**", i)
            if open_idx == -1:
                break
            close_idx = wiki.find("**", open_idx + 2)
            if close_idx == -1:
                break
            content = wiki[open_idx + 2:close_idx]
            i = close_idx + 2
            # 查找括号
            paren_open = content.find("（")
            if paren_open == -1:
                paren_open = content.find("(")
            if paren_open == -1:
                continue
            main = content[:paren_open].strip()
            # 把成员名归一化到主名，防止 LLM 用别名做主名导致归属错误
            resolved_main = self._resolve_alias(main)
            paren_close = content.find("）", paren_open)
            if paren_close == -1:
                paren_close = content.find(")", paren_open)
            if paren_close == -1:
                continue
            alias_str = content[paren_open + 1:paren_close].strip()
            aliases: List[str] = []
            for a in _split_alias_string(alias_str):
                if self._is_valid_alias(a, resolved_main, existing_mains):
                    if self._alias_owned_by_other(a, resolved_main):
                        _logger.warning(
                            f"提取群 wiki 别名时发现冲突，跳过『{a}』"
                            f"（已属于其他用户，不分配给『{resolved_main}』）"
                        )
                        continue
                    if a not in aliases:
                        aliases.append(a)
            if resolved_main and aliases:
                result[resolved_main] = aliases
        return result

    def _merge_aliases(self, user_name: str, new_aliases: List[str]) -> None:
        """把 LLM 发现的别名合并到 aliases.json 和内存中。"""
        with self._aliases_lock:
            self._do_merge_aliases(user_name, new_aliases)

    def _do_merge_aliases(self, user_name: str, new_aliases: List[str]) -> None:
        """_merge_aliases 的无锁实现（内部使用）。"""
        # 1. 更新内存中的 _aliases
        resolved = self._resolve_alias(user_name)
        if resolved not in self._aliases:
            self._aliases[resolved] = []
        existing = set(self._aliases[resolved])
        existing_mains = set(self._aliases.keys())
        # 反向索引：昵称 -> 已归属的主名集合，用于拦截跨人别名冲突
        alias_to_mains: Dict[str, set] = {}
        for main_name, aliases in self._aliases.items():
            for alias in aliases:
                alias_to_mains.setdefault(alias, set()).add(main_name)
        added = []
        for alias in new_aliases:
            # 统一校验：拆分 + 过滤脏数据（防御性，防止上游传入未清洗的串）
            for a in _split_alias_string(alias):
                if not self._is_valid_alias(a, resolved, existing_mains):
                    continue
                if a in existing:
                    continue
                # 跨人冲突：一个昵称不能同时属于两个不同的人
                conflict_owners = alias_to_mains.get(a, set()) - {resolved}
                if conflict_owners:
                    _logger.warning(
                        f"别名冲突，拒绝将『{a}』分配给『{resolved}』，"
                        f"已属于: {sorted(conflict_owners)}"
                    )
                    continue
                self._aliases[resolved].append(a)
                existing.add(a)
                alias_to_mains.setdefault(a, set()).add(resolved)
                added.append(a)

        if not added:
            return

        # 2. 持久化到文件
        try:
            aliases_path = self.overrides_dir / "aliases.json"
            data: Dict[str, Any] = {"users": {}}
            if aliases_path.exists():
                data = json.loads(aliases_path.read_text(encoding="utf-8"))

            users = data.setdefault("users", {})
            if resolved not in users:
                users[resolved] = {"aliases": [], "notes": ""}

            current_aliases = set(users[resolved].get("aliases", []))
            for alias in added:
                if alias not in current_aliases:
                    users[resolved]["aliases"].append(alias)
                    current_aliases.add(alias)

            aliases_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            _logger.info(f"别名发现: {resolved} <- {added}")
        except Exception as e:
            _logger.warning(f"保存别名失败: {e}")

    def _expand_search_keywords(self, keyword: str) -> List[str]:
        """把关键词扩展为包含主名和所有别名的搜索词列表。"""
        keywords = {keyword}
        resolved = self._resolve_alias(keyword)
        keywords.add(resolved)
        # 找到该用户对应的所有别名（双向：主名→别名，别名→主名）
        for main_name, aliases in self._aliases.items():
            if resolved == main_name or keyword == main_name:
                keywords.update(aliases)
                keywords.add(main_name)
            elif keyword in aliases or resolved in aliases:
                keywords.update(aliases)
                keywords.add(main_name)
        return list(keywords)

    def _extract_all_snippets(self, content: str, keywords: List[str], max_snippets: int = 2) -> List[str]:
        """从内容中提取所有包含关键词的片段，去重，限制数量。

        策略：按关键词命中数量从少到多排序，优先提取稀有命中点，
        避免通用词（如人名）占满所有 snippet 配额。
        """
        # 先统计每个关键词的命中数量
        kw_hit_counts = []
        for kw in keywords:
            start = 0
            count = 0
            while True:
                idx = content.find(kw, start)
                if idx < 0:
                    break
                count += 1
                start = idx + len(kw)
            kw_hit_counts.append((kw, count))
        # 按命中数量排序（少的优先）
        kw_hit_counts.sort(key=lambda x: x[1])

        snippets = []
        seen_ranges: set[tuple[int, int]] = set()  # 避免重叠片段
        hit_keywords = set()
        for kw, _ in kw_hit_counts:
            start = 0
            kw_hits = 0
            while True:
                idx = content.find(kw, start)
                if idx < 0:
                    break
                kw_hits += 1
                # 检查是否与已有片段重叠（±50字范围内视为重叠）
                overlap = False
                for (s, e) in seen_ranges:
                    if abs(idx - s) < 50 or abs(idx - e) < 50:
                        overlap = True
                        break
                if not overlap:
                    snippet_start = max(0, idx - 80)
                    snippet_end = min(len(content), idx + 500)
                    snippet = content[snippet_start:snippet_end].strip()
                    if snippet_start > 0:
                        snippet = "…" + snippet
                    if snippet_end < len(content):
                        snippet = snippet + "…"
                    snippets.append(snippet)
                    seen_ranges.add((snippet_start, snippet_end))
                    hit_keywords.add(kw)
                start = idx + len(kw)
                if len(snippets) >= max_snippets:
                    break
            _logger.debug(f"[Snippet] keyword='{kw}' hits={kw_hits}, snippets_so_far={len(snippets)}")
            if len(snippets) >= max_snippets:
                break
        _logger.info(f"[Snippet] extracted {len(snippets)} snippets, hit_keywords={hit_keywords}")
        return snippets

    def search_keyword(self, keyword: str, max_chars: int = 6000, return_scored: bool = False) -> Any:
        """BM25 搜索：召回 + 排序，返回最相关的 wiki 内容。

        召回：遍历所有 wiki，找到包含任意关键词的文档。
        排序：BM25 打分，按相关性排序，只返回 Top 10。
        命中本人 → 返回完整 wiki；命中别人 → 返回片段。

        return_scored=True（诊断用）：返回 (结果文本, scored)，scored 是 BM25
        召回+primary 调整后、LLM rerank 前的候选池（含所有 score>0 的文档）。
        benchmark 用它测召回阶段召回率，区分召回问题 vs 排序问题。
        """
        if not keyword or len(keyword.strip()) < 2:
            return ("", []) if return_scored else ""
        # 按空格分词（去掉太短的词）
        raw_keywords = [kw.strip() for kw in keyword.split() if len(kw.strip()) >= 2]
        if not raw_keywords:
            raw_keywords = [keyword.strip()]
        # 扩展每个关键词的别名
        keywords = []
        for kw in raw_keywords:
            keywords.extend(self._expand_search_keywords(kw))
        keywords = list(dict.fromkeys(keywords))

        resolved_keyword = self._resolve_alias(raw_keywords[0] if raw_keywords else keyword.strip())
        _logger.info(f"[Search] keyword='{keyword}' raw={raw_keywords} expanded={keywords} resolved='{resolved_keyword}'")

        # ── 1. 召回：收集所有文档 ──
        # 实体级纠正与 [待验证] 剔除都是泛化规则，不针对任何具体名字。
        docs: List[Tuple[str, str, bool]] = []  # (name, content, is_group)
        covered_entities: set = set()  # 纠正已附着在现有 wiki 文档上的实体
        for path in (self.wiki_dir / "users").glob("*.md"):
            user = path.stem
            if not self._is_valid_main_name(user):
                _logger.warning(f"非法主名 wiki 文件，跳过搜索: {path.name}")
                continue
            resolved_user = self._resolve_alias(user)
            wiki = strip_unverified_lines(self._load_wiki(path))
            facts = self._facts.get(resolved_user, [])
            facts_text = ""
            if facts:
                facts_text = "\n".join([f"- {f.get('relation', '')}：{f.get('value', '')}" for f in facts])
            # 实体级权威纠正放最前（支持基底名匹配，如 实体甲@某群 归属 实体甲）
            corr_text = "\n".join(f"- {c}" for c in self._entity_corrections_for_doc(resolved_user))
            content = "\n\n".join(
                p for p in (corr_text, facts_text, wiki) if p
            )
            if content:
                # 记录该文档已承载的实体纠正，避免下面重复添加 correction-only 文档
                ek = self._entity_key_for_doc(resolved_user)
                if ek is not None:
                    covered_entities.add(ek)
                docs.append((resolved_user, content, False))

        # 实体级权威纠正：即使没有对应用户 wiki 文件，也作为"仅纠正"文档参与召回，
        # 保证每个实体级纠正都不会因缺 wiki 而丢失（泛化遍历，不硬编码实体名）。
        for entity, corrs in self._entity_corrections.items():
            if entity in covered_entities:
                continue
            corr_text = "\n".join(f"- {c}" for c in corrs)
            if corr_text:
                docs.append((entity, corr_text, False))

        for path in (self.wiki_dir / "groups").glob("*.md"):
            group = path.stem
            wiki = strip_unverified_lines(self._load_wiki(path))
            corrections = self._corrections.get(group, [])
            corr_text = ""
            if corrections:
                corr_text = "\n".join([f"- {c}" for c in corrections])
            content = corr_text + "\n\n" + wiki if corr_text and wiki else (corr_text or wiki)
            if content:
                docs.append((group, content, True))

        _logger.info(f"[Search] retrieved {len(docs)} docs")
        if not docs:
            msg = f"未在本地记忆中找到关于'{keyword}'的信息"
            return (msg, []) if return_scored else msg

        # ── 2. BM25 排序 ──
        N = len(docs)

        # 把关键词分为：原始搜索词 vs 扩展别名
        original_keywords_set = set([resolved_keyword] + raw_keywords)
        original_keywords = [q for q in keywords if q in original_keywords_set]
        expanded_aliases = [q for q in keywords if q not in original_keywords_set]

        # 计算 doc_has_q：原始搜索词单独计算，扩展别名合并为一组
        doc_has_q = {}
        for q in original_keywords:
            doc_has_q[q] = sum(1 for _, c, _ in docs if q in c)
        if expanded_aliases:
            doc_has_q["__aliases__"] = sum(1 for _, c, _ in docs if any(q in c for q in expanded_aliases))

        # 原始搜索词永远保留，扩展别名统一降权（不过滤）
        filtered_original = list(original_keywords)
        use_alias_group = bool(expanded_aliases)

        # IDF：原始搜索词单独计算，扩展别名共享一组 idf
        idf = {}
        for q in filtered_original:
            nq = doc_has_q[q]
            idf[q] = max(math.log((N - nq + 0.5) / (nq + 0.5)), 0.01) if nq > 0 else 0
        if use_alias_group:
            nq = doc_has_q["__aliases__"]
            idf["__aliases__"] = max(math.log((N - nq + 0.5) / (nq + 0.5)), 0.01) if nq > 0 else 0

        # 平均文档长度
        total_len = sum(len(c) for _, c, _ in docs)
        avgdl = total_len / N if N > 0 else 1.0

        k1, b = 1.5, 0.75
        _logger.info(f"[Search] N={N} avgdl={avgdl:.1f} original_keywords={filtered_original} use_alias_group={use_alias_group} idf={ {k: round(v, 4) for k, v in idf.items()} }")

        # 打分：原始搜索词正常计算，扩展别名合并计算并降权 0.3
        # 人名查询场景：查询词能 resolve 到某个 user 主名时，给 user wiki 轻微
        # boost，避免提到该人名的群 wiki 因篇幅长、词频高而 BM25 分数压过
        # user wiki，把跨人关系（如林岚 wiki 提到配偶许安）挤出 Top10。
        is_person_query = any(
            self._resolve_alias(name) == self._resolve_alias(resolved_keyword)
            for name, _, is_group in docs if not is_group
        )
        _logger.info(f"[Search] is_person_query={is_person_query} (resolved='{resolved_keyword}')")
        scored = []  # (name, content, is_group, score, is_primary)
        for name, content, is_group in docs:
            score = 0.0
            dl = len(content)
            # 原始搜索词
            for q in filtered_original:
                f = content.count(q)
                if f > 0 and idf[q] > 0.001:
                    denom = f + k1 * (1 - b + b * dl / avgdl)
                    score += idf[q] * f * (k1 + 1) / denom
            # 扩展别名（合并 + 降权）
            if use_alias_group and expanded_aliases:
                alias_f = sum(content.count(q) for q in expanded_aliases)
                if alias_f > 0 and idf["__aliases__"] > 0.001:
                    denom = alias_f + k1 * (1 - b + b * dl / avgdl)
                    score += 0.3 * idf["__aliases__"] * alias_f * (k1 + 1) / denom
            # 人名查询时 user wiki 加 boost，让跨人关系 user wiki 能进 Top10
            if is_person_query and not is_group and score > 0:
                score *= 1.3
            if score > 0:
                # 本人判断：只有别名精确一致才算 primary
                name_match = (self._resolve_alias(name) == self._resolve_alias(resolved_keyword))
                is_primary = (name_match and not is_group)
                scored.append((name, content, is_group, score, is_primary))

        _logger.info(f"[Search] {len(scored)} docs scored>0")
        for name, _, _, score, is_primary in scored:
            if is_primary:
                _logger.info(f"[Search] primary doc: {name} score={score:.4f}")
        # Log top 10 non-primary for diagnosis
        non_primary = [(n, s) for n, _, _, s, ip in scored if not ip]
        non_primary.sort(key=lambda x: -x[1])
        for name, score in non_primary[:10]:
            _logger.info(f"[Search] non-primary top: {name} score={score:.4f}")

        if not scored:
            msg = f"未在本地记忆中找到关于'{keyword}'的信息"
            return (msg, []) if return_scored else msg

        # 本人优先，然后按 BM25 分数排序
        scored.sort(key=lambda x: (not x[4], -x[3]))

        # 只保留 1 个分数最高的 primary，其余取消 primary 资格（按 BM25 正常竞争）
        primary_seen = False
        adjusted = []
        for name, content, is_group, score, is_primary in scored:
            if is_primary:
                if primary_seen:
                    adjusted.append((name, content, is_group, score, False))
                    _logger.info(f"[Search] demote primary: {name} (score={score:.4f}), only 1 primary allowed")
                else:
                    primary_seen = True
                    adjusted.append((name, content, is_group, score, True))
            else:
                adjusted.append((name, content, is_group, score, False))
        scored = adjusted
        scored.sort(key=lambda x: (not x[4], -x[3]))

        # 召回阶段候选池（rerank 前）的文档名列表，供 return_scored 诊断用
        recall_pool_names = [name for name, _, _, _, _ in scored]

        # ── 2.5 LLM rerank：用 LLM 对 BM25 候选按语义相关性重排 ──
        # BM25 子串匹配无法理解关系语义（如"妈妈"=母亲≠岳母），LLM rerank 补这个短板。
        # llm_client 为 None（benchmark 未传）或调用失败时自动降级回 BM25 顺序。
        all_kw = filtered_original + (expanded_aliases if use_alias_group else [])
        scored = self._llm_rerank(scored, keyword.strip(), all_kw)

        # ── 3. 取 Top 10，组装结果 ──
        results = []
        primary_names = set()  # 记录本人的实际 wiki 名
        selected = scored[:10]
        _logger.info(f"[Search] selected top {len(selected)} docs")
        for name, content, is_group, score, is_primary in selected:
            _logger.info(f"[Search] select: {name} score={score:.4f} primary={is_primary} group={is_group}")
            if is_primary:
                primary_names.add(name)
                # 本人 wiki：优先返回开头（包含基本信息/别名/职业/公司等核心身份字段），
                # 比关键词片段更稳，避免关键身份信息被截断丢失。
                if len(content) > max_chars:
                    results.append(f"【{name}的记忆】{content[:max_chars]}\n（…后续内容省略）")
                else:
                    results.append(f"【{name}的记忆】{content}")
            else:
                # snippet 提取：先用原始搜索词，提取不到再用别名回退
                snippet_keywords = filtered_original if filtered_original else [resolved_keyword]
                snippets = self._extract_all_snippets(content, snippet_keywords, max_snippets=2)
                if not snippets and use_alias_group and expanded_aliases:
                    snippets = self._extract_all_snippets(content, expanded_aliases, max_snippets=2)
                _logger.info(f"[Search] snippets for {name}: {len(snippets)} extracted")
                for snippet in snippets:
                    tag = f"【{name}群记忆】" if is_group else f"【{name}的记忆】"
                    results.append(tag + snippet)

        if not results:
            msg = f"未在本地记忆中找到关于'{keyword}'的信息"
            return (msg, []) if return_scored else msg

        text = "\n".join(results)
        _logger.info(f"[Search] raw results length={len(text)} chars, max_chars={max_chars}")
        if len(text) <= max_chars:
            return (text, recall_pool_names) if return_scored else text

        # 优先保留本人的完整 wiki，其他人的 snippet 后截断
        primary_snippets = []
        other_snippets = []
        for s in results:
            if any(s.startswith(f"【{name}的记忆】") or s.startswith(f"【{name}群记忆】") for name in primary_names):
                primary_snippets.append(s)
            else:
                other_snippets.append(s)

        truncated = ""
        # 先加本人的（完整保留）
        for snippet in primary_snippets:
            if len(truncated) + len(snippet) + 1 > max_chars:
                if not truncated:
                    truncated = snippet[:max_chars] + "\n（…内容截断）"
                break
            truncated = truncated + "\n" + snippet if truncated else snippet

        # 再加其他人的（超长的截断）
        for snippet in other_snippets:
            if len(truncated) + len(snippet) + 1 > max_chars:
                truncated += "\n（…更多结果省略）"
                break
            truncated = truncated + "\n" + snippet if truncated else snippet
        _logger.info(f"[Search] truncated results length={len(truncated)} chars")
        if return_scored:
            return truncated, recall_pool_names
        return truncated

    def _llm_rerank(
        self,
        scored: List[Tuple[str, str, bool, float, bool]],
        query: str,
        keywords: List[str],
    ) -> List[Tuple[str, str, bool, float, bool]]:
        """用 LLM 对 BM25 召回的候选重排。

        输入 scored（已按 BM25 排序的 (name, content, is_group, score, is_primary)），
        提取每个候选的最相关 snippet，让 LLM 按与 query 的语义相关性重排。

        降级：llm_client 为 None / 调用异常 / 解析失败 → 返回原 scored（不阻断）。
        LLM 能理解 BM25 无法处理的语义，如"妈妈=母亲≠岳母"的关系消歧。
        """
        if not scored or self.llm_client is None:
            return scored
        # 候选过多时只取前 10（与最终 top10 一致）
        candidates = scored[:10]
        if len(candidates) <= 1:
            return scored

        # 为每个候选提取最相关 snippet（复用 _extract_all_snippets）
        snippet_kws = keywords if keywords else [query]
        cand_lines: List[str] = []
        for i, (name, content, _is_group, _score, _is_primary) in enumerate(candidates, 1):
            snippets = self._extract_all_snippets(content, snippet_kws, max_snippets=1)
            snippet = snippets[0] if snippets else content[:200]
            # 截断防 prompt 过长：200 字让 LLM 看到完整关系描述（如"母亲"/"岳母"）
            snippet = snippet.replace("\n", " ").strip()[:200]
            cand_lines.append(f"[{i}] {name}：{snippet}")

        prompt = (
            "查询：{q}\n候选历史记忆片段（编号-内容）：\n{cands}\n\n"
            "请按与查询的语义相关性从高到低排序，返回编号数组（JSON，如 [3,1,2,5]）。\n"
            "排序原则：\n"
            "1. 关系消歧：查询中的人物关系词（如妈妈=母亲、爸爸=父亲）指直系亲属，"
            "不是配偶的父母（岳母/婆婆）或同名无关的人。候选 snippet 中明确写"
            "'XX的妈妈/母亲'的优先于'XX配偶的母'。\n"
            "2. 直接相关优先：候选内容直接描述查询对象本人的，优先于仅提及查询词的。\n"
            "3. 过滤词面重叠：仅因群名/昵称含查询词而无实质关系的排最后。\n"
            "只返回 JSON 数组，不要解释。"
        ).format(q=query, cands="\n".join(cand_lines))

        try:
            resp = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
                timeout=30,
            )
            text = resp if isinstance(resp, str) else getattr(resp, "content", str(resp))
            # 解析 [n,n,...] 编号数组
            import re as _re
            m = _re.search(r"\[([0-9,\s]+)\]", text)
            if not m:
                _logger.warning("[Search][LLMRerank] 未解析到编号数组，回退 BM25: %s", text[:80])
                return scored
            order = [int(x.strip()) for x in m.group(1).split(",") if x.strip().isdigit()]
            if not order:
                return scored

            # 按 LLM 顺序重排候选；LLM 未列出的候选保留在后面（按原 BM25 顺序）
            seen = set()
            reranked: List[Tuple[str, str, bool, float, bool]] = []
            for idx in order:
                if 1 <= idx <= len(candidates) and idx not in seen:
                    seen.add(idx)
                    reranked.append(candidates[idx - 1])
            for i, c in enumerate(candidates, 1):
                if i not in seen:
                    reranked.append(c)
            # 保留 scored 10 之后的（若有）
            reranked.extend(scored[10:])
            _logger.info("[Search][LLMRerank] rerank ok, new order: %s", order)
            return reranked
        except Exception as e:
            _logger.warning("[Search][LLMRerank] 调用失败，回退 BM25: %s", e)
            return scored

    def _start_worker(self) -> None:
        """启动后台 worker 线程，定期处理更新队列。"""
        def _worker():
            while True:
                batch = []
                with self._queue_condition:
                    if self._shutdown:
                        batch = self._update_queue[:]
                        self._update_queue.clear()
                        if not batch:
                            return
                    elif len(self._update_queue) >= 3:
                        batch = self._update_queue[:3]
                        del self._update_queue[:3]
                    elif self._update_queue:
                        now = time.time()
                        cutoff = [i for i, t in enumerate(self._update_queue) if now - t["timestamp"] > 300]
                        if cutoff:
                            batch = self._update_queue[:cutoff[-1] + 1]
                            del self._update_queue[:len(batch)]
                    if not batch:
                        self._queue_condition.wait(timeout=5)
                        continue
                for task in batch:
                    try:
                        self._do_update(task)
                    except Exception as e:
                        _logger.error(f"Worker 处理任务失败: {e}, task_type={task.get('type')}, user={task.get('user_name') or task.get('group_name')}")
                    time.sleep(1)

        self._worker_thread = threading.Thread(target=_worker, daemon=True)
        self._worker_thread.start()

    def search_related_mentions(self, text: str, exclude_user: Optional[str] = None, max_files: int = 5) -> List[str]:
        """扫描文本中提到的人名，只加载这些人自己的 wiki，不加载别人的 wiki 里提到他的情况。

        wiki 文件本身很短，直接返回完整内容即可，不需要截断/去重。
        """
        if not text:
            return []

        # 1. 收集所有已知名字（主名 + 别名）
        all_names = set()
        for main_name, aliases in self._aliases.items():
            all_names.add(main_name)
            all_names.update(aliases)

        # 2. 找出文本中提到的名字
        mentioned = set()
        for name in all_names:
            if name in text:
                mentioned.add(name)
        if not mentioned:
            return []

        # 3. 对每个提到的名字 resolve 到主名，只加载该主名自己的 wiki
        results: List[str] = []
        seen_users: set = set()

        for name in mentioned:
            main_name = self._resolve_alias(name)
            if main_name == exclude_user:
                continue
            if main_name in seen_users:
                continue
            seen_users.add(main_name)

            path = self.wiki_dir / "users" / f"{main_name}.md"
            if not path.exists():
                continue
            wiki_content = strip_unverified_lines(self._load_wiki(path))
            facts = self._facts.get(main_name, [])
            facts_text = "## 已知事实\n" + "\n".join(f"- {f}" for f in facts) if facts else ""
            corr_text = "\n".join(f"- {c}" for c in self._entity_corrections_for_doc(main_name))
            full_content = "\n\n".join(
                p for p in (corr_text, facts_text, wiki_content) if p
            )
            if not full_content:
                continue
            header = f"【{main_name} 相关】"
            results.append(header + full_content)
            if len(results) >= max_files:
                break

        return results

    # ── Lint 健康检查（FR-7/8，LLMWiki 核心操作）──

    def lint_memory(self, max_wiki_chars: int = 4000) -> dict:
        """扫描记忆库，产出问题清单。

        检查项：
        - conflicts: 同一别名指向多个主名
        - bloated: 超长度上限的 wiki 文件
        - duplicates: 归一化后同名的群/用户
        - stale: （占位）过时近期动态，需 LLM 判定，此处不自动检测

        返回结构化 dict。调用方可据此自动截断 bloated、标记 conflicts 供人工审核。
        """
        report: Dict[str, Any] = {
            "conflicts": [], "bloated": [], "duplicates": [], "ad_groups": [], "stale": [],
        }

        # 1. 别名冲突：同一别名出现在多个主名下
        alias2mains: Dict[str, List[str]] = {}
        for main, aliases in self._aliases.items():
            for a in aliases:
                alias2mains.setdefault(a, []).append(main)
        for alias, mains in alias2mains.items():
            if len(mains) > 1:
                report["conflicts"].append({"alias": alias, "mains": mains})

        # 2. 膨胀 wiki
        for sub in ("users", "groups"):
            d = self.wiki_dir / sub
            if not d.exists():
                continue
            for path in d.glob("*.md"):
                content = ""
                try:
                    content = path.read_text(encoding="utf-8")
                except Exception as e:
                    _logger.warning("[lint_memory] 读取 wiki 失败 %s: %s", path, e)
                if content and len(content) > max_wiki_chars:
                    report["bloated"].append({
                        "path": str(path.relative_to(self.wiki_dir)),
                        "chars": len(content),
                    })
        # 3. 重复群/用户（归一化后同名）
        for sub in ("users", "groups"):
            d = self.wiki_dir / sub
            if not d.exists():
                continue
            norm_map: Dict[str, List[str]] = {}
            for path in d.glob("*.md"):
                norm = normalize_chat_name(path.stem)
                norm_map.setdefault(norm, []).append(path.stem)
            for norm, names in norm_map.items():
                if len(names) > 1:
                    report["duplicates"].append({"normalized": norm, "names": names})

        _logger.info(
            f"[Lint] conflicts={len(report['conflicts'])} bloated={len(report['bloated'])} "
            f"duplicates={len(report['duplicates'])} ad_groups={len(report['ad_groups'])}"
        )
        return report

    def lint_truncate_bloated(self, max_wiki_chars: int = 4000, apply: bool = False) -> list:
        """对膨胀 wiki 执行截断（FR-8 可写回操作）。

        apply=False 时只返回将截断的清单（dry-run）；
        apply=True 时回写文件（不自动备份，调用方应先备份）。
        """
        report = self.lint_memory(max_wiki_chars)
        actions = []
        for item in report["bloated"]:
            path = self.wiki_dir / item["path"]
            try:
                orig = path.read_text(encoding="utf-8")
                truncated = self._enforce_wiki_limits(orig, max_wiki_chars)
                if len(truncated) < len(orig):
                    actions.append({"path": item["path"], "before": len(orig), "after": len(truncated)})
                    if apply:
                        path.write_text(truncated, encoding="utf-8")
            except Exception as e:
                _logger.warning(f"[Lint] 截断失败 {path}: {e}")
        return actions
