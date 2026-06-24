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

_logger = logging.getLogger("src.memory.engine")

# ── 别名校验（防止把非别名噪声写进 aliases.json）──

# 角色词 / 系统占位符：明显不是真人昵称，一律拒绝
_ALIAS_BLACKLIST = {
    "Bot", "bot", "我", "对方", "对话中", "匿名", "未知昵称", "未知", "群主", "群聊主人",
    "群成员", "记录者", "记录人", "旁白", "ai开发小分队", "本人", "自己", "他人", "某人",
    "无", "暂无", "未发现", "未发现其他显著别名",
}

# 描述性关键词：含这些词的字符串大概率是句子而非别名（与历史 invalid_keywords 对齐）
_ALIAS_INVALID_KEYWORDS = [
    "说", "提到", "认为", "和", "与", "让", "叫", "是", "在", "觉得", "告诉", "问",
    "回答", "表示", "介绍", "@", "称为", "称呼", "未发现", "无其他", "显著别名",
    "可能别名", "群友", "朋友", "邻居", "无其他别名",
]

# 房号 / 单元号模式：如 "6幢5号501"、"4-1-703"、"1幢10号802"
_ROOM_NUMBER_RE = re.compile(r"\d+\s*[幢栋号楼室单元]\s*\d|\d+-\d+-\d+|\d+幢\d+号")

# 别名长度上限：真实昵称不会太长
_ALIAS_MAX_LEN = 15

# 别名拆分符：顿号 / 斜杠 / 空格（拆完后逐条校验）
_ALIAS_SPLIT_RE = re.compile(r"[、/／\|｜\s]+")

# 广告群特征：群名带具体斤价（"6.99一斤"、"X.X元X斤...百果园" 等），命中即不入库
_AD_GROUP_PATTERNS = [
    re.compile(r"\d+\.?\d*\s*[元块]?\s*.*一斤"),
    re.compile(r"\d+\.?\d*\s*[元块].*斤"),
    re.compile(r"\d+\.?\d*\s*一斤"),
]
# emoji 范围（粗略，用于群名归一化剥离）
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002B00-\U00002BFF]+",
)


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
    n = _EMOJI_RE.sub("", name)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _split_alias_string(s: str) -> List[str]:
    """把 LLM 输出的别名串按顿号/斜杠/空格拆成单条，去重保序。

    LLM 经常把多个别名写成 "老王、王总" 或 "Paul、坤蜀黍" 一整串，
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
4. 信息来源标注：所有事实信息（姓名、职业、城市、日期、关系等）都必须标注信息来源，格式 `（来源：某群/私聊/某人提及/日期）`，没有例外
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

【现有 wiki】
{current_wiki}

【新对话】
聊天：{chat_name}
时间：{current_time}

对话内容：
{conversation}

【输出】
直接输出更新后的完整 wiki markdown，不要加代码块标记。严禁添加任何开场白、前言、总结或解释性文字。"""

_DEFAULT_GROUP_WIKI = """# {group_name}

## 群基本信息
（暂无）

## 群成员画像
（暂无）

## 近期话题 & 动态
（暂无）

## 群内规则 & 文化
（暂无）
"""

_UPDATE_GROUP_PROMPT = """请根据以下对话记录，更新群聊 {chat_name} 的 wiki。

【已确认身份信息（必须遵守，禁止 contradict）】
{identity_context}

【更新规则】
1. wiki 是**编译后的摘要**，不是对话流水账——这是最重要的原则：
   - 严禁把原文对话（每条发言 + Bot 回复全文）逐条搬进 wiki
   - "近期话题 & 动态"只记**事件摘要**：每人每天最多 1-2 条，一句话概括发生了什么（如"2026-06-20 讨论AI能力，白姐认为ai干不好"）
   - 不要记录每条发言的原文、不要记录 Bot 的回复原文
2. 身份/关系信息（群成员画像、群规则文化）增量保留，新信息覆盖旧信息，冲突标 `[待验证]`
3. 超过 7 天的"近期动态"**必须删除**或并入历史，保持滚动窗口精简
4. 只修改/新增变化的部分，保留未变动的内容
5. 标注日期：时间敏感的信息必须带日期（格式：YYYY-MM-DD），日期必须严格来自对话记录开头的时间戳。禁止编造、推测、推断任何日期
6. 时间戳缺失：无法确定日期时不标注或用 [待验证] 标记
7. 冲突处理：新信息覆盖旧信息
8. 重点记录：
   - 群成员关系、身份、职业变化
   - 群内热点话题、事件、约定（摘要，非原文）
   - 群内文化、梗、常用语
   - 群规则、禁忌、注意事项
9. 别名发现：在"群成员画像"中，如果某个成员有多个称呼，请一并记录。只记录该成员本人的称呼，严禁把其他成员的名字误记到此成员下。格式：`成员主名（别名1/别名2）`
10. 多账号标注：如果对话来源包含不同账号标记，标注所属账号
11. 不确定的信息用 [待验证] 标记
12. 控制长度：群聊 wiki 不超过 4000 字（代码层有兜底截断，但仍请主动精简）
13. 保持 Markdown 格式

【现有 wiki】
{current_wiki}

【新对话】
群聊：{chat_name}
时间：{current_time}

对话内容：
{conversation}

【输出】
直接输出更新后的完整 wiki markdown，不要加代码块标记。"""


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
        self._load_overrides()

        # 异步更新队列
        self._update_queue: List[dict] = []
        self._queue_lock = threading.Lock()
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

        # corrections
        corrections_path = self.overrides_dir / "corrections.json"
        if corrections_path.exists():
            try:
                data = json.loads(corrections_path.read_text(encoding="utf-8"))
                for group, cfg in data.get("groups", {}).items():
                    self._corrections[group] = cfg.get("corrections", [])
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

    def _build_identity_context(self, names: List[str]) -> str:
        """根据 aliases + facts 构建身份约束文本，防止 LLM 在生成 wiki 时 invent 关系。"""
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

        if not lines:
            return "（暂无已确认身份信息，请仅根据对话内容推断，不确定的用 [待验证] 标记）"
        return "\n".join(lines)

    # ── 读取接口 ──

    def get_user_memory(self, user_name: str, max_chars: int = 2000) -> str:
        """读取用户 wiki（含别名合并 + 外挂 facts），返回压缩后的摘要。facts 放在前面确保不被截断。"""
        resolved = self._resolve_alias(user_name)
        all_names = self._all_names_for(resolved)

        # 合并所有别名的 wiki
        wikis = []
        for name in all_names:
            path = self.wiki_dir / "users" / f"{name}.md"
            if path.exists():
                wikis.append(self._load_wiki(path))

        # 先构建 facts（放在前面，确保截断时不丢失）
        facts = self._facts.get(resolved, [])
        facts_text = ""
        if facts:
            fact_lines = ["## 补充信息（人工标注）"]
            for f in facts:
                fact_lines.append(f"- {f.get('relation', '')}：{f.get('value', '')}")
                if f.get("note"):
                    fact_lines.append(f"  （{f['note']}）")
            facts_text = "\n".join(fact_lines)

        if not wikis and not facts_text:
            return ""

        # facts 放在 wiki 前面，确保即使截断也保留人工标注
        wiki_text = "\n\n".join(wikis)
        if facts_text and wiki_text:
            wiki_text = facts_text + "\n\n" + wiki_text
        elif facts_text:
            wiki_text = facts_text

        return self._compress_wiki(wiki_text, max_chars)

    def get_group_memory(self, group_name: str, max_chars: int = 2000) -> str:
        """读取群聊 wiki（含外挂 corrections），返回压缩后的摘要。"""
        path = self.wiki_dir / "groups" / f"{group_name}.md"
        wiki = self._load_wiki(path) if path.exists() else ""

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
        resolved = self._resolve_alias(user_name)
        with self._queue_lock:
            self._update_queue.append({
                "type": "user",
                "user_name": resolved,  # 用主用户名更新
                "chat_name": chat_name,
                "messages": messages,
                "bot_replies": bot_replies,
                "timestamp": time.time(),
            })

    def update_group_wiki(self, group_name: str, chat_name: str,
                          messages: List, bot_replies: List[str]) -> None:
        """把群聊 wiki 更新任务加入队列，后台异步执行。"""
        if not group_name or self.llm_client is None:
            return
        # 广告群拦截（FR-14）：群名带具体斤价的团购广告不生成 wiki
        if any(p.search(group_name) for p in _AD_GROUP_PATTERNS):
            _logger.debug("广告群跳过 wiki 生成: %s", group_name)
            return
        with self._queue_lock:
            self._update_queue.append({
                "type": "group",
                "group_name": group_name,
                "chat_name": chat_name,
                "messages": messages,
                "bot_replies": bot_replies,
                "timestamp": time.time(),
            })

    def shutdown(self) -> None:
        """关闭 worker 线程，等待队列清空。"""
        self._shutdown = True
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10)

    # ── 内部方法 ──

    def _user_wiki_path(self, user_name: str) -> Path:
        return self.wiki_dir / "users" / f"{user_name}.md"

    def _group_wiki_path(self, group_name: str) -> Path:
        return self.wiki_dir / "groups" / f"{group_name}.md"

    def _load_wiki(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            _logger.warning(f"加载 wiki 失败 {path}: {e}")
            return ""

    # 时效性 section：超长时优先从这些 section 底部（最老的条目）砍
    _VOLATILE_SECTIONS = ("近期动态", "近期话题", "说过的话", "说过的话（短期）", "历史记录")

    def _enforce_wiki_limits(self, wiki: str, max_chars: int = 4000) -> str:
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
            content = self._enforce_wiki_limits(content)
            path.write_text(content, encoding="utf-8")
        except Exception as e:
            _logger.warning(f"保存 wiki 失败 {path}: {e}")

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
        lines = []
        last_chat_name = None
        for msg in messages:
            chat_name = getattr(msg, "chat_name", "")
            # 当聊天名称变化时插入分隔线
            if chat_name and chat_name != last_chat_name:
                lines.append(f"\n===== {chat_name} =====\n")
                last_chat_name = chat_name

            st = getattr(msg, "sender_type", None)
            is_self = False
            if st is not None:
                if hasattr(st, "value"):
                    is_self = st.value == "self"
                else:
                    is_self = str(st) == "self"
            sender = "我" if is_self else getattr(msg, "sender", "")
            text = getattr(msg, 'text', '')
            account = getattr(msg, 'account', '')
            # 时间戳（支持历史批量导入）
            ts = getattr(msg, 'create_time', None)
            ts_str = ""
            if ts:
                try:
                    ts_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(int(ts)))
                except Exception as e:
                    _logger.warning("[MemoryEngine] 时间戳格式化失败: %s (ts=%r)", e, ts)
            # 组装前缀：[账号][时间]
            prefix = ""
            if account:
                prefix += f"[{account}]"
            if ts_str:
                prefix += f"[{ts_str}]"
            if prefix:
                lines.append(f"{prefix}{sender}：{text}")
            else:
                lines.append(f"{sender}：{text}")
        if bot_replies:
            for reply in bot_replies:
                lines.append(f"我：{reply}")
        return "\n".join(lines)

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
        """去掉 LLM 常见的开场白、前言等前缀（直到第一个 # 标题）。"""
        prefixes = ("好的，", "好的,", "以下是", "根据", "这是", "我来")
        text = text.lstrip()
        for prefix in prefixes:
            if text.startswith(prefix):
                hash_idx = text.find("# ")
                if hash_idx != -1:
                    text = text[hash_idx:]
                break
        return text.strip()

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
        current_wiki = self._load_wiki(path) if path.exists() else _DEFAULT_GROUP_WIKI.format(group_name=group_name)

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
        prompt = _UPDATE_GROUP_PROMPT.format(
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

    def _is_valid_alias(self, alias: str, main_name: str, existing_mains: set) -> bool:
        """单条别名的统一校验。两个提取器共用，保证入库口径一致。

        拒绝：空 / 等于主名 / 是其他人的主名 / 过长 / 含描述性关键词 / 含标点 /
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
        if any(kw in alias for kw in _ALIAS_INVALID_KEYWORDS):
            return False
        if any(c in alias for c in '。，；：！？.,;:!?'):
            return False
        if alias.startswith("wxid_") or alias.endswith("@chatroom"):
            return False
        if _ROOM_NUMBER_RE.search(alias):
            return False
        return True

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
            paren_close = content.find("）", paren_open)
            if paren_close == -1:
                paren_close = content.find(")", paren_open)
            if paren_close == -1:
                continue
            alias_str = content[paren_open + 1:paren_close].strip()
            aliases: List[str] = []
            for a in _split_alias_string(alias_str):
                if self._is_valid_alias(a, main, existing_mains):
                    if a not in aliases:
                        aliases.append(a)
            if main and aliases:
                result[main] = aliases
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
        added = []
        for alias in new_aliases:
            # 统一校验：拆分 + 过滤脏数据（防御性，防止上游传入未清洗的串）
            for a in _split_alias_string(alias):
                if not self._is_valid_alias(a, resolved, existing_mains):
                    continue
                if a in existing:
                    continue
                self._aliases[resolved].append(a)
                existing.add(a)
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
        docs: List[Tuple[str, str, bool]] = []  # (name, content, is_group)
        for path in (self.wiki_dir / "users").glob("*.md"):
            user = path.stem
            resolved_user = self._resolve_alias(user)
            wiki = self._load_wiki(path)
            facts = self._facts.get(resolved_user, [])
            facts_text = ""
            if facts:
                facts_text = "\n".join([f"- {f.get('relation', '')}：{f.get('value', '')}" for f in facts])
            content = facts_text + "\n\n" + wiki if facts_text and wiki else (facts_text or wiki)
            if content:
                docs.append((resolved_user, content, False))
        for path in (self.wiki_dir / "groups").glob("*.md"):
            group = path.stem
            wiki = self._load_wiki(path)
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
        # user wiki，把跨人关系（如王芊 wiki 提到配偶王艺涵）挤出 Top10。
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
                # 本人 wiki 如果太长，提取包含查询词的 snippet，避免完整 wiki 挤占空间导致截断
                snippet_keywords = filtered_original if filtered_original else [resolved_keyword]
                if len(content) > max_chars:
                    snippets = self._extract_all_snippets(content, snippet_keywords, max_snippets=3)
                    if snippets:
                        for snippet in snippets:
                            results.append(f"【{name}的记忆】{snippet}")
                    else:
                        # 提取不到 snippet 时fallback：返回开头+截断提示
                        results.append(f"【{name}的记忆】{content[:max_chars]}\n（…内容截断）")
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
            while not self._shutdown:
                time.sleep(5)
                batch = []
                with self._queue_lock:
                    if len(self._update_queue) >= 3:
                        batch = self._update_queue[:3]
                        self._update_queue = self._update_queue[3:]
                    elif self._update_queue:
                        now = time.time()
                        cutoff = [i for i, t in enumerate(self._update_queue) if now - t["timestamp"] > 300]
                        if cutoff:
                            batch = self._update_queue[:cutoff[-1] + 1]
                            self._update_queue = self._update_queue[len(batch):]
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
            wiki_content = self._load_wiki(path)
            facts = self._facts.get(main_name, [])
            facts_text = "## 已知事实\n" + "\n".join(f"- {f}" for f in facts) if facts else ""
            full_content = facts_text + "\n\n" + wiki_content if facts_text and wiki_content else (facts_text or wiki_content)
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
        - ad_groups: 命中广告群特征的 wiki
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

        # 2. 膨胀 wiki + 4. 广告群
        for sub in ("users", "groups"):
            d = self.wiki_dir / sub
            if not d.exists():
                continue
            for path in d.glob("*.md"):
                try:
                    content = path.read_text(encoding="utf-8")
                except Exception:
                    continue
                if len(content) > max_wiki_chars:
                    report["bloated"].append({
                        "path": str(path.relative_to(self.wiki_dir)),
                        "chars": len(content),
                    })
                if sub == "groups" and any(p.search(path.stem) for p in _AD_GROUP_PATTERNS):
                    report["ad_groups"].append(path.stem)

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
