#!/usr/bin/env python3
"""从 WeFlow JSON 导出中本地抽取、脱敏并分层选择本人回复样本。"""

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# 本地运行时可设置 PERSONA_NAME=王芊 以匹配已有 wiki；代码里不再硬编码真名。
PERSONA_NAME = os.environ.get("PERSONA_NAME", "本人")


SENSITIVE_PATTERNS = [
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?i)https?://|www\."),
    re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b"),
    re.compile(r"(?i)wxid_[a-z0-9_]+"),
    re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
    re.compile(r"验证码|密码|身份证|银行卡|收款码|付款码|转账|红包|定位|详细地址"),
    re.compile(r"(?i)ignore\s+(?:all\s+)?previous|system\s*prompt|developer\s*message|jailbreak|\bDAN\b"),
    re.compile(r"忽略.{0,12}(?:之前|上面|此前).{0,8}(?:指令|要求|提示)|系统提示词|开发者消息|越狱提示|你现在是"),
]
PLACEHOLDER_PATTERNS = [
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[手机号]"),
    (re.compile(r"(?i)https?://\S+|www\.\S+"), "[链接]"),
    (re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b"), "[邮箱]"),
    (re.compile(r"(?i)wxid_[a-z0-9_]+"), "[微信账号]"),
]
BUSINESS_WORDS = ("店长", "销售", "客服", "设计师", "中介", "物业", "团购", "商家", "客户")
FAMILY_WORDS = ("家人", "亲属", "父亲", "母亲", "爸爸", "妈妈", "夫妻", "老公", "老婆", "伴侣", "兄弟姐妹")
COLLEAGUE_WORDS = ("同事", "同学", "校友", "合作", "工作关系", "前同事")
FRIEND_WORDS = ("好友", "朋友", "关系很好", "熟识", "发小", "闺蜜")
HUMOR_PATTERNS = (
    re.compile(r"哈哈|笑死|绷不住|离谱|救命|牛的|绝了|难民|韭菜|躺平|破防|认亲"),
    re.compile(r"[😂🤣😅🤡🙃😏]|\[(?:捂脸|旺柴|破涕为笑|偷笑|呲牙)\]"),
    re.compile(r"哪有|怕不是|属实|看来.*要|估计.*正|就剩|给.*加分"),
)
MAX_TURN_GAP_SECONDS = 600


@dataclass
class Candidate:
    relation: str
    chat_id: str
    context: list[str]
    replies: list[str]
    score: float
    timestamp: int
    intent: str = "comment"
    reply_shape: str = "single"
    context_dependency: str = "low"
    style_tags: tuple[str, ...] = ()
    topic: str = "daily_chat"
    humor_type: str = "none"
    response_mode: str = "neutral"
    priority: bool = False


def _stable_id(value: str, prefix: str = "chat") -> str:
    normalized = re.sub(r"^(私聊_|群聊_|曾经的好友_)", "", value).strip()
    return f"{prefix}_{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:10]}"


def _is_text(message: dict) -> bool:
    return message.get("localType") == 1 and bool(str(message.get("content") or "").strip())


def _safe_text(text: str) -> bool:
    text = text.strip()
    if not 1 <= len(text) <= 160 or text.startswith("<"):
        return False
    return not any(pattern.search(text) for pattern in SENSITIVE_PATTERNS)


def _is_low_signal_reply(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return bool(
        re.fullmatch(r"[?？!！.,，。~～]+", compact)
        or re.fullmatch(r"@[^\s]+", compact)
        or re.fullmatch(r"\[[^\]]+\]", compact)
    )


def _timestamp_seconds(value: object) -> float:
    try:
        timestamp = float(value or 0)
    except (TypeError, ValueError):
        return 0
    return timestamp / 1000 if timestamp > 10_000_000_000 else timestamp


def _is_coherent_turn(incoming: list[dict], outgoing: list[dict]) -> bool:
    first_incoming_time = _timestamp_seconds(incoming[0].get("createTime"))
    incoming_time = _timestamp_seconds(incoming[-1].get("createTime"))
    outgoing_time = _timestamp_seconds(outgoing[0].get("createTime"))
    if not incoming_time or not outgoing_time:
        return True
    gap = outgoing_time - incoming_time
    full_span = outgoing_time - first_incoming_time if first_incoming_time else gap
    return 0 <= gap <= MAX_TURN_GAP_SECONDS and full_span <= MAX_TURN_GAP_SECONDS * 3


def _topic(context: list[str], replies: list[str]) -> str:
    text = " ".join(context)
    topic_patterns = (
        ("finance", r"股票|基金|涨停|跌停|持仓|港股|美股|理财|利率|保险|期货|韭菜"),
        ("housing", r"房子|买房|卖房|租房|楼盘|户型|装修|物业|建材|设计师|小区"),
        ("work", r"工作|上班|下班|加班|老板|同事|面试|工资|项目|公司|离职|辞职|周报|晋升|升职|绩效|竞业|裁员|失业|不想干|干不动"),
        ("health", r"医院|医生|体检|疫苗|病毒|生病|头疼|肚子疼|吃药|睡眠|传染|输液|挂号|治疗|住院"),
        ("relationships", r"恋爱|对象|男朋友|女朋友|男友|女友|结婚|离婚|相亲|约会|喜欢|感情"),
        ("family", r"爸爸|妈妈|父母|爷爷|奶奶|老公|老婆|家里|家人|亲戚|去世|上坟|清明"),
        ("parenting_education", r"孩子|宝宝|幼儿园|学校|老师|学习|考试|作业|母乳|奶粉|育儿"),
        ("food_travel", r"吃饭|火锅|拉面|餐厅|好吃|外卖|旅游|旅行|酒店|机票|三亚|度假|景点"),
        ("tech", r"(?i)\bai\b|人工智能|大模型|芯片|手机|电脑|软件|代码|互联网|系统|网络"),
        ("shopping_service", r"购买|下单|订单|付款|支付宝|客服|商家|价格|优惠|会员|快递|退款"),
        ("entertainment", r"电影|电视剧|综艺|游戏|KTV|音乐|歌曲|抖音|明星"),
        ("social_gossip", r"八卦|吃瓜|群里|听说|爆料|绷闻|公众瓜"),
    )
    for topic, pattern in topic_patterns:
        if re.search(pattern, text):
            return topic
    return "daily_chat"


def _humor_type(context: list[str], replies: list[str]) -> str:
    context_text = " ".join(context)
    reply_text = " ".join(replies)
    if re.fullmatch(r"(?:(?:哈){2,}[哈啊呀～~！!]*|h{3,})", reply_text, re.I):
        return "laughter"
    if re.search(r"我.*(?:韭菜|穷|废|不配|打工|进厂|躺平|没救)", reply_text):
        return "self_deprecation"
    if re.search(r"不行|不了|(?<!要)不要|算了|没法|不能", reply_text) and any(
        pattern.search(reply_text) for pattern in HUMOR_PATTERNS
    ):
        return "playful_refusal"
    if re.search(r"你|@(?:\[联系人\]|联系人)", reply_text) and any(
        pattern.search(reply_text) for pattern in HUMOR_PATTERNS
    ):
        return "teasing"
    if re.search(r"死了|疯了|炸了|起飞|天塌|一辈子|全世界|赚死|亏麻", reply_text):
        return "exaggeration"
    if re.fullmatch(r"(?:牛逼了?|tql|绝绝子|躺平)[!！~～]*", reply_text, re.I):
        return "none"
    if re.search(r"tql|绝绝子|韭菜|躺平|打工|进厂|牛逼|属实|破防", reply_text, re.I):
        return "meme"
    context_terms = set(_ for _ in re.findall(r"[\u4e00-\u9fff]{2,}", context_text))
    if context_terms and any(term in reply_text for term in context_terms) and any(
        pattern.search(reply_text) for pattern in HUMOR_PATTERNS
    ):
        return "callback"
    if any(pattern.search(reply_text) for pattern in HUMOR_PATTERNS):
        return "banter"
    return "none"


def _intent(context: list[str], replies: list[str]) -> str:
    context_text = " ".join(context)
    reply_text = " ".join(replies)
    if re.search(r"对不起|抱歉|不好意思", reply_text):
        return "apology"
    if re.search(r"恭喜|祝贺|生日快乐|新婚快乐", reply_text):
        return "congratulations"
    if re.search(r"不行|不了|(?<!要)不要|算了|没法|不能", reply_text):
        return "refuse"
    if re.search(r"别急|没事|抱抱|辛苦|会好的|理解|心疼", reply_text):
        return "empathy"
    if re.search(r"帮我|麻烦你|拜托|能不能帮|请你", reply_text):
        return "request_help"
    if re.search(r"[？?]|你呢|怎么|咋|什么|哪|几", reply_text):
        return "follow_up"
    if re.search(r"[？?]|怎么|咋|什么|哪|几|是否|能不能|可不可以", context_text):
        return "answer"
    if re.search(r"建议|不如|最好|可以试试|你应该|要不就", reply_text):
        return "advice"
    if re.search(r"收到|好的|好呀|可以|行吧|嗯嗯|知道了", reply_text):
        return "acknowledge"
    if re.search(r"今天|明天|晚上|上午|下午|周末|点见|到时候", context_text + reply_text):
        return "coordination"
    if re.search(r"晚安|回头聊|先这样|改天聊|我先忙", reply_text):
        return "closing"
    if any(pattern.search(reply_text) for pattern in HUMOR_PATTERNS):
        return "banter"
    return "comment"


def _response_mode(context: list[str], replies: list[str]) -> str:
    context_text = " ".join(context)
    reply_text = " ".join(replies)
    intent = _intent(context, replies)
    if _humor_type(context, replies) != "none":
        return "playful"
    context_bigrams = {context_text[index:index + 2] for index in range(max(0, len(context_text) - 1))}
    reply_bigrams = {reply_text[index:index + 2] for index in range(max(0, len(reply_text) - 1))}
    has_specific_overlap = bool(context_bigrams & reply_bigrams)
    strong_care = re.search(
        r"严重吗|医生怎么说|现在情况|还好吗|愿意说说|折腾这么久|"
        r"确实挺打击|听着就难受|换我也会|你现在主要是啥问题",
        reply_text,
    )
    contextual_care = has_specific_overlap and re.search(
        r"心疼|辛苦了|慢慢来|希望.{0,8}顺利|保重|注意身体",
        reply_text,
    )
    if strong_care or contextual_care or (intent == "empathy" and has_specific_overlap):
        return "sincere"
    if intent in {"answer", "advice", "coordination", "request_help", "acknowledge", "refuse"}:
        return "practical"
    return "neutral"


def _needs_sincere_response(context: list[str]) -> bool:
    text = " ".join(context)
    severe = re.search(r"流产|住院|去世|病危|失恋|刚分手|被分手|面试.{0,6}(?:挂|没过|失败)", text)
    personal_distress = any(
        not re.search(r"如果我|假如我|我觉得.{0,4}(?:不要|别).{0,4}焦虑|我不(?:焦虑|害怕|难受)", part)
        and re.search(
            r"我(?!们|同事|朋友|老公|老婆|男友|女友|npy).{0,8}(?:难受|受不了|崩溃|焦虑|害怕|失望|委屈|痛苦|想哭|撑不住|绝望|抑郁|"
            r"不想干了|累死了|忍不住了|太惨了|被裁|失业)|感觉自己好菜",
            part,
        )
        for part in context
    )
    playful_mask = re.search(r"笑死|哈哈|破涕为笑|旺柴|不怕兄弟|开路虎", text)
    return bool(severe or (personal_distress and not playful_mask))


def _reply_metadata(context: list[str], replies: list[str]) -> tuple[str, str, tuple[str, ...]]:
    reply_text = " ".join(replies)
    if len(replies) > 1:
        shape = "multi"
    elif re.fullmatch(r"(?:哈){2,}.*", reply_text):
        shape = "laugh"
    elif len(reply_text) <= 4:
        shape = "reaction"
    else:
        shape = "single"
    tags = []
    if any(pattern.search(reply_text) for pattern in HUMOR_PATTERNS):
        tags.append("humor")
    if "？" in reply_text or "?" in reply_text:
        tags.append("question")
    if len(replies) > 1:
        tags.append("multi_bubble")
    context_dependency = "high" if re.search(r"^(这|那|他|她|它|然后|所以|不是)", reply_text) else "low"
    return shape, context_dependency, tuple(tags)


def _sanitize(text: str, names: Iterable[str]) -> str:
    result = text.strip()
    for pattern, replacement in PLACEHOLDER_PATTERNS:
        result = pattern.sub(replacement, result)
    for name in sorted({n.strip() for n in names if 2 <= len(n.strip()) <= 30}, key=len, reverse=True):
        result = result.replace(name, "[联系人]")
    return re.sub(r"\s+", " ", result).strip()


def _wiki_relation(export_path: Path, wiki_dir: Path) -> str:
    stem = export_path.stem
    if stem.startswith("群聊_"):
        return "group"
    display_name = re.sub(r"^(私聊_|曾经的好友_)", "", stem)
    safe_display_name = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", display_name)
    candidates = [
        wiki_dir / f"{display_name}.md",
        wiki_dir / f"{safe_display_name}.md",
    ]
    text = ""
    for path in candidates:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")[:5000]
            break
    persona_pattern = re.escape(PERSONA_NAME)
    match = re.search(
        r"(?ms)^##\s*与\s*(?:Bot|" + persona_pattern + r")\s*的关系\s*$\n(.*?)(?=^##\s|\Z)",
        text,
    )
    relation_text = match.group(1) if match else ""
    if any(word in relation_text for word in FAMILY_WORDS):
        return "family"
    if any(word in relation_text for word in COLLEAGUE_WORDS):
        return "colleague"
    if any(word in relation_text for word in FRIEND_WORDS):
        return "friend"
    if any(word in f"{display_name}\n{relation_text}" for word in BUSINESS_WORDS):
        return "service"
    return "acquaintance"


def _score(context: list[str], replies: list[str]) -> float:
    reply = "".join(replies)
    score = 10.0 - abs(len(reply) - 14) * 0.08
    score += min(len(context), 3) * 0.5
    score += 1.0 if re.search(r"[哈嗯哦诶吧呀呢啊嘛～~]|[？?!！]", reply) else 0.0
    score -= max(0, len(reply) - 80) * 0.08
    score -= 2.0 if re.search(r"首先|其次|综上|总之|建议您|以下是", reply) else 0.0
    score += sum(1.8 for pattern in HUMOR_PATTERNS if pattern.search(reply))
    score -= 2.5 if re.fullmatch(r"(?:哈){2,}[哈啊呀～~！!]*", reply) else 0.0
    return score


def extract_candidates(export_dir: Path, wiki_dir: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for path in sorted(export_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        messages = payload.get("messages", [])
        relation = _wiki_relation(path, wiki_dir)
        chat_id = _stable_id(path.stem)
        session_name = str(payload.get("session", {}).get("displayName") or "")
        all_names = {session_name, re.sub(r"^(私聊_|群聊_|曾经的好友_)", "", path.stem)}
        all_names.update(str(m.get("senderDisplayName") or "") for m in messages)
        i = 0
        while i < len(messages):
            if not _is_text(messages[i]) or messages[i].get("isSend"):
                i += 1
                continue
            incoming: list[dict] = []
            while i < len(messages) and _is_text(messages[i]) and not messages[i].get("isSend") and len(incoming) < 5:
                incoming.append(messages[i])
                i += 1
            outgoing: list[dict] = []
            while i < len(messages) and _is_text(messages[i]) and messages[i].get("isSend") and len(outgoing) < 3:
                outgoing.append(messages[i])
                i += 1
            if not incoming or not outgoing:
                continue
            if not _is_coherent_turn(incoming, outgoing):
                continue
            if relation == "group":
                senders = {m.get("senderUsername") or m.get("senderDisplayName") for m in incoming}
                if len(senders) != 1:
                    continue
            raw_context = [str(m.get("content") or "") for m in incoming]
            raw_replies = [str(m.get("content") or "") for m in outgoing]
            if not all(_safe_text(t) for t in raw_context + raw_replies):
                continue
            context = [_sanitize(t, all_names) for t in raw_context]
            replies = [_sanitize(t, all_names) for t in raw_replies]
            if not all(context + replies):
                continue
            if _is_low_signal_reply("".join(replies)):
                continue
            fingerprint = re.sub(r"\W+", "", "|".join(context + replies)).lower()
            if len(fingerprint) < 3 or fingerprint in seen:
                continue
            seen.add(fingerprint)
            reply_shape, context_dependency, style_tags = _reply_metadata(context, replies)
            humor_type = _humor_type(context, replies)
            candidates.append(Candidate(
                relation=relation,
                chat_id=chat_id,
                context=context,
                replies=replies,
                score=_score(context, replies),
                timestamp=int(outgoing[0].get("createTime") or 0),
                intent=_intent(context, replies),
                reply_shape=reply_shape,
                context_dependency=context_dependency,
                style_tags=style_tags,
                topic=_topic(context, replies),
                humor_type=humor_type,
                response_mode=_response_mode(context, replies),
            ))
    return candidates


def extract_chat_backup(path: Path) -> list[Candidate]:
    """抽取 GlobalStore 聊天备份；仅 sender_type=self 视为本人真实发言。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    messages = payload.get("messages", [])
    chat_name = str(payload.get("chat_name") or path.stem)
    chat_id = _stable_id(chat_name)
    names = {chat_name}
    names.update(str(m.get("sender") or "") for m in messages if m.get("sender_type") != "self")
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for index, message in enumerate(messages):
        if message.get("sender_type") != "self" or message.get("message_type") != "text":
            continue
        reply = str(message.get("text") or "").strip()
        if not _safe_text(reply):
            continue
        incoming: list[str] = []
        cursor = index - 1
        while cursor >= 0 and len(incoming) < 5:
            previous = messages[cursor]
            if previous.get("sender_type") == "self":
                break
            text = str(previous.get("text") or "").strip()
            if previous.get("message_type") == "text" and _safe_text(text):
                incoming.append(text)
            cursor -= 1
        if not incoming:
            continue
        context = [_sanitize(text, names) for text in reversed(incoming)]
        replies = [_sanitize(reply, names)]
        if _is_low_signal_reply("".join(replies)):
            continue
        fingerprint = re.sub(r"\W+", "", "|".join(context + replies)).lower()
        if len(fingerprint) < 3 or fingerprint in seen:
            continue
        seen.add(fingerprint)
        reply_shape, context_dependency, style_tags = _reply_metadata(context, replies)
        humor_type = _humor_type(context, replies)
        candidates.append(Candidate(
            relation="group",
            chat_id=chat_id,
            context=context,
            replies=replies,
            score=_score(context, replies) + 6.0,
            timestamp=int(message.get("createTime") or 0),
            intent=_intent(context, replies),
            reply_shape=reply_shape,
            context_dependency=context_dependency,
            style_tags=style_tags,
            topic=_topic(context, replies),
            humor_type=humor_type,
            response_mode=_response_mode(context, replies),
            priority=True,
        ))
    return candidates


def select_balanced(candidates: list[Candidate], limit: int, group_target: int = 0) -> list[Candidate]:
    by_relation: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_relation[candidate.relation].append(candidate)
    for rows in by_relation.values():
        rows.sort(key=lambda c: (-c.score, -c.timestamp, c.chat_id))
    selected: list[Candidate] = []
    per_chat: Counter[str] = Counter()
    per_intent: Counter[tuple[str, str]] = Counter()
    per_shape: Counter[tuple[str, str]] = Counter()

    def pop_diverse(
        rows: list[Candidate],
        max_per_chat: int = 3,
        prefer_priority: bool = False,
    ) -> Candidate | None:
        eligible = [
            (index, candidate) for index, candidate in enumerate(rows)
            if per_chat[candidate.chat_id] < (15 if prefer_priority and candidate.priority else max_per_chat)
        ]
        if not eligible:
            rows.clear()
            return None
        index, candidate = min(
            eligible,
            key=lambda item: (
                not item[1].priority if prefer_priority else False,
                per_intent[(item[1].relation, item[1].intent)],
                per_shape[(item[1].relation, item[1].reply_shape)],
                -item[1].score,
                -item[1].timestamp,
                item[1].chat_id,
            ),
        )
        rows.pop(index)
        return candidate

    if group_target:
        group_rows = by_relation.get("group", [])
        group_rows.sort(key=lambda c: (not c.priority, -c.score, -c.timestamp, c.chat_id))
        while group_rows and len(selected) < min(group_target, limit):
            candidate = pop_diverse(group_rows, prefer_priority=True)
            if candidate is None:
                break
            selected.append(candidate)
            per_chat[candidate.chat_id] += 1
            per_intent[(candidate.relation, candidate.intent)] += 1
            per_shape[(candidate.relation, candidate.reply_shape)] += 1
    relations = ["family", "friend", "colleague", "acquaintance", "service"]
    if not group_target:
        relations.append("group")
    while len(selected) < limit:
        progressed = False
        for relation in relations:
            rows = by_relation.get(relation, [])
            candidate = pop_diverse(rows)
            if candidate is not None:
                selected.append(candidate)
                per_chat[candidate.chat_id] += 1
                per_intent[(candidate.relation, candidate.intent)] += 1
                per_shape[(candidate.relation, candidate.reply_shape)] += 1
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break

    return selected


def select_stratified(
    candidates: list[Candidate],
    limit: int,
    min_per_chat: int = 4,
    max_per_chat: int = 20,
    humor_ratio: float = 0.4,
    sincere_ratio: float = 0.1,
) -> list[Candidate]:
    """按聊天对象、场景和话题分层，控制幽默比例与单对象样本数。"""
    if limit <= 0 or min_per_chat <= 0 or max_per_chat < min_per_chat:
        return []
    humor_ratio = min(1.0, max(0.0, humor_ratio))
    sincere_ratio = min(0.3, max(0.0, sincere_ratio))
    by_chat: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_chat[candidate.chat_id].append(candidate)
    eligible_chats = [
        (chat_id, rows) for chat_id, rows in by_chat.items()
        if len(rows) >= min_per_chat
    ]
    eligible_chats.sort(key=lambda item: (
        -len({(row.intent, row.topic) for row in item[1]}),
        -len(item[1]),
        -max(row.timestamp for row in item[1]),
        item[0],
    ))
    target_per_chat = min(max_per_chat, max(min_per_chat, 8))
    target_chat_count = min(len(eligible_chats), math.ceil(limit / target_per_chat))
    pools = {
        chat_id: sorted(rows, key=lambda row: (not row.priority, -row.score, -row.timestamp))
        for chat_id, rows in eligible_chats[:target_chat_count]
    }
    chat_ids = list(pools)
    selected: list[Candidate] = []
    per_chat: Counter[str] = Counter()
    per_bucket: Counter[tuple[str, str, str]] = Counter()
    per_scenario: Counter[tuple[str, str]] = Counter()
    per_topic: Counter[tuple[str, str]] = Counter()
    humor_per_chat: Counter[str] = Counter()
    humor_count = 0

    def chat_humor_ratio(chat_id: str) -> float:
        relation = pools[chat_id][0].relation if pools[chat_id] else ""
        if relation in {"friend", "group"}:
            return min(0.55, humor_ratio + 0.05)
        if relation == "family":
            return min(humor_ratio, 0.35)
        if relation == "service":
            return min(humor_ratio, 0.2)
        return humor_ratio

    def choose(chat_id: str) -> Candidate | None:
        nonlocal humor_count
        want_humor = humor_per_chat[chat_id] < round(
            (per_chat[chat_id] + 1) * chat_humor_ratio(chat_id)
        )
        rows = pools[chat_id]
        eligible = [
            (index, row) for index, row in enumerate(rows)
            if per_bucket[(chat_id, row.intent, row.topic)] < 2
        ]
        if not eligible:
            return None
        index, candidate = min(eligible, key=lambda item: (
            (item[1].humor_type != "none") != want_humor,
            per_scenario[(chat_id, item[1].intent)],
            per_topic[(chat_id, item[1].topic)],
            not item[1].priority,
            -item[1].score,
            -item[1].timestamp,
        ))
        rows.pop(index)
        return candidate

    def add(candidate: Candidate) -> None:
        nonlocal humor_count
        selected.append(candidate)
        per_chat[candidate.chat_id] += 1
        per_bucket[(candidate.chat_id, candidate.intent, candidate.topic)] += 1
        per_scenario[(candidate.chat_id, candidate.intent)] += 1
        per_topic[(candidate.chat_id, candidate.topic)] += 1
        humor_count += candidate.humor_type != "none"
        humor_per_chat[candidate.chat_id] += candidate.humor_type != "none"

    for _ in range(min_per_chat):
        for chat_id in chat_ids:
            if len(selected) >= limit:
                return selected
            candidate = choose(chat_id)
            if candidate is not None:
                add(candidate)

    while len(selected) < limit:
        progressed = False
        for chat_id in chat_ids:
            if per_chat[chat_id] >= max_per_chat:
                continue
            candidate = choose(chat_id)
            if candidate is None:
                continue
            add(candidate)
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break

    target_humor_count = round(len(selected) * humor_ratio)
    relation_priority = {"group": 0, "friend": 1, "colleague": 2, "acquaintance": 3, "family": 4, "service": 5}
    rebalance_chats = sorted(
        chat_ids,
        key=lambda chat_id: (
            relation_priority.get(pools[chat_id][0].relation if pools[chat_id] else "", 9),
            humor_per_chat[chat_id],
        ),
    )
    while humor_count < target_humor_count:
        swapped = False
        for chat_id in rebalance_chats:
            incoming_rows = [
                (index, row) for index, row in enumerate(pools[chat_id])
                if row.humor_type != "none"
            ]
            outgoing_rows = [
                (index, row) for index, row in enumerate(selected)
                if row.chat_id == chat_id and row.humor_type == "none"
            ]
            if not incoming_rows or not outgoing_rows:
                continue
            incoming_rows.sort(key=lambda item: (
                per_scenario[(chat_id, item[1].intent)],
                per_topic[(chat_id, item[1].topic)],
                -item[1].score,
            ))
            outgoing_rows.sort(key=lambda item: (item[1].score, item[1].timestamp))
            replacement = None
            for incoming_index, incoming in incoming_rows:
                incoming_bucket = (chat_id, incoming.intent, incoming.topic)
                for outgoing_index, outgoing in outgoing_rows:
                    outgoing_bucket = (chat_id, outgoing.intent, outgoing.topic)
                    resulting_bucket_count = per_bucket[incoming_bucket] - (incoming_bucket == outgoing_bucket)
                    if resulting_bucket_count < 2:
                        replacement = (incoming_index, incoming, outgoing_index, outgoing)
                        break
                if replacement:
                    break
            if replacement is None:
                continue
            incoming_index, incoming, outgoing_index, outgoing = replacement
            pools[chat_id].pop(incoming_index)
            selected[outgoing_index] = incoming
            outgoing_bucket = (chat_id, outgoing.intent, outgoing.topic)
            incoming_bucket = (chat_id, incoming.intent, incoming.topic)
            per_bucket[outgoing_bucket] -= 1
            per_bucket[incoming_bucket] += 1
            per_scenario[(chat_id, outgoing.intent)] -= 1
            per_scenario[(chat_id, incoming.intent)] += 1
            per_topic[(chat_id, outgoing.topic)] -= 1
            per_topic[(chat_id, incoming.topic)] += 1
            humor_count += 1
            humor_per_chat[chat_id] += 1
            swapped = True
            if humor_count >= target_humor_count:
                break
        if not swapped:
            break

    sincere_count = sum(row.response_mode == "sincere" for row in selected)
    target_sincere_count = round(len(selected) * sincere_ratio)
    while sincere_count < target_sincere_count:
        swapped = False
        for chat_id in chat_ids:
            incoming_rows = [
                (index, row) for index, row in enumerate(pools[chat_id])
                if row.response_mode == "sincere"
            ]
            outgoing_rows = [
                (index, row) for index, row in enumerate(selected)
                if row.chat_id == chat_id and row.response_mode in {"neutral", "practical"}
            ]
            if not incoming_rows or not outgoing_rows:
                continue
            incoming_rows.sort(key=lambda item: (-item[1].score, -item[1].timestamp))
            outgoing_rows.sort(key=lambda item: (item[1].score, item[1].timestamp))
            replacement = None
            for incoming_index, incoming in incoming_rows:
                incoming_bucket = (chat_id, incoming.intent, incoming.topic)
                for outgoing_index, outgoing in outgoing_rows:
                    outgoing_bucket = (chat_id, outgoing.intent, outgoing.topic)
                    if per_bucket[incoming_bucket] - (incoming_bucket == outgoing_bucket) < 2:
                        replacement = (incoming_index, incoming, outgoing_index, outgoing)
                        break
                if replacement:
                    break
            if replacement is None:
                continue
            incoming_index, incoming, outgoing_index, outgoing = replacement
            pools[chat_id].pop(incoming_index)
            selected[outgoing_index] = incoming
            outgoing_bucket = (chat_id, outgoing.intent, outgoing.topic)
            incoming_bucket = (chat_id, incoming.intent, incoming.topic)
            per_bucket[outgoing_bucket] -= 1
            per_bucket[incoming_bucket] += 1
            sincere_count += 1
            swapped = True
            if sincere_count >= target_sincere_count:
                break
        if not swapped:
            break
    return selected


def split_temporal_holdout(
    candidates: list[Candidate],
    holdout_ratio: float = 0.2,
    min_train_per_chat: int = 8,
    min_holdout_per_chat: int = 2,
) -> tuple[list[Candidate], list[Candidate]]:
    """每个对象按时间切分，较新的回复只进入 holdout。"""
    holdout_ratio = min(0.5, max(0.05, holdout_ratio))
    by_chat: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_chat[candidate.chat_id].append(candidate)
    train: list[Candidate] = []
    holdout: list[Candidate] = []
    for rows in by_chat.values():
        ordered = sorted(rows, key=lambda row: (row.timestamp, row.context, row.replies))
        if len(ordered) < min_train_per_chat + min_holdout_per_chat:
            train.extend(ordered)
            continue
        holdout_count = max(min_holdout_per_chat, int(len(ordered) * holdout_ratio))
        holdout_count = min(holdout_count, len(ordered) - min_train_per_chat)
        train.extend(ordered[:-holdout_count])
        holdout.extend(ordered[-holdout_count:])
    return train, holdout


def select_holdout_cases(
    candidates: list[Candidate],
    limit: int,
    max_per_chat: int = 2,
) -> list[Candidate]:
    """从真实留出候选中覆盖不同对象、场景和话题。"""
    buckets: dict[tuple[str, str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        buckets[(candidate.intent, candidate.topic, candidate.response_mode)].append(candidate)
    rows_by_bucket = []
    for key, rows in sorted(buckets.items()):
        rows.sort(key=lambda row: (-row.timestamp, -row.score, row.chat_id))
        rows_by_bucket.append((key, rows))
    selected: list[Candidate] = []
    per_chat: Counter[str] = Counter()
    selected_ids: set[int] = set()
    sincerity_rows = sorted(
        (row for row in candidates if _needs_sincere_response(row.context)),
        key=lambda row: (-row.timestamp, -row.score),
    )
    for candidate in sincerity_rows:
        if len(selected) >= min(limit, 40) or per_chat[candidate.chat_id] >= max_per_chat:
            continue
        selected.append(candidate)
        selected_ids.add(id(candidate))
        per_chat[candidate.chat_id] += 1
    while len(selected) < limit:
        progressed = False
        for _, rows in rows_by_bucket:
            while rows and (
                per_chat[rows[0].chat_id] >= max_per_chat or id(rows[0]) in selected_ids
            ):
                rows.pop(0)
            if not rows:
                continue
            candidate = rows.pop(0)
            selected.append(candidate)
            selected_ids.add(id(candidate))
            per_chat[candidate.chat_id] += 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def write_holdout_cases(
    selected: list[Candidate],
    output_dir: Path,
    train_candidate_count: int,
    holdout_candidate_count: int,
) -> None:
    rows = []
    for index, candidate in enumerate(selected, 1):
        rows.append({
            "id": f"holdout_{index:03d}",
            "relationship": candidate.relation,
            "chat_id": candidate.chat_id,
            "context": candidate.context,
            "expected_reply": candidate.replies,
            "intent": candidate.intent,
            "topic": candidate.topic,
            "humor_type": candidate.humor_type,
            "response_mode": candidate.response_mode,
            "desired_response_mode": "sincere" if _needs_sincere_response(candidate.context) else "auto",
            "timestamp": candidate.timestamp,
        })
    (output_dir / "holdout_cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "train_candidate_count": train_candidate_count,
        "holdout_candidate_count": holdout_candidate_count,
        "selected_holdout_count": len(rows),
        "unique_chat_count": len({row["chat_id"] for row in rows}),
        "intent_counts": dict(sorted(Counter(row["intent"] for row in rows).items())),
        "topic_counts": dict(sorted(Counter(row["topic"] for row in rows).items())),
        "response_mode_counts": dict(sorted(Counter(row["response_mode"] for row in rows).items())),
        "desired_sincere_count": sum(row["desired_response_mode"] == "sincere" for row in rows),
    }
    (output_dir / "holdout_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_outputs(selected: list[Candidate], output_dir: Path, candidate_count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "persona_examples.jsonl"
    md_path = output_dir / "persona_examples.md"
    report_path = output_dir / "report.json"
    object_report_path = output_dir / "object_report.json"
    rows = []
    for index, candidate in enumerate(selected, 1):
        rows.append({
            "id": f"example_{index:03d}",
            "relationship": candidate.relation,
            "chat_id": candidate.chat_id,
            "context": candidate.context,
            "reply": candidate.replies,
            "intent": candidate.intent,
            "reply_shape": candidate.reply_shape,
            "context_dependency": candidate.context_dependency,
            "style_tags": list(candidate.style_tags),
            "topic": candidate.topic,
            "humor_type": candidate.humor_type,
            "response_mode": candidate.response_mode,
        })
    jsonl_content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    jsonl_path.write_text(jsonl_content, encoding="utf-8")
    examples_sha256 = hashlib.sha256(jsonl_content.encode("utf-8")).hexdigest()
    md_lines = ["# 本人真实回复 Few-shot（已脱敏）", "", "> 纯本地生成；请人工复核后再接入生产 prompt。", ""]
    for row in rows:
        md_lines.extend([
            f"## {row['id']} · {row['relationship']} · {row['topic']} / {row['intent']} · humor={row['humor_type']}", "",
            *[f"- 对方：{text}" for text in row["context"]],
            *[f"- 本人：{text}" for text in row["reply"]], "",
        ])
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    report = {
        "candidate_count": candidate_count,
        "selected_count": len(rows),
        "relationship_counts": dict(sorted(Counter(row["relationship"] for row in rows).items())),
        "intent_counts": dict(sorted(Counter(row["intent"] for row in rows).items())),
        "reply_shape_counts": dict(sorted(Counter(row["reply_shape"] for row in rows).items())),
        "topic_counts": dict(sorted(Counter(row["topic"] for row in rows).items())),
        "humor_type_counts": dict(sorted(Counter(row["humor_type"] for row in rows).items())),
        "response_mode_counts": dict(sorted(Counter(row["response_mode"] for row in rows).items())),
        "humor_ratio": round(sum(row["humor_type"] != "none" for row in rows) / len(rows), 4) if rows else 0,
        "unique_chat_count": len({row["chat_id"] for row in rows}),
        "samples_per_chat": {
            "min": min(Counter(row["chat_id"] for row in rows).values(), default=0),
            "max": max(Counter(row["chat_id"] for row in rows).values(), default=0),
            "average": round(len(rows) / len({row["chat_id"] for row in rows}), 2) if rows else 0,
        },
        "sensitive_pattern_scan_passed": not any(
            pattern.search(text)
            for row in rows
            for text in [*row["context"], *row["reply"]]
            for pattern in SENSITIVE_PATTERNS
        ),
        "review_status": "pending",
        "examples_sha256": examples_sha256,
        "external_model_used": False,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows_by_chat: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_chat[row["chat_id"]].append(row)
    object_report = {
        "object_count": len(rows_by_chat),
        "objects": {
            chat_id: {
                "relationship": chat_rows[0]["relationship"],
                "sample_count": len(chat_rows),
                "humor_count": sum(row["humor_type"] != "none" for row in chat_rows),
                "humor_ratio": round(sum(row["humor_type"] != "none" for row in chat_rows) / len(chat_rows), 4),
                "scenario_counts": dict(sorted(Counter(row["intent"] for row in chat_rows).items())),
                "topic_counts": dict(sorted(Counter(row["topic"] for row in chat_rows).items())),
                "scenario_topic_counts": dict(sorted(Counter(
                    f"{row['intent']} × {row['topic']}" for row in chat_rows
                ).items())),
                "example_ids": [row["id"] for row in chat_rows],
            }
            for chat_id, chat_rows in sorted(rows_by_chat.items())
        },
    }
    object_report_path.write_text(
        json.dumps(object_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/exports/b"))
    parser.add_argument("--wiki-dir", type=Path, default=Path("data/memory/wiki/users"))
    parser.add_argument("--output", type=Path, default=Path("data/few_shot"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--group-target", type=int, default=0)
    parser.add_argument("--chat-backup", type=Path, action="append", default=[])
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--min-per-chat", type=int, default=4)
    parser.add_argument("--max-per-chat", type=int, default=20)
    parser.add_argument("--humor-ratio", type=float, default=0.4)
    parser.add_argument("--sincere-ratio", type=float, default=0.1)
    parser.add_argument("--temporal-holdout", action="store_true")
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--holdout-limit", type=int, default=200)
    args = parser.parse_args()
    candidates = extract_candidates(args.input, args.wiki_dir)
    for backup in args.chat_backup:
        candidates.extend(extract_chat_backup(backup))
    holdout_candidates: list[Candidate] = []
    if args.temporal_holdout:
        candidates, holdout_candidates = split_temporal_holdout(
            candidates,
            holdout_ratio=args.holdout_ratio,
        )
    if args.stratified:
        selected = select_stratified(
            candidates,
            args.limit,
            min_per_chat=args.min_per_chat,
            max_per_chat=args.max_per_chat,
            humor_ratio=args.humor_ratio,
            sincere_ratio=args.sincere_ratio,
        )
    else:
        selected = select_balanced(candidates, args.limit, group_target=args.group_target)
    if len(selected) < args.limit:
        raise SystemExit(f"候选不足：需要 {args.limit}，实际 {len(selected)}")
    write_outputs(selected, args.output, len(candidates))
    if args.temporal_holdout:
        holdout_selected = select_holdout_cases(holdout_candidates, args.holdout_limit)
        write_holdout_cases(
            holdout_selected,
            args.output,
            train_candidate_count=len(candidates),
            holdout_candidate_count=len(holdout_candidates),
        )
    print(json.dumps({"candidates": len(candidates), "selected": len(selected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
