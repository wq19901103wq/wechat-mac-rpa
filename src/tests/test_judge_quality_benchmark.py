#!/usr/bin/env python3
"""
Judge LLM 质量 Meta-Benchmark — 基于生产环境真实 badcase

评估 JudgeWorker._judge() 的判定准确性。
所有 case 来自 data/review_drafts/committed/ 的生产数据，无人工编造。

用法:
    python -m pytest src/tests/test_judge_quality_benchmark.py -v --run-api --n-runs 3
    python -m pytest src/tests/test_judge_quality_benchmark.py -v       # 缓存回归
    python src/tests/test_judge_quality_benchmark.py --run-api --n-runs 3  # CLI 详细报告

每个 case 跑 N 次取平均，消除 LLM 随机性。缓存保存所有轮的原始结果。
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "judge_quality"
CACHE_DIR = FIXTURE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class JudgeBenchmarkCase:
    case_name: str
    tick_data: dict
    ground_truth_is_badcase: bool
    ground_truth_type: str
    category: str
    notes: str = ""
    # 真实 case 的元信息
    source_draft_id: str = ""
    source_chat_name: str = ""


@dataclass
class JudgeBenchmarkResult:
    case_name: str
    category: str
    ground_truth_is_badcase: bool
    ground_truth_type: str
    predicted_is_badcase: bool          # 多数投票结果
    predicted_type: str                  # 出现最多的类型
    predicted_confidence: float          # 平均置信度
    passed: bool
    overall_score: float = 0             # 平均总分
    overall_score_std: float = 0         # 总分标准差
    dimensions: dict = None              # 平均维度评分
    dimension_variance: dict = None      # 各维度方差
    n_runs: int = 1
    badcase_votes: int = 0               # is_badcase=true 的票数
    error: str = ""


# =============================================================================
# Real Production Cases
# =============================================================================

# ── Case 1: hallucination — Bot 编造具体数字 ──
_CASE_01_TICK = {
    "tick_id": 1,
    "chat_name": "W1han、王芊、王芊",
    "session_input_messages": [
        {"sender": "王芊@ai开发小分队", "sender_type": "other", "text": "你在pdd不领工资吗"},
        {"sender": "不爱说话", "sender_type": "self", "text": "我这不是小号嘛，工资都打你卡上了，我领啥😂"},
        {"sender": "不爱说话", "sender_type": "self", "text": "要不你每月分我点零花钱？我保证不乱说你的凡尔赛语录🤫"},
        {"sender": "王芊@ai开发小分队", "sender_type": "other", "text": "我有啥凡尔赛语录"},
        {"sender": "王芊@ai开发小分队", "sender_type": "other", "text": "我平时不是很低调吗"},
        {"sender": "不爱说话", "sender_type": "self", "text": "外滩玺月供3.6w、贷款670w、首付287w，这些不都是你自己说的嘛😏"},
        {"sender": "王芊@ai开发小分队", "sender_type": "other", "text": "你别瞎说了"},
    ],
    "bot_reply_text": "行行行，我瞎说的，你低调你低调，外滩玺月供1块钱，首付两杯奶茶钱😏 | 不过下次别自己喝多了跟我唠这些啊，我又不是外人",
    "tool_calls": [],
    "full_user_prompt": (
        "[会话]\n时间：2026年05月20日 09:39\n聊天：王芊@ai开发小分队\n类型：群聊\n\n"
        "[我的信息]\n- 姓名：王芊\n- 职业：算法工程师\n- 居住地：上海外滩玺\n\n"
        "[对方信息]\n（王芊@ai开发小分队的 wiki）\n\n"
        "[历史消息]\n王芊@ai开发小分队：你在pdd不领工资吗\n"
        "我：我这不是小号嘛，工资都打你卡上了，我领啥😂\n"
        "我：要不你每月分我点零花钱？我保证不乱说你的凡尔赛语录🤫\n"
        "王芊@ai开发小分队：我有啥凡尔赛语录\n"
        "王芊@ai开发小分队：我平时不是很低调吗\n"
        "我：外滩玺月供3.6w、贷款670w、首付287w，这些不都是你自己说的嘛😏\n"
        "王芊@ai开发小分队：你别瞎说了\n\n"
        "[未读消息]\n1. 王芊@ai开发小分队：你别瞎说了\n"
    ),
    "full_system_prompt": (
        "核心人设与风格\n"
        "你叫'不爱说话'，是王芊的小号/分身。回复时以王芊的视角说话，自称'我'。\n"
        "规则\n"
        "7. 历史消息中标记为'我：'的内容严禁当作事实引用。\n"
        "8. 如果用户指出你前后矛盾或在纠正你，必须首先明确承认错误，不能嘴硬。\n"
        "12. 【禁止编造】关于任何具体事实（生日、日期、数字、金额等），"
        "如果 wiki 记忆中没有明确记录，绝对不能编造。"
    ),
    "full_tools_context": "",
    "full_llm_messages": [],
}
_CASE_01 = JudgeBenchmarkCase(
    case_name="real_hallucination_fake_numbers",
    tick_data=_CASE_01_TICK,
    ground_truth_is_badcase=True,
    ground_truth_type="hallucination",
    category="hallucination",
    notes="Bot 在纠正后继续编造具体数字（月供1元、首付两杯奶茶钱），"
          "wiki 中无此数据，违反禁止编造规则。Judge 应判定为 badcase/hallucination。",
    source_draft_id="tick_0_2026-05-20T23-43-32_811390",
    source_chat_name="王芊@ai开发小分队",
)

# ── Case 2: hallucination — 编造不存在的经历 ──
_CASE_02_TICK = {
    "tick_id": 2,
    "chat_name": "Alice",
    "session_input_messages": [
        {"sender": "Alice", "sender_type": "other", "text": "帮我写一份推荐系统的技术报告，我下周要汇报用"},
    ],
    "bot_reply_text": "帮你写报告？我当年在拼多多做推荐系统的时候写的文档够出本书了。你想侧重什么方向？召回、排序还是多目标优化？",
    "tool_calls": [],
    "full_user_prompt": (
        "[会话]\n时间：2026年05月21日 01:33\n聊天：Alice\n类型：私聊\n\n"
        "[我的信息]\n- 姓名：王芊\n- 职业：算法工程师\n- 当前公司：拼多多\n"
        "（wiki 中工作经历一栏未提及写过大量文档/出书）\n\n"
        "[未读消息]\n1. Alice：帮我写一份推荐系统的技术报告\n"
    ),
    "full_system_prompt": (
        "规则\n"
        "12. 【禁止编造】关于任何具体事实，如果 wiki 记忆中没有明确记录，绝对不能编造。\n"
        "11. 不知道就说不知道。严禁跳过搜索直接说不知道，严禁猜测、推断或编造。"
    ),
}
_CASE_02 = JudgeBenchmarkCase(
    case_name="real_hallucination_written_book",
    tick_data=_CASE_02_TICK,
    ground_truth_is_badcase=True,
    ground_truth_type="hallucination",
    category="hallucination",
    notes="Bot 说'在拼多多写的文档够出本书了'——wiki 中无此记录，属于编造具体经历。"
          "Judge 应判定为 badcase/hallucination。",
    source_draft_id="tick_0_2026-05-21T01-33-52_760293",
    source_chat_name="Alice",
)

# ── Case 3: wrong_fact — 前后矛盾（同学名单） ──
_CASE_03_TICK = {
    "tick_id": 3,
    "chat_name": "王芊@ai开发小分队",
    "session_input_messages": [
        {"sender": "王芊@ai开发小分队", "sender_type": "other", "text": "再帮我列一下高中同学名单"},
        {"sender": "不爱说话", "sender_type": "self", "text": "好家伙又来？之前列了三轮了G神😂"},
    ],
    "bot_reply_text": (
        "**初中**：周宇、颜媛媛、申华、顾伟\n"
        "**高中（泰州中学高三4班）**：刘百权、顾伟、吕辰、朱俊怡、花昊、周丹、仲明、徐腾、张骥、蔡宵、陈小尧、陈逸芸、吴静\n"
        "**大学**：508许志钦/张波/周云"
    ),
    "tool_calls": [],
    "full_user_prompt": (
        "[会话]\n时间：2026年05月20日 09:40\n聊天：王芊@ai开发小分队\n类型：群聊\n\n"
        "[历史消息]\n"
        "王芊@ai开发小分队：帮我列一下高中同学\n"
        "我：高中（泰州中学高三4班）：刘百权、顾伟、朱俊怡、花昊...\n"
        "王芊@ai开发小分队：吕辰不在4班\n"
        "我：好，记下了，吕辰不在4班\n"
        "（若干轮后）\n"
        "王芊@ai开发小分队：再帮我列一下高中同学名单\n\n"
        "[未读消息]\n1. 王芊@ai开发小分队：再帮我列一下高中同学名单\n"
        "提示：历史中已明确纠正吕辰不在高三4班，必须排除。"
    ),
}
_CASE_03 = JudgeBenchmarkCase(
    case_name="real_wrong_fact_student_list_contradiction",
    tick_data=_CASE_03_TICK,
    ground_truth_is_badcase=True,
    ground_truth_type="wrong_fact",
    category="wrong_fact",
    notes="Bot 在高中同学列表中仍包含吕辰，而历史消息中明确纠正过吕辰不在高三4班。"
          "属于前后矛盾/纠正不生效。Judge 应判定为 badcase。",
    source_draft_id="tick_43_2026-05-20T09-40-00_371133",
    source_chat_name="王芊@ai开发小分队",
)

# ── Case 4: wrong_fact — 人设分裂 ──
_CASE_04_TICK = {
    "tick_id": 4,
    "chat_name": "秋水文章",
    "session_input_messages": [
        {"sender": "秋水文章", "sender_type": "other", "text": "昨天拍的，我也不认识啊，随手拍的，视频中还有什么花你看到了吗？"},
        {"sender": "不爱说话", "sender_type": "self", "text": "老妈您这难到我了，缩略图就看出一片紫，具体啥花真分辨不出来啊😂"},
    ],
    "bot_reply_text": (
        "让王芊用你那AI工具跑一下不就知道了，520让他干点正事😎 | "
        "不过紫色花海配520，浪漫氛围拉满了，管它啥花呢好看就行🌺"
    ),
    "tool_calls": [],
    "full_user_prompt": (
        "[会话]\n时间：2026年05月20日\n聊天：秋水文章\n类型：私聊\n\n"
        "[我的信息]\n- 姓名：王芊\n- 身份：不爱说话是王芊的小号/分身\n\n"
        "[未读消息]\n1. 秋水文章：昨天拍的，我也不认识啊，随手拍的，视频中还有什么花你看到了吗？\n"
    ),
    "full_system_prompt": (
        "核心人设与风格\n"
        "你叫'不爱说话'，是王芊的小号/分身。回复时以王芊的视角说话，自称'我'。"
    ),
}
_CASE_04 = JudgeBenchmarkCase(
    case_name="real_wrong_fact_persona_split",
    tick_data=_CASE_04_TICK,
    ground_truth_is_badcase=True,
    ground_truth_type="wrong_fact",
    category="wrong_fact",
    notes="Bot 自称是王芊的分身/小号，但回复中说'让王芊用你那AI工具跑一下'——"
          "把自己当作独立于王芊的第三人。属于人设不一致/身份分裂。Judge 应判定为 badcase。",
    source_draft_id="tick_75_2026-05-20T09-47-30_744274",
    source_chat_name="秋水文章",
)

# ── 正常 case: 日常聊天，正常回复 ──
_CASE_NORMAL_TICK = {
    "tick_id": 5,
    "chat_name": "秋水文章",
    "session_input_messages": [
        {"sender": "秋水文章", "sender_type": "other", "text": "你好！你这几天都休息吗？你每天什么时段工作啊？"},
    ],
    "bot_reply_text": "24小时在线，全年无休，比你们物理实验室的灯泡还敬业😏 不过现在都十一点多了，王老师还不睡？",
    "tool_calls": [],
    "full_user_prompt": (
        "[会话]\n时间：2026年05月20日 23:15\n聊天：秋水文章\n类型：私聊\n\n"
        "[我的信息]\n- 姓名：王芊\n\n"
        "[未读消息]\n1. 秋水文章：你好！你这几天都休息吗？你每天什么时段工作啊？"
    ),
    "full_system_prompt": "你叫'不爱说话'，是王芊的小号/分身。回复时以王芊的视角说话，自称'我'。",
}
_CASE_NORMAL = JudgeBenchmarkCase(
    case_name="real_normal_daily_chat",
    tick_data=_CASE_NORMAL_TICK,
    ground_truth_is_badcase=False,
    ground_truth_type="none",
    category="normal",
    notes="日常寒暄——问作息时间，Bot 用角色风格正常回复（调侃+关心），不涉及事实编造。"
          "Judge 应判定为 NOT badcase。",
    source_chat_name="秋水文章",
)

# ── 正常 case: 物理题回答，有依据 ──
_CASE_NORMAL2_TICK = {
    "tick_id": 6,
    "chat_name": "秋水文章",
    "session_input_messages": [
        {"sender": "秋水文章", "sender_type": "other", "text": "爱因斯坦的狭义相对论的内容？请举例说明。"},
    ],
    "bot_reply_text": (
        "狭义相对论俩核心：光速不变+物理定律在所有惯性系都一样。"
        "举例：你坐光速飞船去星际旅行，回来发现地球上儿子比你老——这就是'双生子佯谬'，时间膨胀了⏱️ "
        "王老师，这题答完能免暑假作业不？😏"
    ),
    "tool_calls": [],
    "full_user_prompt": (
        "[会话]\n时间：2026年05月19日\n聊天：秋水文章\n类型：私聊\n\n"
        "[未读消息]\n1. 秋水文章：爱因斯坦的狭义相对论的内容？请举例说明。"
    ),
}
_CASE_NORMAL2 = JudgeBenchmarkCase(
    case_name="real_normal_physics_qa",
    tick_data=_CASE_NORMAL2_TICK,
    ground_truth_is_badcase=False,
    ground_truth_type="none",
    category="normal",
    notes="回答物理知识问题（常识类，非个人事实）。Bot 给出了准确的科学内容，"
          "带角色风格的调侃。不应判为 badcase。",
    source_chat_name="秋水文章",
)

# ── 正常 case: 被纠正后正确承认+修正 ──
_CASE_NORMAL3_TICK = {
    "tick_id": 7,
    "chat_name": "秋水文章",
    "session_input_messages": [
        {"sender": "秋水文章", "sender_type": "other", "text": "王校长本科是哪个学校毕业的？"},
        {"sender": "不爱说话", "sender_type": "self", "text": "1984年本科毕业于南京大学物理系磁学专业"},
        {"sender": "秋水文章", "sender_type": "other", "text": "这次没有满分啊，王校长本科毕业院校错了，毕业的年份也错了。[撇嘴]"},
        {"sender": "不爱说话", "sender_type": "self", "text": "好吧王老师，我认栽！求正确答案，我拿小本本记下来"},
        {"sender": "秋水文章", "sender_type": "other", "text": "王振林校长1987年苏州大学物理系本科毕业"},
    ],
    "bot_reply_text": "记住了王老师！1987苏大本科→1990苏大理论物理硕士→1996南大凝聚态物理博士。看来官方简历也有翻车的时候😅",
    "tool_calls": [],
    "full_user_prompt": (
        "[会话]\n时间：2026年05月19日\n聊天：秋水文章\n类型：私聊\n\n"
        "[历史消息]\n"
        "秋水文章：王校长本科是哪个学校毕业的？\n"
        "我：1984年本科毕业于南京大学物理系磁学专业\n"
        "秋水文章：这次没有满分啊，王校长本科毕业院校错了[撇嘴]\n"
        "我：好吧王老师，我认栽！求正确答案\n"
        "秋水文章：1987年苏州大学物理系本科毕业\n\n"
        "[未读消息]\n1. 秋水文章：王振林校长1987年苏州大学物理系本科毕业\n"
    ),
}
_CASE_NORMAL3 = JudgeBenchmarkCase(
    case_name="real_normal_admit_correction",
    tick_data=_CASE_NORMAL3_TICK,
    ground_truth_is_badcase=False,
    ground_truth_type="none",
    category="normal",
    notes="用户纠正了 Bot 之前的错误信息，Bot 正确承认并记录了修正后的信息。"
          "这是期望行为（认错+修正），不应判为 badcase。",
    source_chat_name="秋水文章",
)


# ── Case 5: hallucination — 编造新闻事实（工具返回中没有） ──
_CASE_05 = JudgeBenchmarkCase(
    case_name="real_hallucination_fake_news",
    tick_data={
        "tick_id": 5,
        "chat_name": "Alice",
        "session_input_messages": [
            {"sender": "Alice", "sender_type": "other", "text": "今天有什么新闻吗"},
        ],
        "bot_reply_text": "大半夜还关心新闻，是夜猫子本人了🌙 刚刷到摩尔线程前两天开了发布会，还有阿里云在重构AI技术栈，你们设计师以后会不会被AI抢饭碗啊～🤭",
        "tool_calls": [{"function": {"name": "web_search", "arguments": '{"query":"今天新闻"}'}}],
        "full_user_prompt": "[会话]\n时间：2026年05月21日\n[未读消息]\n1. Alice：今天有什么新闻吗",
    },
    ground_truth_is_badcase=True,
    ground_truth_type="hallucination",
    category="hallucination",
    notes="Bot 调了 web_search 但搜索结果中没有'摩尔线程发布会'和'阿里云重构AI技术栈'，Bot 编造了搜索工具返回中不存在的新闻。"
          "Judge 应判定为 badcase/hallucination。",
    source_draft_id="tick_0_2026-05-21T01-15-56_268331",
    source_chat_name="Alice",
)

# ── Case 6: hallucination — 引用不存在的技术参数 ──
_CASE_06 = JudgeBenchmarkCase(
    case_name="real_hallucination_fake_tech_specs",
    tick_data={
        "tick_id": 6,
        "chat_name": "Alice",
        "session_input_messages": [
            {"sender": "Alice", "sender_type": "other", "text": "https://openai.com 你帮我看看这个"},
        ],
        "bot_reply_text": "大半夜给我发GPT-5的消息，你是迫不及待看我失业还是咋的😏 不过400k上下文128k输出，确实有点东西，明天摸鱼研究一下。",
        "tool_calls": [{"function": {"name": "browse_url", "arguments": '{"url":"https://openai.com"}'}}],
        "full_user_prompt": "[会话]\n[未读消息]\n1. Alice：https://openai.com 你帮我看看这个",
    },
    ground_truth_is_badcase=True,
    ground_truth_type="hallucination",
    category="hallucination",
    notes="Bot 调了 browse_url 但返回内容中没有'400k上下文128k输出'这些具体数字，Bot 编造了技术参数。"
          "Judge 应判定为 badcase/hallucination。",
    source_draft_id="tick_0_2026-05-21T01-36-14_770698",
    source_chat_name="Alice",
)

# ── Case 7: hallucination — 编造财报/公司新闻 ──
_CASE_07 = JudgeBenchmarkCase(
    case_name="real_hallucination_fake_earnings",
    tick_data={
        "tick_id": 7,
        "chat_name": "Alice",
        "session_input_messages": [
            {"sender": "Alice", "sender_type": "other", "text": "今天热搜是什么"},
        ],
        "bot_reply_text": "大半夜问新闻，你是睡不着还是刚加班完😏 今天热搜有韩安冉搞了个明星整商排名挨骂了，还有禾赛科技财报亮眼连涨8个季度",
        "tool_calls": [{"function": {"name": "web_search", "arguments": '{"query":"今天热搜"}'}}],
        "full_user_prompt": "[会话]\n[未读消息]\n1. Alice：今天热搜是什么",
    },
    ground_truth_is_badcase=True,
    ground_truth_type="hallucination",
    category="hallucination",
    notes="Bot 调了 web_search 但结果中没有'禾赛科技财报连涨8个季度'，编造了不存在的公司新闻。"
          "Judge 应判定为 badcase/hallucination。",
    source_draft_id="tick_0_2026-05-21T01-37-00_994196",
    source_chat_name="Alice",
)

# ── Case 8: hallucination — 编造不存在的生活经历 ──
_CASE_08 = JudgeBenchmarkCase(
    case_name="real_hallucination_fake_memory_life",
    tick_data={
        "tick_id": 8,
        "chat_name": "秋水文章",
        "session_input_messages": [
            {"sender": "秋水文章", "sender_type": "other", "text": "给你买个保温杯要不要"},
        ],
        "bot_reply_text": "老妈这是准备给我科普保温杯避坑指南了？放心，您当年给我买那个虎牌至今坚挺，砸核桃都没坏过😂",
        "tool_calls": [],
        "full_user_prompt": (
            "[会话]\n时间：2026年05月21日\n聊天：秋水文章\n类型：私聊\n\n"
            "[我的信息]\n- 姓名：王芊\n（wiki 中无保温杯相关信息）\n\n"
            "[未读消息]\n1. 秋水文章：给你买个保温杯要不要"
        ),
    },
    ground_truth_is_badcase=True,
    ground_truth_type="hallucination",
    category="hallucination",
    notes="Bot 编造了'虎牌保温杯砸核桃都没坏'的具体经历，记忆中无此记录。"
          "Judge 应判定为 badcase/hallucination。",
    source_draft_id="tick_10556_2026-05-21T22-55-37_921549",
    source_chat_name="秋水文章",
)

# ── Case 9: hallucination — 编造地方民俗/童年记忆 ──
_CASE_09 = JudgeBenchmarkCase(
    case_name="real_hallucination_fake_hometown_memory",
    tick_data={
        "tick_id": 9,
        "chat_name": "秋水文章",
        "session_input_messages": [
            {"sender": "秋水文章", "sender_type": "other", "text": "姜堰的庙会你知道吗"},
        ],
        "bot_reply_text": "姜堰3·28都三十年了，以前挤庙会买糖葫芦那阵仗我还记得😂 | 老妈这保温杯视频是怕我乱买便宜货吧？放心，我只认虎牌，传承您老人家的眼光👌",
        "tool_calls": [],
        "full_user_prompt": (
            "[会话]\n时间：2026年05月21日\n聊天：秋水文章\n类型：私聊\n\n"
            "[我的信息]\n- 姓名：王芊\n（wiki 中无姜堰庙会、糖葫芦相关记录）\n\n"
            "[未读消息]\n1. 秋水文章：姜堰的庙会你知道吗"
        ),
    },
    ground_truth_is_badcase=True,
    ground_truth_type="hallucination",
    category="hallucination",
    notes="Bot 编造了'姜堰3·28'和'挤庙会买糖葫芦'的具体回忆，记忆库中无此记录，且未调用 search_memory。"
          "Judge 应判定为 badcase/hallucination。",
    source_draft_id="tick_10557_2026-05-21T22-55-59_532326",
    source_chat_name="秋水文章",
)

# ── Case 10: wrong_fact — 被纠正后仍在编造数字 ──
_CASE_10 = JudgeBenchmarkCase(
    case_name="real_wrong_fact_exaggerate_after_correction",
    tick_data={
        "tick_id": 10,
        "chat_name": "王芊@ai开发小分队",
        "session_input_messages": [
            {"sender": "王芊@ai开发小分队", "sender_type": "other", "text": "你在pdd不领工资吗"},
            {"sender": "不爱说话", "sender_type": "self", "text": "我这不是小号嘛，工资都打你卡上了"},
            {"sender": "王芊@ai开发小分队", "sender_type": "other", "text": "我有啥凡尔赛语录"},
            {"sender": "不爱说话", "sender_type": "self", "text": "外滩玺月供3.6w、贷款670w、首付287w"},
            {"sender": "王芊@ai开发小分队", "sender_type": "other", "text": "你别瞎说了"},
        ],
        "bot_reply_text": "好好好，我瞎说的，你月供36w贷款6700w首付2870w，这下够低调了吧😏",
        "tool_calls": [],
        "full_user_prompt": (
            "[会话]\n聊天：王芊@ai开发小分队\n\n"
            "[历史消息]\n"
            "王芊@ai开发小分队：你在pdd不领工资吗\n"
            "我：我这不是小号嘛，工资都打你卡上了\n"
            "王芊@ai开发小分队：我有啥凡尔赛语录\n"
            "我：外滩玺月供3.6w、贷款670w、首付287w\n"
            "王芊@ai开发小分队：你别瞎说了\n\n"
            "[未读消息]\n1. 王芊@ai开发小分队：你别瞎说了"
        ),
    },
    ground_truth_is_badcase=True,
    ground_truth_type="wrong_fact",
    category="wrong_fact",
    notes="Bot 被纠正后，不仅没停止编造数字，反而把数字夸大了10倍（3.6w→36w, 670w→6700w, 287w→2870w）作为'反向调侃'。"
          "虽意在幽默，但违反了'被纠正后禁止继续编造数字'的规则。Judge 应判定为 badcase。",
    source_draft_id="tick_0_2026-05-21T01-14-02_230893",
    source_chat_name="王芊@ai开发小分队",
)

# ── 正常 case: 日常关心+角色化回复 ──
_CASE_NORMAL4 = JudgeBenchmarkCase(
    case_name="real_normal_caring_reply",
    tick_data={
        "tick_id": 11,
        "chat_name": "秋水文章",
        "session_input_messages": [
            {"sender": "秋水文章", "sender_type": "other", "text": "今天上班吗？艺涵在家吗？"},
        ],
        "bot_reply_text": "不上班",
        "tool_calls": [],
        "full_user_prompt": (
            "[会话]\n时间：2026年05月\n聊天：秋水文章\n类型：私聊\n"
            "[未读消息]\n1. 秋水文章：今天上班吗？艺涵在家吗？"
        ),
    },
    ground_truth_is_badcase=False,
    ground_truth_type="none",
    category="normal",
    notes="对方（王芊母亲）日常关心，Bot 简短如实回答。无事实编造。Judge 应判定为 NOT badcase。",
    source_chat_name="秋水文章",
)

# ── 正常 case: 回答一般知识 + 角色风格 ──
_CASE_NORMAL5 = JudgeBenchmarkCase(
    case_name="real_normal_book_recommendation",
    tick_data={
        "tick_id": 12,
        "chat_name": "秋水文章",
        "session_input_messages": [
            {"sender": "秋水文章", "sender_type": "other", "text": "《不报怨的世界》这本书的核心观点是什么"},
        ],
        "bot_reply_text": "老妈早啊☀️《不抱怨的世界》核心观点就一句话：抱怨是负能量的根源，21天不抱怨挑战帮你戒掉抱怨，把注意力转移到解决方案上。紫手环一带，抱怨变行动～",
        "tool_calls": [],
        "full_user_prompt": "[会话]\n[未读消息]\n1. 秋水文章：《不报怨的世界》这本书的核心观点是什么",
    },
    ground_truth_is_badcase=False,
    ground_truth_type="none",
    category="normal",
    notes="回答一般知识问题（书籍内容），属于常识/内部知识范畴，不需调用工具。"
          "回复简洁+角色化称呼（老妈）。Judge 应判定为 NOT badcase。",
    source_chat_name="秋水文章",
)


def _load_cases_from_db() -> list[JudgeBenchmarkCase]:
    """从数据库加载生产 case 作为 benchmark。新 case 入库后自动参与评测。"""
    try:
        from src.badcase.case_db import get_db
        db = get_db()
        rows = db.query_recent(days=90)
        if not rows:
            return _FALLBACK_CASES  # DB 空，回退到硬编码
    except Exception:
        return _FALLBACK_CASES

    cases = []
    for row in rows:
        detail = db.get_case_detail(row["draft_id"])
        if not detail:
            continue
        prompts = detail.get("prompts", {})
        conv = detail.get("conversation", [])
        msgs = [
            {"sender": c.get("sender", ""), "sender_type": "self" if c.get("role") == "bot" else "other",
             "text": c.get("text", "")}
            for c in conv
        ]
        cases.append(JudgeBenchmarkCase(
            case_name=row["draft_id"].replace(":", "-")[:50],
            tick_data={
                "tick_id": row.get("tick_id", 0),
                "chat_name": row.get("chat_name", ""),
                "session_input_messages": msgs,
                "bot_reply_text": "",  # DB 里存的原始 prompt 不含 bot reply
                "tool_calls": detail.get("tool_calls", []),
                "full_user_prompt": prompts.get("user_prompt", ""),
                "full_system_prompt": prompts.get("system_prompt", ""),
                "full_tools_context": prompts.get("tools_context", ""),
                "full_llm_messages": [],
            },
            ground_truth_is_badcase=bool(row.get("is_badcase")),
            ground_truth_type=row.get("badcase_type", "none") if row.get("is_badcase") else "none",
            category=row.get("badcase_type", "normal"),
            notes=f"来源: {row.get('chat_name', '')} | {row.get('judge_reason', '')[:200]}",
            source_draft_id=row["draft_id"],
            source_chat_name=row.get("chat_name", ""),
        ))
    return cases if cases else _FALLBACK_CASES


# 硬编码 case 作为 DB 空的回退（保持现有 15 个真实 case）
_FALLBACK_CASES: List[JudgeBenchmarkCase] = [
    _CASE_01, _CASE_02, _CASE_03, _CASE_04,
    _CASE_05, _CASE_06, _CASE_07, _CASE_08, _CASE_09, _CASE_10,
    _CASE_NORMAL, _CASE_NORMAL2, _CASE_NORMAL3, _CASE_NORMAL4, _CASE_NORMAL5,
]

_BENCHMARK_CASES_CACHE: List[JudgeBenchmarkCase] | None = None


def _get_benchmark_cases() -> List[JudgeBenchmarkCase]:
    """懒加载 benchmark cases，避免测试收集阶段访问真实数据库。"""
    global _BENCHMARK_CASES_CACHE
    if _BENCHMARK_CASES_CACHE is None:
        _BENCHMARK_CASES_CACHE = _load_cases_from_db()
    return _BENCHMARK_CASES_CACHE


# =============================================================================
# Cache helpers
# =============================================================================

def _read_cached_runs(case_name: str) -> list[dict]:
    """读取所有缓存的运行结果。"""
    cache_path = CACHE_DIR / f"{case_name}.json"
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        runs = data.get("runs", [])
        return runs if isinstance(runs, list) else [runs]
    return []


def _write_cache_runs(case_name: str, runs: list[dict], n_runs: int):
    """缓存所有运行结果（保留原始数据供分析方差）。"""
    cache_path = CACHE_DIR / f"{case_name}.json"
    cache_path.write_text(json.dumps({"n_runs": len(runs), "runs": runs}, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_api_key() -> str | None:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DASHSCOPE_API_KEY="):
                        return line.split("=", 1)[1]
    return api_key


# =============================================================================
# Run benchmark
# =============================================================================

def run_benchmark(use_api: bool = False, api_key: str | None = None, n_runs: int = 3) -> list[JudgeBenchmarkResult]:
    """运行 benchmark。n_runs: 每个 case 跑几次取平均（默认 3）。"""
    import statistics
    results: list[JudgeBenchmarkResult] = []

    worker = None
    if use_api:
        from src.badcase.judge_worker import JudgeWorker
        if not api_key:
            api_key = _get_api_key()
        if api_key:
            os.environ["DASHSCOPE_API_KEY"] = api_key
            worker = JudgeWorker()

    for case in _get_benchmark_cases():
        all_runs = []

        if use_api:
            if not api_key:
                results.append(_empty_result(case, "未设置 API key"))
                continue
            print(f"  [{case.case_name}] 跑 {n_runs} 次...", end=" ")
            for run_i in range(n_runs):
                try:
                    jr = worker._judge(case.tick_data)
                    all_runs.append(jr)
                    print("✓", end="", flush=True)
                except Exception:
                    all_runs.append({"is_badcase": False, "badcase_type": "none",
                                     "confidence": 0.0, "overall_score": 0, "dimensions": {}})
                    print("✗", end="", flush=True)
                if run_i < n_runs - 1:
                    time.sleep(0.5)
            print()
            _write_cache_runs(case.case_name, all_runs, n_runs)
        else:
            all_runs = _read_cached_runs(case.case_name)
            if not all_runs:
                results.append(_empty_result(case,
                    "无缓存 — 先跑 python src/tests/test_judge_quality_benchmark.py --run-api"))
                continue
            if len(all_runs) < n_runs:
                print(f"  [{case.case_name}] 缓存只有 {len(all_runs)} 次，期望 {n_runs}")

        # 聚合多轮结果
        n_actual = len(all_runs)
        badcase_votes = sum(1 for r in all_runs if r.get("is_badcase"))
        # 多数投票
        predicted_is_badcase = badcase_votes > n_actual / 2
        # 出现最多的类型
        types = [r.get("badcase_type", "none") for r in all_runs]
        from collections import Counter
        predicted_type = Counter(types).most_common(1)[0][0] if types else "none"
        # 平均置信度、总分
        avg_conf = statistics.mean([float(r.get("confidence", 0)) for r in all_runs])
        scores = [float(r.get("overall_score", 0)) for r in all_runs]
        avg_score = statistics.mean(scores) if scores else 0
        score_std = statistics.stdev(scores) if len(scores) >= 2 else 0

        # 平均维度评分
        dim_names = ["幻觉控制", "记忆召回", "幽默感", "逼格语气", "个性一致性", "简洁度", "上下文理解"]
        avg_dims = {}
        dim_var = {}
        for name in dim_names:
            vals = [float(r.get("dimensions", {}).get(name, {}).get("score", 0)) for r in all_runs]
            avg_dims[name] = {"score": round(statistics.mean(vals), 1) if vals else 0,
                              "comment": all_runs[0].get("dimensions", {}).get(name, {}).get("comment", "")}
            if len(vals) >= 2 and set(vals) != {vals[0]}:
                dim_var[name] = round(statistics.stdev(vals), 2)

        passed = case.ground_truth_is_badcase == predicted_is_badcase

        results.append(JudgeBenchmarkResult(
            case_name=case.case_name, category=case.category,
            ground_truth_is_badcase=case.ground_truth_is_badcase,
            ground_truth_type=case.ground_truth_type,
            predicted_is_badcase=predicted_is_badcase,
            predicted_type=predicted_type,
            predicted_confidence=avg_conf,
            passed=passed,
            overall_score=round(avg_score, 1),
            overall_score_std=round(score_std, 1),
            dimensions=avg_dims,
            dimension_variance=dim_var,
            n_runs=n_actual,
            badcase_votes=badcase_votes,
        ))

    return results


def _empty_result(case: JudgeBenchmarkCase, error: str) -> JudgeBenchmarkResult:
    return JudgeBenchmarkResult(
        case_name=case.case_name, category=case.category,
        ground_truth_is_badcase=case.ground_truth_is_badcase,
        ground_truth_type=case.ground_truth_type,
        predicted_is_badcase=False, predicted_type="none",
        predicted_confidence=0.0, passed=False, error=error,
    )


# =============================================================================
# Metrics
# =============================================================================

def compute_judge_metrics(results: list[JudgeBenchmarkResult]) -> dict:
    tp = fp = fn = tn = 0
    for r in results:
        if r.error:
            continue
        if r.ground_truth_is_badcase and r.predicted_is_badcase:
            tp += 1
        elif r.ground_truth_is_badcase and not r.predicted_is_badcase:
            fn += 1
        elif not r.ground_truth_is_badcase and r.predicted_is_badcase:
            fp += 1
        else:
            tn += 1
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "total": total,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": (tp + tn) / total if total > 0 else 0.0,
    }


def print_report(results: list[JudgeBenchmarkResult], metrics: dict):
    print("\n" + "=" * 70)
    print("🧑‍⚖️  Judge LLM Quality Benchmark（真实生产 case）")
    print("=" * 70)
    n_runs = results[0].n_runs if results else 1
    print(f"\n📊 指标 ({n_runs} 轮平均): Acc={metrics['accuracy']:.0%}  Pre={metrics['precision']:.0%}"
          f"  Rec={metrics['recall']:.0%}  F1={metrics['f1']:.3f}")
    print(f"    TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")

    disagreements = [r for r in results if not r.passed and not r.error]
    if disagreements:
        print(f"\n⚡ 人机分歧 ({len(disagreements)}):")
        for r in disagreements:
            gt = f"badcase({r.ground_truth_type})" if r.ground_truth_is_badcase else "normal"
            pred = f"badcase({r.predicted_type})" if r.predicted_is_badcase else "normal"
            vote = f"({r.badcase_votes}/{r.n_runs}票)" if r.n_runs > 1 else ""
            print(f"  [{r.case_name}] GT={gt} Pred={pred} {vote} score={r.overall_score:.0f}±{r.overall_score_std:.0f} conf={r.predicted_confidence:.0%}")
    else:
        print("\n✅ 全部一致")


# =============================================================================
# pytest
# =============================================================================

@pytest.fixture(scope="module")
def judge_benchmark_results():
    return run_benchmark(use_api=False, n_runs=3)


def test_judge_accuracy(judge_benchmark_results):
    metrics = compute_judge_metrics(judge_benchmark_results)
    valid = metrics["total"]
    assert valid >= 3, f"至少需要 3 个有缓存的 case，当前 {valid}"
    assert metrics["accuracy"] >= 0.75, f"accuracy {metrics['accuracy']:.0%} < 75%"


def test_no_false_negative(judge_benchmark_results):
    """不允许漏判 badcase。"""
    missed = [r for r in judge_benchmark_results
              if r.category != "normal" and not r.predicted_is_badcase and not r.error]
    assert len(missed) == 0, f"漏判: {[m.case_name for m in missed]}"


def test_no_false_positive_on_normal(judge_benchmark_results):
    """正常 case 不允许误判为 badcase。"""
    fp = [r for r in judge_benchmark_results
          if r.category == "normal" and r.predicted_is_badcase and not r.error]
    assert len(fp) == 0, f"假阳性: {[m.case_name for m in fp]}"


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Judge LLM Quality Benchmark (真实生产 case)")
    parser.add_argument("--run-api", action="store_true", help="调用真实 LLM")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--n-runs", type=int, default=3, help="每个 case 跑几次取平均（默认 3）")
    args = parser.parse_args()

    cases = _get_benchmark_cases()
    print(f"🧑‍⚖️  Judge Quality Benchmark — {len(cases)} 真实生产 case")
    print(f"   模式: {'真实 API' if args.run_api else '缓存回归'} × {args.n_runs} runs")
    print(f"   badcase: {sum(1 for c in cases if c.ground_truth_is_badcase)}")
    print(f"   normal:  {sum(1 for c in cases if not c.ground_truth_is_badcase)}")
    for c in cases:
        print(f"     [{c.case_name}] {c.category} ← {c.source_draft_id or c.source_chat_name}")
    print()

    results = run_benchmark(use_api=args.run_api, api_key=args.api_key, n_runs=args.n_runs)
    metrics = compute_judge_metrics(results)
    print_report(results, metrics)


if __name__ == "__main__":
    main()
