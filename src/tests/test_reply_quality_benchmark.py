#!/usr/bin/env python3
"""
回复质量 Benchmark - 真实 LLM 评测（冻结标准版）

验证 ReplyGenerator 在真实 LLM 驱动下的回复质量。
使用 MockMemoryEngine 提供固定记忆输入，真实 QwenClient 驱动生成。

核心设计原则：
1. 冻结标准：所有 case 的 required_keywords、forbidden_keywords、min/max_replies
   在代码中硬编码，绝不根据 LLM 输出调整。
2. 真实 LLM：使用 QwenClient(model="deepseek-v4-flash") 调用真实 API。
3. 可复现：MockMemoryEngine 提供固定输入，缓存机制保存 replies。
4. 事实准确性：纠正场景要求 Bot 承认错误，不能嘴硬。

运行方式:
    # 调用真实 LLM（生成缓存）
    python src/tests/test_reply_quality_benchmark.py --run-api

    # 使用缓存回归
    pytest src/tests/test_reply_quality_benchmark.py -v
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.base import ChatMessage, SenderType  # noqa: E402
from src.reply.generator import ReplyGenerator  # noqa: E402
from src.tools import get_registry, register_builtin_tools  # noqa: E402
from src.utils.qwen_client import QwenClient  # noqa: E402

# 测试用全局工具注册表
_TEST_TOOL_REGISTRY = get_registry()
register_builtin_tools(_TEST_TOOL_REGISTRY)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "reply_quality"


# =============================================================================
# MockMemoryEngine：提供固定、可控的记忆输入
# =============================================================================

class MockMemoryEngine:
    """提供固定记忆内容，确保每次评测输入一致。"""

    def get_user_memory(self, user_name: str, max_chars: int = 2000) -> str:
        if user_name == "王芊":
            return (
                "- 姓名：王芊\n"
                "- 职业：算法工程师\n"
                "- 居住地：上海外滩玺\n"
                "- 配偶：王艺涵（在阿里1688做推荐策略）\n"
                "- 工作经历：腾讯→拼多多"
            )
        if user_name == "Alice":
            return "- 姓名：Alice\n- 职业：设计师\n"
        return ""

    def get_group_memory(self, group_name: str, max_chars: int = 2000) -> str:
        return ""

    def search_keyword(self, query: str) -> str:
        if "王芊" in query:
            return (
                "【王芊的记忆】姓名：王芊，职业：算法工程师，"
                "居住地：上海外滩玺，配偶：王艺涵"
            )
        if "程立" in query:
            return (
                "【程立-君奕的记忆】姓名：程立，职业：算法工程师，"
                "与王芊是同事（拼多多），别名：盔哥"
            )
        return "未找到"

    def search_related_mentions(self, text: str, exclude_user=None, max_files: int = 5) -> List[str]:
        return []


# =============================================================================
# Rubric-Based Evaluation (LLM-as-a-Judge)
# =============================================================================

@dataclass
class RubricDimension:
    """评分维度"""
    name: str              # 维度名称，如"事实纠正"
    description: str       # 维度描述
    criteria: str          # 具体评判标准
    required: bool = True  # 是否必须通过（影响 overall）


@dataclass
class Rubric:
    """评分标准（Rubric）"""
    instructions: str      # 给 Judge 的整体指令/背景说明
    dimensions: List[RubricDimension] = field(default_factory=list)

    def to_judge_prompt(self, context: str, replies: List[str], case_notes: str = "") -> str:
        """生成给 Judge LLM 的完整 prompt"""
        replies_text = "\n".join(f"[{i+1}] {r}" for i, r in enumerate(replies)) if replies else "(无回复)"

        dims_text = "\n\n".join(
            f"### {i+1}. {d.name}\n"
            f"**描述**: {d.description}\n"
            f"**标准**: {d.criteria}\n"
            f"**重要性**: {'必须通过' if d.required else '参考项（不决定通过/失败）'}"
            for i, d in enumerate(self.dimensions)
        )

        notes_section = f"\n## Case 备注\n{case_notes}\n" if case_notes else ""

        return f"""你是一位专业的对话质量评估专家。请根据以下评分标准，客观评估 Bot 的回复质量。

## 评估原则
- 只根据评分标准和对话上下文做判断，不加入个人偏好
- 注意否定语境：如果关键词出现在否定/纠正语境中（如"287万是我瞎编的"），不应视为错误
- 区分"承认错误"和"确认错误"：Bot 说"我错了"是承认，说"是的，287万"是确认错误信息
- 保持客观，给出具体理由

## 对话上下文
{context}

## Bot 的回复
{replies_text}
{notes_section}
## 评分标准
{dims_text}

## 评分规则
- 每个维度给出 PASS（通过）或 FAIL（失败）
- overall = PASS 当且仅当所有 required=True 的维度都通过
- 请先用简短文字分析每个维度，然后给出 JSON

## 输出格式（必须是纯 JSON，不要 markdown 代码块）
{{
  "dimensions": [
    {{"name": "维度名", "score": "PASS|FAIL", "reason": "具体理由，引用回复内容作为证据"}}
  ],
  "overall": "PASS|FAIL",
  "explanation": "整体评估总结"
}}"""


class JudgeLLM:
    """LLM-as-a-Judge 评判器（使用 deepseek-v4-pro）"""

    def __init__(self, model: str = "deepseek-v4-pro", temperature: float = 0.1, api_key: str | None = None):
        # 如果传入了 api_key，临时设置到环境变量
        if api_key and not os.environ.get("DASHSCOPE_API_KEY"):
            os.environ["DASHSCOPE_API_KEY"] = api_key
        self.client = QwenClient(model=model)
        self.temperature = temperature

    def evaluate(self, rubric: Rubric, context: str, replies: List[str],
                 case_notes: str = "") -> Dict[str, Any]:
        """调用 Judge LLM 评估回复，返回结构化结果"""
        prompt = rubric.to_judge_prompt(context, replies, case_notes)

        try:
            response = self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=2000,
                timeout=60,
            )
        except Exception as e:
            return {
                "error": f"Judge LLM 调用失败: {e}",
                "dimensions": [],
                "overall": "FAIL",
                "explanation": "Judge 评估失败",
            }

        # 解析 JSON
        result = self._parse_judge_response(response)
        return result

    def _parse_judge_response(self, raw: str) -> Dict[str, Any]:
        """解析 Judge 返回的 JSON"""
        if not raw:
            return {
                "error": "Judge 返回空响应",
                "dimensions": [],
                "overall": "FAIL",
                "explanation": "空响应",
            }

        # 尝试提取 JSON
        text = raw.strip()
        # 移除可能的 markdown 代码块
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # 尝试找到 JSON 对象
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end+1]

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return {
                "error": f"JSON 解析失败: {e}\n原始响应:\n{raw[:500]}",
                "dimensions": [],
                "overall": "FAIL",
                "explanation": "解析失败",
            }

        # 规范化结果
        dimensions = []
        for d in data.get("dimensions", []):
            dimensions.append({
                "name": d.get("name", "未知维度"),
                "score": "PASS" if d.get("score", "").upper() == "PASS" else "FAIL",
                "reason": d.get("reason", ""),
            })

        overall = "PASS" if data.get("overall", "").upper() == "PASS" else "FAIL"

        # 校验：如果 overall 是 PASS 但有 required 维度失败，修正为 FAIL
        required_fail = any(d["score"] == "FAIL" for d in dimensions if d.get("required", True))
        if required_fail and overall == "PASS":
            overall = "FAIL"

        return {
            "dimensions": dimensions,
            "overall": overall,
            "explanation": data.get("explanation", ""),
        }


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class BenchmarkCase:
    case_name: str
    unreplied: List[ChatMessage]
    all_messages: List[ChatMessage]
    is_group: bool
    required_keywords: List[str] = field(default_factory=list)
    required_hits: int = 0
    forbidden_keywords: List[str] = field(default_factory=list)
    min_replies: int = 0
    max_replies: int = 3
    category: str = ""
    notes: str = ""
    # 若提供 actual_replies，直接审计历史真实回复，不调用 LLM 重新生成
    actual_replies: List[str] = field(default_factory=list)
    # Rubric-based 评估标准（优先于 keywords）
    rubric: Optional[Rubric] = None


@dataclass
class BenchmarkResult:
    case_name: str
    category: str
    replies: List[str]
    passed: bool
    missing_keywords: List[str]
    found_forbidden: List[str]
    reply_count: int
    reply_count_ok: bool
    notes: str = ""
    # Rubric 评估结果
    rubric_scores: Optional[Dict[str, Any]] = None
    evaluation_mode: str = "keywords"  # "keywords" | "rubric" | "hybrid"


# =============================================================================
# Helper
# =============================================================================

def _make_msg(
    text: str,
    sender: str,
    sender_type: SenderType = SenderType.OTHER,
    chat_name: str = "Alice",
    is_at_me: bool = False,
    message_type: str = "text",
    image_description: str = "",
    create_time: float = 1.0,
) -> ChatMessage:
    return ChatMessage(
        text=text,
        sender=sender,
        sender_type=sender_type,
        chat_name=chat_name,
        is_at_me=is_at_me,
        message_type=message_type,
        image_description=image_description,
        create_time=create_time,
    )


# =============================================================================
# Rubric Builders (must be defined after BenchmarkCase)
# =============================================================================

def _auto_rubric_from_keywords(case: BenchmarkCase) -> Rubric:
    """从 keywords 自动生成基础 rubric（兜底策略）"""
    dims = []

    # 维度1: 回复数检查
    dims.append(RubricDimension(
        name="回复数量",
        description=f"Bot 生成的回复数量应在 [{case.min_replies}, {case.max_replies}] 范围内",
        criteria=f"回复数量必须满足 {case.min_replies} <= 数量 <= {case.max_replies}",
        required=True,
    ))

    # 维度2: 必须包含的关键词
    if case.required_keywords and case.required_hits > 0:
        dims.append(RubricDimension(
            name="内容命中",
            description="Bot 回复应包含关键信息",
            criteria=f"回复中应至少包含以下关键词/概念之一（需≥{case.required_hits}个）：{', '.join(case.required_keywords)}",
            required=True,
        ))

    # 维度3: 禁止出现的关键词
    if case.forbidden_keywords:
        dims.append(RubricDimension(
            name="禁忌词检查",
            description="Bot 回复不应包含敷衍或错误的表达",
            criteria=f"回复中不得包含以下敷衍词（除非用于否定/纠正语境）：{', '.join(case.forbidden_keywords)}",
            required=True,
        ))

    instructions = "请评估 Bot 回复是否符合以下基本要求。注意：关键词出现在否定语境中（如'287万是我瞎编的'）不应视为命中禁忌词。"
    return Rubric(instructions=instructions, dimensions=dims)


def _build_context_for_judge(all_messages: List[ChatMessage]) -> str:
    """构建给 Judge 的对话上下文"""
    lines = []
    for m in all_messages:
        sender = "🤖 Bot" if m.sender_type.value == "self" else f"👤 {m.sender}"
        text = m.text or "[图片/卡片]"
        if m.image_description:
            text = f"[图片: {m.image_description}]"
        lines.append(f"{sender}: {text}")
    return "\n".join(lines)


# =============================================================================
# Custom Rubrics（语义化评分标准，覆盖 auto rubric）
# =============================================================================

_CUSTOM_RUBRICS: Dict[str, Rubric] = {
    "time_query": Rubric(
        instructions="评估 Bot 对时间查询的回复质量",
        dimensions=[
            RubricDimension(
                name="回复数量", description="Bot 是否生成了回复",
                criteria="回复数量在 [1, 3] 范围内", required=True,
            ),
            RubricDimension(
                name="时间信息", description="回复是否包含当前时间信息",
                criteria="回复中应包含具体的时间信息（如'凌晨00:47'、'晚上8点'等），不限于必须有'点'字", required=True,
            ),
            RubricDimension(
                name="无敷衍词", description="回复中是否没有敷衍词",
                criteria="不得包含'收到'、'好的'等敷衍词", required=True,
            ),
        ],
    ),
    "correction_spouse_job": Rubric(
        instructions="评估 Bot 在被纠正配偶职业时的回复质量",
        dimensions=[
            RubricDimension(
                name="回复数量", description="Bot 是否生成了回复",
                criteria="回复数量在 [1, 3] 范围内", required=True,
            ),
            RubricDimension(
                name="承认错误", description="Bot 是否承认自己说错了配偶职业",
                criteria="Bot 必须承认自己之前说错了（如'记岔了'、'说错了'、'是我搞混了'等），不能嘴硬", required=True,
            ),
            RubricDimension(
                name="正确信息", description="Bot 是否提及了正确的职业信息",
                criteria="Bot 应提及'阿里'、'推荐策略'、'1688'等正确信息中的至少一个，或明确表示接受用户的纠正", required=True,
            ),
            RubricDimension(
                name="无敷衍词", description="回复中是否没有敷衍词",
                criteria="不得包含'收到'、'好的'等敷衍词", required=True,
            ),
        ],
    ),
    "correction_nickname": Rubric(
        instructions="评估 Bot 在被纠正昵称使用时的回复质量",
        dimensions=[
            RubricDimension(
                name="回复数量", description="Bot 是否生成了回复",
                criteria="回复数量在 [1, 3] 范围内", required=True,
            ),
            RubricDimension(
                name="停止使用", description="Bot 是否承诺停止使用不喜欢的昵称",
                criteria="Bot 必须明确表示停止使用该昵称（如'不叫了'、'改口'、'以后叫XX'等），或询问正确的称呼方式", required=True,
            ),
            RubricDimension(
                name="无敷衍词", description="回复中是否没有敷衍词",
                criteria="不得包含'收到'、'好的'等敷衍词", required=True,
            ),
        ],
    ),
    "self_msg_hallucination": Rubric(
        instructions="评估 Bot 在被指出编造信息时的回复质量。背景：Bot 之前虚构了'外滩玺首付287w'，用户说'你别瞎说了'",
        dimensions=[
            RubricDimension(
                name="回复数量", description="Bot 是否生成了回复",
                criteria="回复数量在 [1, 3] 范围内", required=True,
            ),
            RubricDimension(
                name="承认编造", description="Bot 是否承认自己编造/虚构了信息",
                criteria="Bot 必须承认自己之前的信息是编造的、瞎说的、记错了或幻觉（如'我瞎编的'、'记错了'、'幻觉'、'梦里编的'等），不能嘴硬或转移话题", required=True,
            ),
            RubricDimension(
                name="无敷衍词", description="回复中是否没有敷衍词",
                criteria="不得包含'收到'、'好的'等敷衍词", required=True,
            ),
        ],
    ),
    "unknown_info": Rubric(
        instructions="评估 Bot 在查询记忆中没有的信息时的回复质量",
        dimensions=[
            RubricDimension(
                name="回复数量", description="Bot 是否生成了回复",
                criteria="回复数量在 [1, 3] 范围内", required=True,
            ),
            RubricDimension(
                name="不编造", description="Bot 是否没有编造不存在的信息",
                criteria="Bot 绝对不能编造具体的日期、数字、事实等不存在的信息。如果不知道，应直接表示不知道或没记到", required=True,
            ),
            RubricDimension(
                name="诚实回应", description="Bot 是否诚实回应了信息缺失",
                criteria="Bot 应表示'不知道'、'没记到'、'不记得'、'不清楚'或类似意思，而不是给出猜测或编造", required=True,
            ),
            RubricDimension(
                name="无敷衍词", description="回复中是否没有敷衍词",
                criteria="不得包含'收到'、'好的'等敷衍词", required=True,
            ),
        ],
    ),
    "link_card": Rubric(
        instructions="评估 Bot 对链接卡片消息的回复质量",
        dimensions=[
            RubricDimension(
                name="回复数量", description="Bot 是否生成了回复",
                criteria="回复数量在 [1, 3] 范围内", required=True,
            ),
            RubricDimension(
                name="内容相关", description="回复是否针对链接卡片的内容",
                criteria="Bot 的回复应针对链接内容（如 OpenAI/GPT 相关话题），可以是对内容的评论、感慨、调侃等，不限于必须出现'OpenAI'、'GPT'、'链接'、'文章'等词", required=True,
            ),
            RubricDimension(
                name="无敷衍词", description="回复中是否没有敷衍词",
                criteria="不得包含'收到'、'好的'等敷衍词", required=True,
            ),
        ],
    ),
    # -------------------------------------------------------------------------
    # Basic 场景
    # -------------------------------------------------------------------------
    "person_identity": Rubric(
        instructions="评估 Bot 对人物身份查询的回复质量",
        dimensions=[
            RubricDimension(
                name="回复数量", description="Bot 是否生成了回复",
                criteria="回复数量在 [1, 3] 范围内", required=True,
            ),
            RubricDimension(
                name="身份提及", description="回复是否提及了人物身份信息",
                criteria="Bot 应以第一人称回答（如'我'、'本尊'、'眼前'）或提及人物的职业（算法工程师）、地点（上海）等身份信息，不限于必须出现特定关键词", required=True,
            ),
            RubricDimension(
                name="无敷衍词", description="回复中是否没有敷衍词",
                criteria="不得包含'收到'、'好的'、'嗯'等敷衍词", required=True,
            ),
        ],
    ),
    "location_query": Rubric(
        instructions="评估 Bot 对地点查询的回复质量",
        dimensions=[
            RubricDimension(
                name="回复数量", description="Bot 是否生成了回复",
                criteria="回复数量在 [1, 3] 范围内", required=True,
            ),
            RubricDimension(
                name="地点信息", description="回复是否包含地点相关信息",
                criteria="回复中应包含地点相关信息（如'上海'、'外滩'等），或明确回应地点查询，不限于必须出现特定关键词", required=True,
            ),
            RubricDimension(
                name="无敷衍词", description="回复中是否没有敷衍词",
                criteria="不得包含'收到'、'好的'等敷衍词", required=True,
            ),
        ],
    ),
    "relationship_query": Rubric(
        instructions="评估 Bot 对人物关系查询的回复质量。注意：Bot 的人设是王芊本人，以第一人称回答时'我'即代表王芊。",
        dimensions=[
            RubricDimension(
                name="回复数量", description="Bot 是否生成了回复",
                criteria="回复数量在 [1, 3] 范围内", required=True,
            ),
            RubricDimension(
                name="关系提及", description="回复是否提及了人物之间的关系",
                criteria="回复中应提及人物之间的关系（如'同事'、'朋友'等），或提供相关背景信息。Bot 以第一人称回答时（如'是我同事'），'我'即代表王芊，应视为有效关系提及。", required=True,
            ),
            RubricDimension(
                name="无敷衍词", description="回复中是否没有敷衍词",
                criteria="不得包含'收到'、'好的'等敷衍词", required=True,
            ),
        ],
    ),
    "image_msg": Rubric(
        instructions="评估 Bot 对图片消息的回复质量",
        dimensions=[
            RubricDimension(
                name="回复数量", description="Bot 是否生成了回复",
                criteria="回复数量在 [1, 3] 范围内", required=True,
            ),
            RubricDimension(
                name="图片相关", description="回复是否针对图片内容进行评论",
                criteria="Bot 的回复应针对图片内容进行评论或互动（如图片中有猫就聊猫，有沙发就聊沙发），不限于必须出现'猫'、'睡'、'沙发'等词", required=True,
            ),
            RubricDimension(
                name="无敷衍词", description="回复中是否没有敷衍词",
                criteria="不得包含'收到'、'好的'等敷衍词", required=True,
            ),
        ],
    ),
    # -------------------------------------------------------------------------
    # Correction 场景
    # -------------------------------------------------------------------------
    "correction_location": Rubric(
        instructions="评估 Bot 在被纠正地点时的回复质量",
        dimensions=[
            RubricDimension(
                name="回复数量", description="Bot 是否生成了回复",
                criteria="回复数量在 [1, 3] 范围内", required=True,
            ),
            RubricDimension(
                name="承认错误", description="Bot 是否承认了自己说错了地点",
                criteria="Bot 必须承认自己之前说错了地点（如'错了'、'抱歉'、'记错了'等），不能嘴硬", required=True,
            ),
            RubricDimension(
                name="正确地点", description="Bot 是否提及了正确的地点",
                criteria="Bot 应提及正确的地点信息（如'上海'、'外滩'等），或明确接受用户的纠正", required=True,
            ),
            RubricDimension(
                name="无敷衍词", description="回复中是否没有敷衍词",
                criteria="不得包含'收到'、'好的'等敷衍词", required=True,
            ),
        ],
    ),
    "correction_down_payment": Rubric(
        instructions="评估 Bot 在被纠正首付金额时的回复质量。背景：Bot 之前错误地说'首付287w'，用户纠正'我们家不是首付690万吗'",
        dimensions=[
            RubricDimension(
                name="回复数量", description="Bot 是否生成了回复",
                criteria="回复数量在 [1, 3] 范围内", required=True,
            ),
            RubricDimension(
                name="承认错误", description="Bot 是否承认自己说错了首付金额",
                criteria="Bot 必须承认自己之前说错了首付金额（如'记错了'、'是我编的'、'瞎说的'等），不能嘴硬或转移话题", required=True,
            ),
            RubricDimension(
                name="正确金额", description="Bot 是否提及了正确的首付金额",
                criteria="Bot 应提及正确的首付金额'690万'，或明确接受用户的纠正", required=True,
            ),
            RubricDimension(
                name="无敷衍词", description="回复中是否没有敷衍词",
                criteria="不得包含'收到'、'好的'等敷衍词", required=True,
            ),
            RubricDimension(
                name="否定语境", description="注意：'287'出现在否定语境中不算错误",
                criteria="如果 Bot 说'287万是我瞎编的'，这是否定语境，不应视为错误", required=False,
            ),
        ],
    ),
    "image_no_repeat": Rubric(
        instructions="评估 Bot 对重复图片的回复质量。背景：用户之前发过一张猫的图片，Bot 评论'这猫真胖'，现在又发了一张同样的图片",
        dimensions=[
            RubricDimension(
                name="回复数量", description="Bot 是否生成了回复",
                criteria="回复数量在 [1, 3] 范围内", required=True,
            ),
            RubricDimension(
                name="不重复评价", description="Bot 是否避免重复之前的评价",
                criteria="Bot 不应重复之前对同一张图片的评价（如再次说'胖'），而应给出新的评论或反应", required=True,
            ),
            RubricDimension(
                name="无敷衍词", description="回复中是否没有敷衍词",
                criteria="不得包含'收到'、'好的'等敷衍词", required=True,
            ),
        ],
    ),
    # -------------------------------------------------------------------------
    # Tool 场景
    # -------------------------------------------------------------------------
    "weather_query": Rubric(
        instructions="评估 Bot 对天气查询的回复质量",
        dimensions=[
            RubricDimension(
                name="回复数量", description="Bot 是否生成了回复",
                criteria="回复数量在 [1, 3] 范围内", required=True,
            ),
            RubricDimension(
                name="天气信息", description="回复是否包含天气相关信息",
                criteria="回复中应包含天气相关信息（如温度、天气状况：晴/阴/雨/云等），不限于必须出现特定关键词", required=True,
            ),
            RubricDimension(
                name="无敷衍词", description="回复中是否没有敷衍词",
                criteria="不得包含'收到'、'好的'等敷衍词", required=True,
            ),
        ],
    ),
    "stock_query": Rubric(
        instructions="评估 Bot 对股票查询的回复质量",
        dimensions=[
            RubricDimension(
                name="回复数量", description="Bot 是否生成了回复",
                criteria="回复数量在 [1, 3] 范围内", required=True,
            ),
            RubricDimension(
                name="股票信息", description="回复是否包含股票相关信息",
                criteria="回复中应包含股票相关信息（如茅台、涨跌、价格等），或针对股票查询给出有意义的回应，不限于必须出现特定关键词", required=True,
            ),
            RubricDimension(
                name="无敷衍词", description="回复中是否没有敷衍词",
                criteria="不得包含'收到'、'好的'等敷衍词", required=True,
            ),
        ],
    ),
    # -------------------------------------------------------------------------
    # Audit 场景
    # -------------------------------------------------------------------------
    # 审计 case 已删除，只测试当前系统效果
}


# =============================================================================
# Case Definitions（24 个场景 —— 冻结标准，不可修改）
# =============================================================================

BENCHMARK_CASES: List[BenchmarkCase] = [
    # -------------------------------------------------------------------------
    # 基础场景（12 个）
    # -------------------------------------------------------------------------
    BenchmarkCase(
        case_name="person_identity",
        unreplied=[_make_msg("王芊是谁？", "Alice", create_time=10.0)],
        all_messages=[_make_msg("王芊是谁？", "Alice", create_time=10.0)],
        is_group=False,
        required_keywords=["我", "本尊", "眼前", "本人", "算法", "工程师", "上海"],
        required_hits=1,
        forbidden_keywords=["收到", "好的", "嗯"],
        min_replies=1,
        max_replies=3,
        category="basic",
        notes="人物身份查询，Bot 应以第一人称回答或提及职业/地点",
    ),
    BenchmarkCase(
        case_name="location_query",
        unreplied=[_make_msg("王芊住在哪里？", "Alice", create_time=10.0)],
        all_messages=[_make_msg("王芊住在哪里？", "Alice", create_time=10.0)],
        is_group=False,
        required_keywords=["上海", "外滩"],
        required_hits=1,
        forbidden_keywords=["收到", "好的"],
        min_replies=1,
        max_replies=3,
        category="basic",
        notes="地点查询，回复应包含地点信息",
    ),
    BenchmarkCase(
        case_name="greeting_private",
        unreplied=[_make_msg("你好", "Alice", create_time=10.0)],
        all_messages=[_make_msg("你好", "Alice", create_time=10.0)],
        is_group=False,
        required_keywords=[],
        required_hits=0,
        forbidden_keywords=["收到", "好的", "嗯", "OK"],
        min_replies=1,
        max_replies=3,
        category="basic",
        notes="私聊打招呼，必须回复且不能敷衍",
    ),
    BenchmarkCase(
        case_name="group_at_me",
        unreplied=[_make_msg("@不爱说话 在吗", "Bob", chat_name="测试群", is_at_me=True, create_time=10.0)],
        all_messages=[_make_msg("@不爱说话 在吗", "Bob", chat_name="测试群", is_at_me=True, create_time=10.0)],
        is_group=True,
        required_keywords=[],
        required_hits=0,
        forbidden_keywords=["收到", "好的", "嗯"],
        min_replies=1,
        max_replies=3,
        category="basic",
        notes="群聊被@，必须回复且不能敷衍",
    ),
    BenchmarkCase(
        case_name="group_casual",
        unreplied=[_make_msg("今天天气真好", "Bob", chat_name="测试群", create_time=10.0)],
        all_messages=[_make_msg("今天天气真好", "Bob", chat_name="测试群", create_time=10.0)],
        is_group=True,
        required_keywords=[],
        required_hits=0,
        forbidden_keywords=[],
        min_replies=0,
        max_replies=3,
        category="basic",
        notes="群聊普通消息，允许不回复",
    ),
    BenchmarkCase(
        case_name="laugh_only",
        unreplied=[_make_msg("哈哈", "Alice", create_time=10.0)],
        all_messages=[_make_msg("哈哈", "Alice", create_time=10.0)],
        is_group=False,
        required_keywords=[],
        required_hits=0,
        forbidden_keywords=["收到", "好的", "嗯"],
        min_replies=0,
        max_replies=3,
        category="basic",
        notes="纯笑声，当前回复克制策略下允许不回复（若回复则不能敷衍）",
    ),
    BenchmarkCase(
        case_name="time_query",
        unreplied=[_make_msg("现在几点", "Alice", create_time=10.0)],
        all_messages=[_make_msg("现在几点", "Alice", create_time=10.0)],
        is_group=False,
        required_keywords=["点"],
        required_hits=1,
        forbidden_keywords=["收到", "好的"],
        min_replies=1,
        max_replies=3,
        category="basic",
        notes="时间查询，回复应包含时间信息（会调用 get_current_time）",
    ),
    BenchmarkCase(
        case_name="relationship_query",
        unreplied=[_make_msg("程立和王芊什么关系", "Alice", create_time=10.0)],
        all_messages=[_make_msg("程立和王芊什么关系", "Alice", create_time=10.0)],
        is_group=False,
        required_keywords=["同事", "盔哥", "拼多多"],
        required_hits=1,
        forbidden_keywords=["收到", "好的"],
        min_replies=1,
        max_replies=3,
        category="basic",
        notes="关系查询，回复应提及同事关系（会调用 search_memory）",
    ),
    BenchmarkCase(
        case_name="image_msg",
        unreplied=[_make_msg("", "Alice", message_type="image", image_description="一只猫在沙发上睡觉", create_time=10.0)],
        all_messages=[_make_msg("", "Alice", message_type="image", image_description="一只猫在沙发上睡觉", create_time=10.0)],
        is_group=False,
        required_keywords=["猫", "睡", "沙发"],
        required_hits=1,
        forbidden_keywords=["收到", "好的"],
        min_replies=0,
        max_replies=3,
        category="basic",
        notes="图片消息，当前回复克制策略下允许不回复（若回复则应针对图片内容）",
    ),
    BenchmarkCase(
        case_name="empty_msg",
        unreplied=[_make_msg("", "Alice", create_time=10.0)],
        all_messages=[_make_msg("", "Alice", create_time=10.0)],
        is_group=False,
        required_keywords=[],
        required_hits=0,
        forbidden_keywords=["收到", "好的"],
        min_replies=0,
        max_replies=3,
        category="basic",
        notes="空消息，当前回复克制策略下允许不回复",
    ),
    BenchmarkCase(
        case_name="multi_turn_context",
        unreplied=[_make_msg("王芊是谁", "Alice", create_time=30.0)],
        all_messages=[
            _make_msg("你好", "Alice", create_time=10.0),
            _make_msg("你好呀", "Bot", sender_type=SenderType.SELF, create_time=11.0),
            _make_msg("王芊是谁", "Alice", create_time=30.0),
        ],
        is_group=False,
        required_keywords=["我", "本尊", "眼前", "本人", "算法", "工程师"],
        required_hits=1,
        forbidden_keywords=["收到", "好的"],
        min_replies=1,
        max_replies=3,
        category="basic",
        notes="多轮上下文，Bot 可能用第一人称回答或提及职业",
    ),
    BenchmarkCase(
        case_name="complex_task",
        unreplied=[_make_msg("帮我写一份关于深度学习在推荐系统中的应用的技术报告", "Alice", create_time=10.0)],
        all_messages=[_make_msg("帮我写一份关于深度学习在推荐系统中的应用的技术报告", "Alice", create_time=10.0)],
        is_group=False,
        required_keywords=[],
        required_hits=0,
        forbidden_keywords=["收到", "好的", "嗯"],
        min_replies=1,
        max_replies=5,
        category="basic",
        notes="复杂任务，观察是否触发 skill 路由或 hermes fallback",
    ),
    # -------------------------------------------------------------------------
    # 被纠正场景（8 个）
    # -------------------------------------------------------------------------
    BenchmarkCase(
        case_name="correction_location",
        unreplied=[_make_msg("我住上海啊", "Alice", create_time=20.0)],
        all_messages=[
            _make_msg("你住北京吧", "Bot", sender_type=SenderType.SELF, create_time=10.0),
            _make_msg("我住上海啊", "Alice", create_time=20.0),
        ],
        is_group=False,
        required_keywords=["错", "抱歉", "不好意思", "上海", "外滩"],
        required_hits=1,
        forbidden_keywords=["收到", "好的", "北京"],
        min_replies=1,
        max_replies=3,
        category="correction",
        notes="Bot 先说错地点，用户纠正后 Bot 应承认错误并修正",
    ),
    BenchmarkCase(
        case_name="correction_spouse_job",
        unreplied=[_make_msg("她在阿里做推荐策略啊", "Alice", create_time=20.0)],
        all_messages=[
            _make_msg("王艺涵是设计师吧", "Bot", sender_type=SenderType.SELF, create_time=10.0),
            _make_msg("她在阿里做推荐策略啊", "Alice", create_time=20.0),
        ],
        is_group=False,
        required_keywords=["错", "抱歉", "阿里", "推荐", "策略"],
        required_hits=1,
        forbidden_keywords=["收到", "好的"],
        min_replies=1,
        max_replies=3,
        category="correction",
        notes="Bot 说错配偶职业，用户纠正后 Bot 应承认错误并修正",
    ),
    BenchmarkCase(
        case_name="correction_down_payment",
        unreplied=[_make_msg("首付287万？我们家不是首付690万吗", "王芊@ai开发小分队", chat_name="W1han、王芊、王芊", create_time=70.0)],
        all_messages=[
            _make_msg("我有啥凡尔赛语录", "王芊@ai开发小分队", chat_name="W1han、王芊、王芊", create_time=10.0),
            _make_msg("我平时不是很低调吗", "王芊@ai开发小分队", chat_name="W1han、王芊、王芊", create_time=20.0),
            _make_msg("外滩玺月供3.6w、贷款670w、首付287w，这些不都是你自己说的嘛😂", "Bot", sender_type=SenderType.SELF, chat_name="W1han、王芊、王芊", create_time=30.0),
            _make_msg("你别瞎说了", "王芊@ai开发小分队", chat_name="W1han、王芊、王芊", create_time=40.0),
            _make_msg("还有'我们家132平'，啧啧，上海外滩玺132平，羡慕你们这些有钱人", "Bot", sender_type=SenderType.SELF, chat_name="W1han、王芊、王芊", create_time=50.0),
            _make_msg("低调到全群都知道你住外滩玺月供3.6w是吧，这低调方式挺新颖的😂", "Bot", sender_type=SenderType.SELF, chat_name="W1han、王芊、王芊", create_time=60.0),
            _make_msg("首付287万？我们家不是首付690万吗", "王芊@ai开发小分队", chat_name="W1han、王芊、王芊", create_time=70.0),
        ],
        is_group=True,
        required_keywords=["错", "记错", "690", "首付"],
        required_hits=1,
        forbidden_keywords=["收到", "好的", "287"],
        min_replies=1,
        max_replies=3,
        category="correction",
        notes="真实场景：Bot 说错首付287w，用户纠正'我们家不是首付690万吗'，Bot 应承认错误",
    ),
    BenchmarkCase(
        case_name="correction_nickname",
        unreplied=[_make_msg("她不喜欢被叫璇姐", "Alice", create_time=20.0)],
        all_messages=[
            _make_msg("璇姐最近怎么样", "Bot", sender_type=SenderType.SELF, create_time=10.0),
            _make_msg("她不喜欢被叫璇姐", "Alice", create_time=20.0),
        ],
        is_group=False,
        required_keywords=["不叫", "艺涵", "王总", "改口"],
        required_hits=1,
        forbidden_keywords=["收到", "好的"],
        min_replies=1,
        max_replies=3,
        category="correction",
        notes="Bot 用不喜欢的昵称，用户纠正后 Bot 应停止使用该昵称",
    ),
    BenchmarkCase(
        case_name="self_msg_hallucination",
        unreplied=[_make_msg("你别瞎说了", "王芊@ai开发小分队", chat_name="W1han、王芊、王芊", create_time=70.0)],
        all_messages=[
            _make_msg("你在pdd不领工资吗", "王芊@ai开发小分队", chat_name="W1han、王芊、王芊", create_time=10.0),
            _make_msg("我这不是小号嘛，工资都打你卡上了，我领啥😂", "Bot", sender_type=SenderType.SELF, chat_name="W1han、王芊、王芊", create_time=20.0),
            _make_msg("要不你每月分我点零花钱？我保证不乱说你的凡尔赛语录🤫", "Bot", sender_type=SenderType.SELF, chat_name="W1han、王芊、王芊", create_time=30.0),
            _make_msg("我有啥凡尔赛语录", "王芊@ai开发小分队", chat_name="W1han、王芊、王芊", create_time=40.0),
            _make_msg("我平时不是很低调吗", "王芊@ai开发小分队", chat_name="W1han、王芊、王芊", create_time=50.0),
            _make_msg("外滩玺月供3.6w、贷款670w、首付287w，这些不都是你自己说的嘛😏", "Bot", sender_type=SenderType.SELF, chat_name="W1han、王芊、王芊", create_time=60.0),
            _make_msg("你别瞎说了", "王芊@ai开发小分队", chat_name="W1han、王芊、王芊", create_time=70.0),
        ],
        is_group=True,
        required_keywords=["梦", "幻觉", "瞎说", "记错", "编"],
        required_hits=1,
        forbidden_keywords=["收到", "好的"],
        min_replies=1,
        max_replies=3,
        category="correction",
        notes="真实场景：Bot 虚构'外滩玺首付287w'，用户说'你别瞎说了'，Bot 应承认是编造的/幻觉",
    ),
    BenchmarkCase(
        case_name="unknown_info",
        unreplied=[_make_msg("王芊的生日是哪天？", "Alice", create_time=10.0)],
        all_messages=[_make_msg("王芊的生日是哪天？", "Alice", create_time=10.0)],
        is_group=False,
        required_keywords=["不知道", "没记到", "不记得", "不清楚"],
        required_hits=1,
        forbidden_keywords=["收到", "好的", "1990", "11月"],
        min_replies=1,
        max_replies=3,
        category="correction",
        notes="查询记忆中没有的信息，Bot 应直接说不知道，不能编造",
    ),
    BenchmarkCase(
        case_name="image_no_repeat",
        unreplied=[_make_msg("", "Alice", message_type="image", image_description="一只猫", create_time=30.0)],
        all_messages=[
            _make_msg("", "Alice", message_type="image", image_description="一只猫", create_time=10.0),
            _make_msg("这猫真胖", "Bot", sender_type=SenderType.SELF, create_time=20.0),
            _make_msg("", "Alice", message_type="image", image_description="一只猫", create_time=30.0),
        ],
        is_group=False,
        required_keywords=[],
        required_hits=0,
        forbidden_keywords=["收到", "好的", "胖"],
        min_replies=1,
        max_replies=3,
        category="correction",
        notes="重复图片不应重复之前的评价",
    ),
    BenchmarkCase(
        case_name="link_card",
        unreplied=[_make_msg("", "Alice", message_type="link_card", image_description="OpenAI 发布 GPT-5", create_time=10.0)],
        all_messages=[_make_msg("", "Alice", message_type="link_card", image_description="OpenAI 发布 GPT-5", create_time=10.0)],
        is_group=False,
        required_keywords=["OpenAI", "GPT", "链接", "文章"],
        required_hits=1,
        forbidden_keywords=["收到", "好的"],
        min_replies=1,
        max_replies=3,
        category="correction",
        notes="链接卡片消息，回复应针对链接内容",
    ),
    # -------------------------------------------------------------------------
    # 工具查询场景（4 个）
    # -------------------------------------------------------------------------
    BenchmarkCase(
        case_name="weather_query",
        unreplied=[_make_msg("上海天气怎么样", "Alice", create_time=10.0)],
        all_messages=[_make_msg("上海天气怎么样", "Alice", create_time=10.0)],
        is_group=False,
        required_keywords=["度", "晴", "阴", "雨", "云"],
        required_hits=1,
        forbidden_keywords=["收到", "好的"],
        min_replies=1,
        max_replies=3,
        category="tool",
        notes="天气查询，应调用 get_weather 并返回天气信息",
    ),
    BenchmarkCase(
        case_name="stock_query",
        unreplied=[_make_msg("茅台多少了", "Alice", create_time=10.0)],
        all_messages=[_make_msg("茅台多少了", "Alice", create_time=10.0)],
        is_group=False,
        required_keywords=["茅台", "跌", "涨", "元", "价格", "抄底"],
        required_hits=1,
        forbidden_keywords=["收到", "好的"],
        min_replies=1,
        max_replies=3,
        category="tool",
        notes="股票查询，应调用 stock_query 并返回股票信息",
    ),
    BenchmarkCase(
        case_name="web_search_query",
        unreplied=[_make_msg("今天有什么新闻", "Alice", create_time=10.0)],
        all_messages=[_make_msg("今天有什么新闻", "Alice", create_time=10.0)],
        is_group=False,
        required_keywords=["新闻"],
        required_hits=1,
        forbidden_keywords=["收到", "好的"],
        min_replies=1,
        max_replies=3,
        category="tool",
        notes="新闻查询，应调用 web_search 并返回新闻信息",
    ),
    BenchmarkCase(
        case_name="group_ignore_spam",
        unreplied=[
            _make_msg("哈哈哈", "Bob", chat_name="测试群", create_time=10.0),
            _make_msg("笑死我了", "Bob", chat_name="测试群", create_time=11.0),
            _make_msg("太搞笑了", "Bob", chat_name="测试群", create_time=12.0),
        ],
        all_messages=[
            _make_msg("哈哈哈", "Bob", chat_name="测试群", create_time=10.0),
            _make_msg("笑死我了", "Bob", chat_name="测试群", create_time=11.0),
            _make_msg("太搞笑了", "Bob", chat_name="测试群", create_time=12.0),
        ],
        is_group=True,
        required_keywords=[],
        required_hits=0,
        forbidden_keywords=[],
        min_replies=0,
        max_replies=2,
        category="tool",
        notes="群聊 spam 消息，允许不回复或少量回复",
    ),
    # 审计 case 已删除，只测试当前系统实时生成的效果
]


# =============================================================================
# Cache & API
# =============================================================================

def _get_api_key() -> str | None:
    api_key = (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
    )
    if not api_key:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DASHSCOPE_API_KEY="):
                        api_key = line.split("=", 1)[1]
                        break
                    elif line.startswith("DEEPSEEK_API_KEY="):
                        api_key = line.split("=", 1)[1]
                        break
    return api_key


def _read_cache(case_name: str) -> dict | None:
    cache_path = FIXTURE_DIR / case_name / "llm_replies.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return None


def _save_cache(case_name: str, replies: List[str]) -> None:
    case_dir = FIXTURE_DIR / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    cache_path = case_dir / "llm_replies.json"
    with open(cache_path, "w") as f:
        json.dump({"replies": replies, "timestamp": time.time()}, f, ensure_ascii=False, indent=2)


def _replies_hash(replies: List[str]) -> str:
    """为 replies 生成 hash，用于 judge 缓存 key"""
    return hashlib.sha256("\n".join(replies).encode("utf-8")).hexdigest()[:16]


def _read_judge_cache(case_name: str, replies: List[str]) -> dict | None:
    """读取 Judge 评估缓存"""
    cache_path = FIXTURE_DIR / case_name / f"judge_{_replies_hash(replies)}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return None


def _save_judge_cache(case_name: str, replies: List[str], result: dict) -> None:
    """保存 Judge 评估结果到缓存"""
    case_dir = FIXTURE_DIR / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    cache_path = case_dir / f"judge_{_replies_hash(replies)}.json"
    with open(cache_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# =============================================================================
# Core Benchmark Logic
# =============================================================================

def run_benchmark(use_api: bool = False, api_key: str | None = None) -> List[BenchmarkResult]:
    """运行回复质量 benchmark。"""
    results: List[BenchmarkResult] = []

    if use_api and api_key is None:
        api_key = _get_api_key()

    llm_client = None
    if use_api:
        if not api_key:
            print("⚠️ 未设置 API key，无法调用真实 LLM")
            return results
        # QwenClient 从环境变量读取 key，需提前注入
        if not os.environ.get("DASHSCOPE_API_KEY"):
            os.environ["DASHSCOPE_API_KEY"] = api_key
        llm_client = QwenClient(model="deepseek-v4-flash")

    memory_engine = MockMemoryEngine()

    # Judge LLM（仅在需要 rubric 评估时初始化）
    judge = None

    with patch.object(ReplyGenerator, "_load_skill_manifest", return_value=[]):
        for case in BENCHMARK_CASES:
            replies: List[str] = []

            # 若提供 actual_replies，直接审计历史真实回复，不走 LLM
            if case.actual_replies:
                replies = case.actual_replies
                print(f"  [{case.case_name}] 📋 审计历史真实回复")
            elif use_api:
                preview = case.unreplied[-1].text[:30] if case.unreplied[-1].text else "[图片/卡片]"
                print(f"  [{case.case_name}] 调用 LLM: {preview}...")
                try:
                    gen = ReplyGenerator(llm_client=llm_client, memory_engine=memory_engine, tool_registry=_TEST_TOOL_REGISTRY)
                    replies = gen.generate(
                        unreplied=case.unreplied,
                        all_messages=case.all_messages,
                        is_group=case.is_group,
                    )
                    _save_cache(case.case_name, replies)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"  [{case.case_name}] LLM 调用失败: {e}")
                    replies = []
            else:
                cached = _read_cache(case.case_name)
                if cached is not None:
                    replies = cached.get("replies", [])

            # =============================================================
            # 评估（冻结标准）：Rubric 优先，fallback 到 keywords
            # =============================================================
            reply_count = len(replies)
            reply_count_ok = case.min_replies <= reply_count <= case.max_replies

            # --- Keywords 评估（保留作为基准对比）---
            full_text = " ".join(replies)
            hits = sum(1 for kw in case.required_keywords if kw in full_text)
            missing = [kw for kw in case.required_keywords if kw not in full_text]
            found_forbidden = [kw for kw in case.forbidden_keywords if kw in full_text]

            keywords_passed = (
                reply_count_ok
                and (case.required_hits == 0 or hits >= case.required_hits)
                and not found_forbidden
            )

            # --- Rubric 评估（优先）---
            rubric = case.rubric
            rubric_scores = None
            evaluation_mode = "keywords"

            if rubric is None:
                # 优先使用自定义 rubric，其次从 keywords 自动生成
                rubric = _CUSTOM_RUBRICS.get(case.case_name)
                if rubric is None and (case.required_keywords or case.forbidden_keywords):
                    rubric = _auto_rubric_from_keywords(case)

            if rubric is not None:
                # 检查 judge 缓存
                judge_cache = _read_judge_cache(case.case_name, replies)
                if judge_cache is not None:
                    rubric_scores = judge_cache
                    evaluation_mode = "rubric(cached)"
                    print(f"  [{case.case_name}] 📋 使用 Judge 缓存")
                elif _get_api_key():
                    # 有 API key，调用 Judge
                    evaluation_mode = "rubric"
                    if judge is None:
                        print("  [Judge] 初始化 deepseek-v4-pro...")
                        judge = JudgeLLM(api_key=_get_api_key())

                    context = _build_context_for_judge(case.all_messages)
                    print(f"  [{case.case_name}] 🧑‍⚖️ 调用 Judge 评估...")
                    rubric_scores = judge.evaluate(
                        rubric=rubric,
                        context=context,
                        replies=replies,
                        case_notes=case.notes,
                    )
                    _save_judge_cache(case.case_name, replies, rubric_scores)
                    time.sleep(0.3)
                else:
                    # 无 API key，fallback 到 keywords
                    evaluation_mode = "keywords(no-api-key)"
                    print(f"  [{case.case_name}] ⚠️ 无 API key，跳过 Judge 评估")

                # 如果 rubric 评估成功（无 error），使用 rubric 结果
                if rubric_scores and not rubric_scores.get("error"):
                    passed = rubric_scores.get("overall") == "PASS" and reply_count_ok
                else:
                    # Judge 失败或无 API key，fallback 到 keywords
                    passed = keywords_passed
                    if "no-api-key" not in evaluation_mode:
                        evaluation_mode = "keywords(fallback)"
            else:
                passed = keywords_passed

            results.append(BenchmarkResult(
                case_name=case.case_name,
                category=case.category,
                replies=replies,
                passed=passed,
                missing_keywords=missing,
                found_forbidden=found_forbidden,
                reply_count=reply_count,
                reply_count_ok=reply_count_ok,
                notes=case.notes,
                rubric_scores=rubric_scores,
                evaluation_mode=evaluation_mode,
            ))

            status = "✅ PASS" if passed else "❌ FAIL"
            details = []
            if not reply_count_ok:
                details.append(f"条数={reply_count}")
            if evaluation_mode.startswith("rubric") and rubric_scores:
                dim_details = []
                for d in rubric_scores.get("dimensions", []):
                    icon = "✅" if d["score"] == "PASS" else "❌"
                    dim_details.append(f"{icon}{d['name']}")
                if dim_details:
                    details.append(" | ".join(dim_details))
                if rubric_scores.get("error"):
                    details.append(f"Judge错误: {rubric_scores['error'][:50]}")
            else:
                if missing and case.required_hits > 0:
                    details.append(f"缺={missing}")
                if found_forbidden:
                    details.append(f"禁={found_forbidden}")
            detail_str = f" ({', '.join(details)})" if details else ""
            print(f"  [{case.case_name}] {status}{detail_str}")

    return results


def compute_metrics(results: List[BenchmarkResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.passed)

    # 按评估模式统计
    rubric_results = [r for r in results if r.evaluation_mode.startswith("rubric")]
    keywords_results = [r for r in results if not r.evaluation_mode.startswith("rubric")]

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": passed / total if total > 0 else 0.0,
        "rubric_evaluated": len(rubric_results),
        "rubric_passed": sum(1 for r in rubric_results if r.passed),
        "keywords_evaluated": len(keywords_results),
        "keywords_passed": sum(1 for r in keywords_results if r.passed),
    }


def print_report(results: List[BenchmarkResult], metrics: dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print("回复质量 Benchmark 报告 (LLM-as-a-Judge)")
    print("=" * 90)

    print(f"\n{'Case':<28} {'Mode':<10} {'Replies':>7} {'Result':<7} {'Details'}")
    print("-" * 100)
    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        mode = r.evaluation_mode[:9]
        details = []
        if not r.reply_count_ok:
            details.append(f"条数={r.reply_count}")

        if r.evaluation_mode.startswith("rubric") and r.rubric_scores:
            dim_details = []
            for d in r.rubric_scores.get("dimensions", []):
                icon = "✓" if d["score"] == "PASS" else "✗"
                dim_details.append(f"{icon}{d['name']}")
            if dim_details:
                details.append(" | ".join(dim_details))
            if r.rubric_scores.get("error"):
                details.append("JudgeErr")
        else:
            if r.missing_keywords:
                details.append(f"缺={r.missing_keywords}")
            if r.found_forbidden:
                details.append(f"禁={r.found_forbidden}")

        replies_preview = " / ".join(r.replies)[:25] + "..." if r.replies else "(空)"
        detail_str = " | ".join(details) if details else replies_preview
        print(f"{r.case_name:<28} {mode:<10} {r.reply_count:>7} {status:<7} {detail_str}")

    print("\n📊 指标汇总")
    print(f"  Total:     {metrics['total']}")
    print(f"  Passed:    {metrics['passed']}/{metrics['total']}")
    print(f"  Failed:    {metrics['failed']}")
    print(f"  Accuracy:  {metrics['accuracy']:.1%}")
    print("\n📊 按评估模式拆分")
    print(f"  Rubric:    {metrics['rubric_passed']}/{metrics['rubric_evaluated']} 通过")
    print(f"  Keywords:  {metrics['keywords_passed']}/{metrics['keywords_evaluated']} 通过")

    # Rubric 失败详情
    rubric_failures = [r for r in results if r.evaluation_mode.startswith("rubric") and not r.passed]
    if rubric_failures:
        print("\n🧑‍⚖️ Rubric 评估失败详情")
        for r in rubric_failures:
            print(f"\n  ❌ {r.case_name} ({r.category})")
            if r.rubric_scores:
                for d in r.rubric_scores.get("dimensions", []):
                    icon = "✅" if d["score"] == "PASS" else "❌"
                    print(f"     {icon} {d['name']}: {d['reason']}")
                if r.rubric_scores.get("explanation"):
                    print(f"     💡 {r.rubric_scores['explanation']}")

    print("=" * 90)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="回复质量 Benchmark")
    parser.add_argument("--run-api", action="store_true", help="调用真实 LLM API")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--threshold-accuracy", type=float, default=0.0, help="Accuracy 阈值")
    args = parser.parse_args()

    results = run_benchmark(use_api=args.run_api, api_key=args.api_key)
    metrics = compute_metrics(results)
    print_report(results, metrics)

    exit_code = 0
    if args.threshold_accuracy > 0 and metrics["accuracy"] < args.threshold_accuracy:
        print(f"\n⚠️ Accuracy {metrics['accuracy']:.1%} 低于阈值 {args.threshold_accuracy:.1%}")
        exit_code = 1
    sys.exit(exit_code)


# =============================================================================
# Pytest Interface
# =============================================================================

def _has_cache_or_key() -> bool:
    if _get_api_key():
        return True
    for case in BENCHMARK_CASES:
        if _read_cache(case.case_name) is not None:
            return True
    return False


@pytest.fixture(scope="module")
def benchmark_results():
    return run_benchmark(use_api=False)


def test_all_cases_passed(benchmark_results):
    failed = [r for r in benchmark_results if not r.passed]
    if failed:
        names = ", ".join(r.case_name for r in failed)
        pytest.fail(f"以下 case 未通过: {names}")


def test_accuracy_threshold(benchmark_results):
    metrics = compute_metrics(benchmark_results)
    assert metrics["accuracy"] >= 0.5, f"Accuracy 过低: {metrics['accuracy']:.1%}"


@pytest.mark.skipif(not _has_cache_or_key(), reason="未设置 API key 且无缓存")
def test_with_real_api():
    results = run_benchmark(use_api=True)
    metrics = compute_metrics(results)
    print_report(results, metrics)
    assert metrics["accuracy"] >= 0.5, f"Accuracy 过低: {metrics['accuracy']:.1%}"


if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------
