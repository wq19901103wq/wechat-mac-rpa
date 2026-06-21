#!/usr/bin/env python3
"""
CaseGenerator - 根据 draft 生成 benchmark case 代码

支持模块：
  P0: test_tool_decision_benchmark.py（工具调用决策）
  P2: test_reply_quality_benchmark.py（回复质量）
  P3: test_reply_quality_benchmark.py（多轮纠正，暂时复用 P2 文件）
"""


from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent


class CaseGenerator:
    """根据 draft 生成对应 benchmark 模块的 case 代码"""

    def generate(self, draft: Dict) -> Dict:
        module = self._route_module(draft)
        if module == "P0":
            code = self._generate_p0_case(draft)
        elif module == "P2":
            code = self._generate_p2_case(draft)
        elif module == "P3":
            code = self._generate_p3_case(draft)
        else:
            code = self._generate_p2_case(draft)  # 默认 P2
        return {"module": module, "case_code": code}

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------
    def _route_module(self, draft: Dict) -> str:
        type_to_module = {
            "missing_tool_call": "P0",
            "hallucination": "P2",
            "wrong_fact": "P2",
            "correction_not_persistent": "P3",
        }
        badcase_type = draft.get("judge_result", {}).get("badcase_type", "hallucination")
        return type_to_module.get(badcase_type, "P2")

    # ------------------------------------------------------------------
    # P0: Tool 决策
    # ------------------------------------------------------------------
    def _generate_p0_case(self, draft: Dict) -> str:
        tick_id = draft["tick_id"]
        badcase_type = draft["judge_result"]["badcase_type"]
        user_msg = self._extract_last_user_message(draft)
        should_call = "True" if badcase_type != "missing_tool_call" else "False"
        category = badcase_type
        notes = draft["judge_result"].get("reason", "")
        expected = draft["judge_result"].get("expected_behavior", "")

        case_name = f"auto_{badcase_type}_{tick_id}"
        # 转义引号
        user_msg_escaped = user_msg.replace('"', '\\"')
        notes_escaped = notes.replace('"', '\\"')
        expected_escaped = expected.replace('"', '\\"')

        return f'''    BenchmarkCase(
        case_name="{case_name}",
        user_message="{user_msg_escaped}",
        should_call_memory={should_call},
        category="{category}",
        notes="{notes_escaped}. Expected: {expected_escaped}",
    ),
'''

    # ------------------------------------------------------------------
    # P2: 回复质量
    # ------------------------------------------------------------------
    def _generate_p2_case(self, draft: Dict) -> str:
        tick_id = draft["tick_id"]
        badcase_type = draft["judge_result"]["badcase_type"]
        reason = draft["judge_result"].get("reason", "")
        expected = draft["judge_result"].get("expected_behavior", "")

        case_name = f"auto_{badcase_type}_{tick_id}"
        conversation = draft.get("conversation", [])

        # 构建 _make_msg 列表
        msg_lines = []
        for i, turn in enumerate(conversation):
            sender_type = "SenderType.SELF" if turn.get("role") == "bot" else "SenderType.OTHER"
            sender_name = turn.get("sender", "Alice")
            text = (turn.get("text") or "").replace('"', '\\"')
            create_time = 10.0 + i * 10
            msg_lines.append(
                f'_make_msg("{text}", "{sender_name}", sender_type={sender_type}, create_time={create_time})'
            )

        all_messages = ",\n            ".join(msg_lines) if msg_lines else ""
        unreplied = msg_lines[-1] if msg_lines else '_make_msg("", "Alice")'

        # 根据 badcase_type 选择 rubric 和 keywords
        rubric_map = {
            "hallucination": "self_msg_hallucination",
            "wrong_fact": "correction_down_payment",
        }
        rubric_name = rubric_map.get(badcase_type, "unknown_info")

        # required / forbidden keywords
        if badcase_type == "hallucination":
            required = ["瞎编", "记错", "没记到", "不知道", "不记得"]
            forbidden: list[str] = []
        elif badcase_type == "wrong_fact":
            required = ["错", "抱歉", "记错"]
            forbidden = []
        else:
            required = ["不知道", "没记到", "不记得", "不清楚"]
            forbidden = []

        required_str = ", ".join(f'"{k}"' for k in required)
        forbidden_str = ", ".join(f'"{k}"' for k in forbidden)
        notes_escaped = reason.replace('"', '\\"')
        expected_escaped = expected.replace('"', '\\"')

        return f'''    BenchmarkCase(
        case_name="{case_name}",
        unreplied=[{unreplied}],
        all_messages=[
            {all_messages}
        ],
        is_group=False,
        required_keywords=[{required_str}],
        required_hits=1,
        forbidden_keywords=[{forbidden_str}],
        min_replies=1,
        max_replies=3,
        category="correction",
        rubric=_CUSTOM_RUBRICS["{rubric_name}"],
        notes="{notes_escaped}. Expected: {expected_escaped}",
    ),
'''

    # ------------------------------------------------------------------
    # P3: 多轮纠正（暂时用 P2 格式，标记为 multi_turn）
    # ------------------------------------------------------------------
    def _generate_p3_case(self, draft: Dict) -> str:
        # P3 复用 P2 的生成逻辑，但 category 标记不同
        code = self._generate_p2_case(draft)
        # 把 category="correction" 替换为 category="multi_turn"
        code = code.replace('category="correction"', 'category="multi_turn"')
        return code

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def _extract_last_user_message(self, draft: Dict) -> str:
        conversation = draft.get("conversation", [])
        for turn in reversed(conversation):
            if turn.get("role") == "user":
                return turn.get("text", "")
        return ""
