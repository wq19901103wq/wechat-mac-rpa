#!/usr/bin/env python3
"""
Tick 级调试日志 —— 保存每个截图的完整处理链路，方便排查。

输出：data/debug/tick_<timestamp>_<tick_id>.json
包含：OCR 原始输出 → 布局分组 → 消息提取 → Bot 决策 → 最终 action
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)


@dataclass
class TickDebugInfo:
    """单个 tick 的完整调试信息 —— 每个模块的输入输出都要记录。"""

    tick_id: int = 0
    timestamp: str = ""
    screenshot_path: str = ""

    # ===== Layer 0: Perception 输入输出 =====
    # 输入
    perception_screenshot_path: str = ""
    # 输出
    perception_chat_name: str = ""
    perception_messages_count: int = 0
    perception_chat_list_count: int = 0

    # Layer 0.1: OCR（本地路径时填充）
    ocr_elements: List[Dict[str, Any]] = field(default_factory=list)

    # Layer 0.2: Layout（本地路径时填充）
    layout_left_elements: List[Dict[str, Any]] = field(default_factory=list)
    layout_right_elements: List[Dict[str, Any]] = field(default_factory=list)
    layout_title_elements: List[Dict[str, Any]] = field(default_factory=list)
    layout_input_elements: List[Dict[str, Any]] = field(default_factory=list)
    layout_timestamp_elements: List[Dict[str, Any]] = field(default_factory=list)
    layout_chat_list_groups: List[List[str]] = field(default_factory=list)
    layout_chat_list_nicknames: List[str] = field(default_factory=list)
    layout_chat_list_unread: List[str] = field(default_factory=list)
    layout_message_candidates: List[Dict[str, Any]] = field(default_factory=list)
    layout_self_bubbles: List[Dict[str, Any]] = field(default_factory=list)

    # Layer 0.3: Extraction
    extraction_clusters: List[Dict[str, Any]] = field(default_factory=list)
    extraction_messages: List[Dict[str, Any]] = field(default_factory=list)

    # Layer 0.4: API（API 路径时填充）
    api_prompt: str = ""
    api_response: str = ""
    api_chat_list: List[Dict[str, Any]] = field(default_factory=list)

    # Layer 0.5: Reply Generator（LLM 回复生成）
    reply_system_prompt: str = ""
    reply_user_prompt: str = ""
    reply_raw_response: str = ""
    reply_llm_calls: List[Dict[str, Any]] = field(default_factory=list)
    reply_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    # 完整生成链路 trace（含每次 LLM 请求的完整 messages + response）
    reply_generation_trace: List[Dict[str, Any]] = field(default_factory=list)
    # Skill 加载状态
    loaded_skills: List[str] = field(default_factory=list)
    skill_injected_content: str = ""
    # Hermes 联调专用字段
    active_llm: str = ""  # "deepseek" or "hermes"
    hermes_fallback_triggered: bool = False  # 是否触发了 use_hermes fallback
    hermes_messages: List[Dict[str, Any]] = field(default_factory=list)  # Hermes 请求 messages
    hermes_response: str = ""  # Hermes 响应原文

    # ===== Layer 1: Session 输入输出 =====
    session_input_chat_name: str = ""
    session_input_messages: List[Dict[str, Any]] = field(default_factory=list)
    session_output_unreplied: List[Dict[str, Any]] = field(default_factory=list)
    session_total_stored: int = 0

    # ===== Layer 2: Bot Decision 输入输出 =====
    bot_chat_name: str = ""
    bot_new_messages_count: int = 0
    bot_should_reply: bool = False
    bot_reply_target: str = ""
    bot_reply_text: str = ""
    bot_switch_target: str = ""
    bot_switch_reason: str = ""

    # ===== Layer 3: Action 输入输出 =====
    action: str = ""  # "send" / "switch" / "none"
    action_input: str = ""  # 发送的文本 / 切换的目标
    action_result_success: bool = False
    action_result_error: str = ""


class DebugLogger:
    """统一管理 tick 级调试信息的收集和落盘。"""

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = str(Path(__file__).parent.parent.parent / "data" / "debug")
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.current: Optional[TickDebugInfo] = None

    def start_tick(self, tick_id: int, screenshot_path: str) -> TickDebugInfo:
        self.current = TickDebugInfo(
            tick_id=tick_id,
            timestamp=datetime.now().isoformat(),
            screenshot_path=screenshot_path,
        )
        return self.current

    def save(self) -> Path:
        if self.current is None:
            raise RuntimeError("No active tick. Call start_tick() first.")
        ts = self.current.timestamp.replace(":", "-")
        path = self.base_dir / f"tick_{ts}_{self.current.tick_id:04d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.current.__dict__, f, ensure_ascii=False, indent=2)

        # 同时保存完整 prompt 为 markdown（不截断，方便阅读）
        self._save_prompt_markdown(ts)
        return path

    def _save_prompt_markdown(self, ts: str) -> None:
        """保存完整 prompt 到 markdown 文件，不截断。按实际发生顺序：OCR → 对话回复（多轮）。

        如果没有任何实质性内容（无 OCR、无对话、无 Bot 回复），跳过生成空文件。
        只保留最近 50 个 prompt 文件，防止磁盘无限增长。
        """
        if self.current is None:
            return
        prompts_dir = self.base_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        md_path = prompts_dir / f"tick_{ts}_{self.current.tick_id:04d}.md"

        lines = []
        lines.append(f"# Tick {self.current.tick_id} - {self.current.timestamp}")
        lines.append("")

        # ── 1. OCR 识图（最先发生） ──
        api_prompt = self.current.api_prompt or ""
        api_response = self.current.api_response or ""
        if api_prompt or api_response:
            lines.append("============ [OCR 识图] ============")
            lines.append("")
        if api_prompt:
            lines.append("## API Prompt（截图文字识别）")
            lines.append("```")
            lines.append(api_prompt)
            lines.append("```")
            lines.append("")
        if api_response:
            lines.append("## OCR Response")
            lines.append("```json")
            lines.append(api_response)
            lines.append("```")
            lines.append("")

        # ── 2. 对话回复（按 round 分组） ──
        trace = self.current.reply_generation_trace or []

        # 先提取 skill router（round 0）
        skill_router_req = None
        skill_router_resp = None
        for entry in trace:
            if entry.get("type") == "skill_router_request":
                skill_router_req = entry
            elif entry.get("type") == "skill_router_response":
                skill_router_resp = entry

        if skill_router_req or skill_router_resp:
            lines.append("============ [Skill 路由] ============")
            lines.append("")
            if skill_router_req:
                lines.append("## Skill Router Request")
                lines.append("```")
                lines.append(skill_router_req.get("messages", "")[0].get("content", "") if skill_router_req.get("messages") else "")
                lines.append("```")
                lines.append("")
            if skill_router_resp:
                lines.append("## Skill Router Response")
                lines.append("```")
                lines.append(skill_router_resp.get("content", ""))
                lines.append("```")
                lines.append("")

        # 按 round 分组 llm_request / llm_response / tool_execution
        # 注意：round=0 的 skill_router_request/response 已单独处理，此处跳过
        rounds: Dict[int, Dict] = {}
        for entry in trace:
            t = entry.get("type")
            if t not in ("llm_request", "llm_response", "tool_execution"):
                continue
            r = entry.get("round", 1)
            if r not in rounds:
                rounds[r] = {"request": None, "response": None, "tools": []}
            if t == "llm_request":
                rounds[r]["request"] = entry
            elif t == "llm_response":
                rounds[r]["response"] = entry
            elif t == "tool_execution":
                rounds[r]["tools"].append(entry)

        # 第一轮用完整的 system_prompt + user_prompt 替代 trace 中被截断的 messages
        sys_prompt = self.current.reply_system_prompt or ""
        user_prompt = self.current.reply_user_prompt or ""
        skill_content = self.current.skill_injected_content or ""

        for r in sorted(rounds.keys()):
            rd = rounds[r]
            req = rd["request"]
            resp = rd["response"]
            tools_exec = rd["tools"]

            lines.append(f"============ [对话回复 - 第 {r} 轮] ============")
            lines.append("")

            # Messages（LLM 输入）
            if req:
                lines.append("## Messages（发给 LLM 的完整上下文）")
                lines.append("")
                messages = req.get("messages", [])
                # 第一轮：用完整版替换被截断的 system/user prompt
                if r == 1 and messages:
                    for i, m in enumerate(messages):
                        role = m.get("role", "")
                        content = m.get("content", "")
                        if role == "system":
                            # 用完整的 sys_prompt 替换（第一轮的 system 就是人设）
                            if sys_prompt and (len(content) < len(sys_prompt) or "[truncated]" in str(content)):
                                content = sys_prompt
                        elif role == "user":
                            # 用完整的 user_prompt 替换
                            full_user = user_prompt
                            if skill_content:
                                full_user = user_prompt + "\n\n" + skill_content if user_prompt else skill_content
                            if full_user and (len(content) < len(full_user) or "[truncated]" in str(content)):
                                content = full_user
                        lines.append(f"**{role}**:")
                        lines.append("```")
                        lines.append(str(content))
                        lines.append("```")
                        lines.append("")
                else:
                    for m in messages:
                        role = m.get("role", "")
                        content = m.get("content", "")
                        lines.append(f"**{role}**:")
                        lines.append("```")
                        lines.append(str(content))
                        lines.append("```")
                        lines.append("")

                # Tools schema
                req_tools = req.get("tools")
                if req_tools:
                    lines.append("## Tools（可用工具定义）")
                    lines.append("```json")
                    lines.append(json.dumps(req_tools, ensure_ascii=False, indent=2))
                    lines.append("```")
                    lines.append("")

            # LLM Response
            if resp:
                content = resp.get("content", "")
                tcs = resp.get("tool_calls", [])
                lines.append("## LLM Response")
                lines.append("```")
                lines.append(str(content))
                lines.append("```")
                lines.append("")
                if tcs:
                    lines.append("### Tool Calls")
                    for tc in tcs:
                        lines.append(f"- `{tc.get('name')}`: {tc.get('arguments', '')}")
                    lines.append("")

            # Tool Execution Results
            if tools_exec:
                lines.append("============ [工具执行结果] ============")
                lines.append("")
                for tc in tools_exec:
                    lines.append(f"### {tc.get('tool_name')}({tc.get('arguments', '')})")
                    lines.append("```")
                    lines.append(str(tc.get("result", "")))
                    lines.append("```")
                    lines.append("")

        # ── 3. Bot 最终回复 ──
        raw_response = self.current.reply_raw_response or ""
        if raw_response:
            lines.append("============ [Bot 最终回复] ============")
            lines.append("```")
            lines.append(raw_response)
            lines.append("```")
            lines.append("")

        # 跳过空文件：如果只有标题没有实质内容，直接返回
        if len(lines) <= 2:
            return

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # 清理旧文件：只保留最近 50 个
        try:
            all_files = sorted(prompts_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            for old_file in all_files[50:]:
                old_file.unlink()
        except Exception as e:
            _logger.warning("cleanup old debug logs failed: %s", e)

        return

    def log_ocr(self, elements: List[Any]) -> None:
        if self.current is None:
            return
        self.current.ocr_elements = [
            {
                "text": getattr(e, "text", ""),
                "bbox": {
                    "x": getattr(e.bbox, "x", 0),
                    "y": getattr(e.bbox, "y", 0),
                    "w": getattr(e.bbox, "width", 0),
                    "h": getattr(e.bbox, "height", 0),
                },
                "center": {
                    "x": getattr(e.center, "x", 0),
                    "y": getattr(e.center, "y", 0),
                },
            }
            for e in elements
        ]

    def log_layout_chat_list(
        self,
        left_elements: List[Any],
        groups: List[List[Any]],
        nicknames: List[str],
        unread: List[str],
    ) -> None:
        if self.current is None:
            return
        self.current.layout_left_elements = [
            {"text": e.text, "x": e.bbox.x, "y": e.bbox.y}
            for e in left_elements
        ]
        self.current.layout_chat_list_groups = [
            [e.text for e in g] for g in groups
        ]
        self.current.layout_chat_list_nicknames = nicknames
        self.current.layout_chat_list_unread = unread

    def log_layout_message_area(
        self,
        candidates: List[Any],
        self_bubbles: List[Any],
    ) -> None:
        if self.current is None:
            return
        self.current.layout_message_candidates = [
            {"text": e.text, "cx": e.center.x, "cy": e.center.y}
            for e in candidates
        ]
        self.current.layout_self_bubbles = [
            {"x": b.x, "y": b.y, "w": b.width, "h": b.height}
            for b in self_bubbles
        ]

    def log_layout_full(
        self,
        right_elements: List[Dict[str, Any]],
        title_elements: List[Dict[str, Any]],
        input_elements: List[Dict[str, Any]],
        timestamp_elements: List[Dict[str, Any]],
    ) -> None:
        """记录右侧完整布局分区（title/input/timestamp/message）。

        参数为 dict 列表（已由 LayoutParser 转换），格式:
        [{"text": str, "x": int, "y": int}, ...]
        """
        if self.current is None:
            return
        self.current.layout_right_elements = list(right_elements)
        self.current.layout_title_elements = list(title_elements)
        self.current.layout_input_elements = list(input_elements)
        self.current.layout_timestamp_elements = list(timestamp_elements)

    def log_extraction(self, clusters: List[Any], messages: List[Any]) -> None:
        if self.current is None:
            return
        self.current.extraction_clusters = clusters
        self.current.extraction_messages = [
            {
                "text": m.text,
                "sender": m.sender,
                "sender_type": m.sender_type.value if hasattr(m.sender_type, 'value') else m.sender_type,
                "chat_name": m.chat_name,
            }
            for m in messages
        ]

    def log_bot_decision(
        self,
        chat_name: str = "",
        new_messages_count: int = 0,
        should_reply: bool = False,
        reply_target: str = "",
        reply_text: str = "",
        switch_target: str = "",
        switch_reason: str = "",
    ) -> None:
        if self.current is None:
            return
        self.current.bot_chat_name = chat_name
        self.current.bot_new_messages_count = new_messages_count
        self.current.bot_should_reply = should_reply
        self.current.bot_reply_target = reply_target
        self.current.bot_reply_text = reply_text
        self.current.bot_switch_target = switch_target
        self.current.bot_switch_reason = switch_reason

    def log_action(
        self,
        action: str = "",
        action_input: str = "",
        success: bool = False,
        error: str = "",
    ) -> None:
        """记录 Action 层的完整输入输出。"""
        if self.current is None:
            return
        self.current.action = action
        self.current.action_input = action_input
        self.current.action_result_success = success
        self.current.action_result_error = error

    def log_session(
        self,
        input_chat_name: str = "",
        input_messages: list = None,
        output_unreplied: list = None,
        total_stored: int = 0,
    ) -> None:
        """记录 Session 层的输入输出。"""
        if self.current is None:
            return
        self.current.session_input_chat_name = input_chat_name
        self.current.session_input_messages = input_messages or []
        self.current.session_output_unreplied = output_unreplied or []
        self.current.session_total_stored = total_stored

    def log_perception_output(
        self,
        chat_name: str = "",
        messages_count: int = 0,
        chat_list_count: int = 0,
    ) -> None:
        """记录 Perception 层的输出。"""
        if self.current is None:
            return
        self.current.perception_chat_name = chat_name
        self.current.perception_messages_count = messages_count
        self.current.perception_chat_list_count = chat_list_count

    def log_reply_generation(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        raw_response: str = "",
        llm_calls: list = None,
        tool_calls: list = None,
        trace: list = None,
        loaded_skills: list = None,
        skill_injected_content: str = "",
        active_llm: str = "",
        hermes_fallback_triggered: bool = False,
        hermes_messages: list = None,
        hermes_response: str = "",
    ) -> None:
        """记录 Reply Generator 的完整调用链路（含多轮 LLM + 工具调用）。"""
        if self.current is None:
            return
        self.current.reply_system_prompt = system_prompt
        self.current.reply_user_prompt = user_prompt
        self.current.reply_raw_response = raw_response
        self.current.reply_llm_calls = llm_calls or []
        self.current.reply_tool_calls = tool_calls or []
        self.current.reply_generation_trace = trace or []
        self.current.loaded_skills = loaded_skills or []
        self.current.skill_injected_content = skill_injected_content or ""
        self.current.active_llm = active_llm
        self.current.hermes_fallback_triggered = hermes_fallback_triggered
        self.current.hermes_messages = hermes_messages or []
        self.current.hermes_response = hermes_response or ""
