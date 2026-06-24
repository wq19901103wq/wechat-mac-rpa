#!/usr/bin/env python3
"""L4 Reply Generator - 回复内容生成."""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models.base import MEDIA_MESSAGE_TYPES, ChatMessage, SenderType
from src.reply.session_memory import SessionMemory, _extract_query_key
from src.tools import get_registry

_logger = logging.getLogger("src.reply.generator")

# Badcase JudgeWorker（可选，默认不启用）
_judge_worker = None
_judge_worker_lock = threading.Lock()

def _get_judge_worker():
    global _judge_worker
    if _judge_worker is None:
        with _judge_worker_lock:
            if _judge_worker is None:
                try:
                    from src.badcase.judge_worker import JudgeWorker
                    _judge_worker = JudgeWorker()
                    _logger.info("JudgeWorker initialized")
                except Exception as e:
                    _logger.warning("JudgeWorker init failed: %s", e)
    return _judge_worker


class ReplyGenerator:
    def __init__(self, llm_client=None, complex_llm_client=None, memory_engine=None,
                 tool_registry=None,
                 judge_worker=None,
                 enable_time_awareness: bool = True,
                 enable_reply_restraint: bool = True,
                 enable_unread_dedup: bool = True,
                 enable_timestamps: bool = True,
                 enable_mode_detection: Optional[bool] = None):
        self.llm_client = llm_client
        self.complex_llm_client = complex_llm_client
        self.memory_engine = memory_engine
        self.tool_registry = tool_registry or get_registry()
        self.judge_worker = judge_worker
        self.enable_time_awareness = enable_time_awareness
        self.enable_reply_restraint = enable_reply_restraint
        self.enable_unread_dedup = enable_unread_dedup
        self.enable_timestamps = enable_timestamps
        if enable_mode_detection is None:
            enable_mode_detection = os.environ.get("WECHAT_MODE_DETECTION", "0").lower() in ("1", "true", "yes", "on")
        self.enable_mode_detection = enable_mode_detection
        print(f"[Hermes] ReplyGenerator init: llm_client={type(llm_client).__name__ if llm_client else None}, complex_llm_client={type(complex_llm_client).__name__ if complex_llm_client else None}")
        # 最后一次调用的 prompt/response（供 debug 使用）
        self.last_system_prompt: str = ""
        self.last_tools_context: str = ""
        self.last_user_prompt: str = ""
        self.last_raw_response: str = ""
        self.last_thinking: str = ""
        # 多轮调用完整链路（供 debug 使用）
        self.last_llm_calls: List[Dict] = []
        self.last_tool_calls: List[Dict] = []
        self.last_generation_trace: List[Dict] = []
        # Skill 加载状态（供 debug 使用）
        self.last_loaded_skills: List[str] = []
        self.last_skill_injected_content: str = ""
        # Hermes 联调专用 debug 字段
        self.last_active_llm: str = ""
        self.last_hermes_fallback_triggered: bool = False
        self.last_hermes_messages: List[Dict] = []
        self.last_hermes_response: str = ""
        # 传给 Judge 的完整 LLM 上下文
        self.last_llm_messages: List[Dict] = []
        # 短期记忆（跨 tick 缓存工具结果）
        self.session_memory = SessionMemory()
        # 幽默聊天 RAG 索引（可选，依赖 sklearn/numpy）
        # 使用消息级索引：检索最相似的消息，并拉出前后多轮完整上下文
        self._humor_vector_index = None
        try:
            import numpy  # noqa: F401
            from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401

            from src.memory.vector_index import MessageVectorIndex
            index_path = Path(__file__).parent.parent.parent / "data" / "vector_indexes" / "humor_messages_index.json"
            if index_path.exists():
                self._humor_vector_index = MessageVectorIndex(cache_path=index_path)
                if self._humor_vector_index.messages:
                    print(f"[HumorRAG] 已加载 {len(self._humor_vector_index.messages)} 条消息级历史案例")
            else:
                print(f"[HumorRAG] 索引文件不存在: {index_path}")
        except Exception as e:
            _logger.warning("[HumorRAG] 索引初始化失败: %s", e)
            self._humor_vector_index = None
        # 动态注册记忆搜索工具（如果 memory_engine 可用）
        if self.memory_engine is not None:
            def _search_memory_adapter(query: str = "") -> str:
                """适配器：工具参数名 query → 引擎参数名 keyword"""
                return self.memory_engine.search_keyword(query)

            self.tool_registry.register(
                name="search_memory",
                description="搜索本地长期记忆（wiki）。用于查询身边的人：亲友、同事、同学、家族关系、熟人旧事。当你遇到某个人名但不知道 TA 是谁、和对方什么关系、TA 的近况/在哪/做什么、或对方提到的旧事背景时，必须调用此工具查询，禁止直接说'不知道''好久没联系了'。不要用于公众人物、公司、产品、新闻八卦。",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词。必须是单个具体的人名、昵称或名词，不要组合多个词，不要包含'关系''称呼''是谁''什么'等泛词。正确示例：'王海'、'小海哥'、'王璇'。错误示例：'王璇 王海 关系 称呼'",
                        },
                    },
                    "required": ["query"],
                },
                func=_search_memory_adapter,
            )

        # 动态注册历史原文检索工具（可选，依赖 digital-twin 的 BGE 消息索引）
        # 与 search_memory(wiki 摘要) 并列：search_history 检索历史聊天原文
        # 仅当索引文件与编码器依赖就绪时才注册，否则 bot 行为零变化
        try:
            from src.memory.history_search import is_available as _history_available
            from src.memory.history_search import search_history as _search_history
        except Exception as e:  # pragma: no cover - 防御性
            _logger.warning("[HistorySearch] 模块加载失败，跳过注册: %s", e)
            _history_available = None
            _search_history = None

        if _history_available is not None and _history_available():
            def _search_history_adapter(query: str = "", top_k: int = 5) -> str:
                """适配器：调用 history_search 返回原文片段。"""
                return _search_history(query, top_k=top_k)

            self.tool_registry.register(
                name="search_history",
                description=(
                    "搜索历史聊天原文（77万条历史微信消息的语义检索）。用于回忆"
                    "过去某段对话具体说了什么：当对方提到'上次说的那件事''之前聊过的'"
                    "'你忘了我们讨论过'、或你想复用某次具体对话的措辞/细节时调用。"
                    "返回的是历史消息原文片段（含上下文），不是人物摘要。"
                    "查询人是谁/关系/近况用 search_memory，查询当时说了什么用本工具。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "要检索的内容，用自然语言描述想找的那段对话。"
                                "示例：'关于3D打印材料选择的讨论'、'王海说要去上海那次'、"
                                "'谁推荐过那家日料'。不要只填单个泛词。"
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "返回片段数量，默认 5，范围 1-20。",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
                func=_search_history_adapter,
            )
            print("[HistorySearch] search_history 工具已注册（懒加载，首次调用时载入索引）")
        else:
            print("[HistorySearch] 索引/依赖未就绪，search_history 未注册")

    def _submit_to_judge(self, tick_id: int, replies: List[str], unreplied: List[ChatMessage], all_messages: List[ChatMessage], is_group: bool):
        """把当前 tick 的数据提交给 JudgeWorker 异步判定"""
        import json
        try:
            worker = self.judge_worker
            if worker is None:
                return
            tick_data = {
                "tick_id": tick_id,
                "chat_name": unreplied[-1].chat_name if unreplied else "",
                "session_input_messages": [
                    {
                        "sender": m.sender,
                        "sender_type": m.sender_type.value,
                        "text": m.text or "",
                        "chat_name": m.chat_name,
                    }
                    for m in all_messages
                ],
                "bot_reply_text": " | ".join(replies) if replies else "",
                "reply_text": " | ".join(replies) if replies else "",
                "tool_calls": self.last_tool_calls,
                "memory_injected": self.last_user_prompt,
                "full_user_prompt": self.last_user_prompt,
                "reply_raw_response": self.last_raw_response,
                "reply_generation_trace": self.last_generation_trace,
                "full_system_prompt": self.last_system_prompt,
                "full_tools_context": self.last_tools_context,
                "full_llm_messages": self.last_llm_messages,
                "created_at": __import__('datetime').datetime.now().isoformat(),
                "tool_results_json": json.dumps(
                    [{"tool": t.get("tool_name", ""), "args": t.get("arguments", ""), "result": str(t.get("result_preview", ""))}
                     for t in (self.last_tool_calls or [])], ensure_ascii=False
                ),
            }
            worker.submit(tick_data)
        except Exception as e:
            _logger.debug("JudgeWorker submit failed: %s", e)

    def generate(self, unreplied: List[ChatMessage], all_messages: List[ChatMessage],
                 is_group: bool = False, tick_id: int = 0,
                 enable_time_awareness: Optional[bool] = None,
                 enable_reply_restraint: Optional[bool] = None,
                 enable_unread_dedup: Optional[bool] = None,
                 enable_timestamps: Optional[bool] = None) -> List[str]:
        """
        生成回复内容，返回多条回复列表（最多3条）。
        支持多轮工具调用，但总工具时间不超过 max_tool_seconds，超时后强制生成文本回复。
        """
        t_generate_start = time.time()
        if not unreplied:
            self._submit_to_judge(tick_id, [], unreplied, all_messages, is_group)
            return []

        if self.llm_client is None:
            self._submit_to_judge(tick_id, [], unreplied, all_messages, is_group)
            return []

        # 重置 debug 状态
        self.last_system_prompt = ""
        self.last_tools_context = ""
        self.last_user_prompt = ""
        self.last_raw_response = ""
        self.last_llm_calls = []
        self.last_tool_calls = []
        self.last_generation_trace = []
        self.last_loaded_skills = []
        self.last_skill_injected_content = ""
        self.last_hermes_fallback_triggered = False
        self.last_hermes_messages = []
        self.last_hermes_response = ""
        self.last_llm_messages = []

        chat_name = unreplied[-1].chat_name if unreplied else ""

        # 模型辅助路由：提前到 system prompt 之前，让 prompt 也能感知已加载 skill
        last_msg = unreplied[-1]
        route_text = last_msg.text or last_msg.image_description or ""
        t_route_start = time.time()
        matched_skills = self._route_skills(route_text)
        t_route_ms = (time.time() - t_route_start) * 1000
        self.last_loaded_skills = matched_skills

        # 模型选择：加载了 skill 的复杂任务优先走 complex_llm_client（hermes）
        active_llm = self.llm_client
        active_llm_name = "deepseek"
        is_hermes = False
        if matched_skills and self.complex_llm_client is not None:
            active_llm = self.complex_llm_client
            active_llm_name = "hermes"
            is_hermes = True
            print(f"[Hermes] matched_skills={matched_skills} → 切换 active_llm=hermes，让 Hermes 自己加载 skill")
        else:
            has_hermes = self.complex_llm_client is not None
            print(f"[Hermes] matched_skills={matched_skills}, complex_llm_available={has_hermes} → active_llm=deepseek")
        self.last_active_llm = active_llm_name

        t_sp_start = time.time()
        system_prompt = self._system_prompt(
            enable_reply_restraint=enable_reply_restraint,
            unreplied=unreplied,
            all_messages=all_messages,
            is_group=is_group,
        )
        t_sp_ms = (time.time() - t_sp_start) * 1000

        t_tc_start = time.time()
        tools_context = self._build_tools_context(chat_name)
        t_tc_ms = (time.time() - t_tc_start) * 1000

        t_up_start = time.time()
        user_prompt = self._build_user_prompt(
            unreplied, all_messages, is_group,
            enable_time_awareness=enable_time_awareness,
            enable_unread_dedup=enable_unread_dedup,
            enable_timestamps=enable_timestamps,
            tools_context=tools_context,
        )
        t_up_ms = (time.time() - t_up_start) * 1000

        # Skill 注入：只有 deepseek 才注入 Bot 的 skill，Hermes 用自己的 skill
        skill_parts = []
        if matched_skills and not is_hermes:
            for skill_name in matched_skills:
                content = self._load_skill_content(skill_name)
                if content:
                    skill_parts.append(f"【{skill_name} 技能指南】\n{content}")
            if skill_parts:
                user_prompt += "\n\n" + "\n\n".join(skill_parts)
        self.last_skill_injected_content = "\n\n".join(skill_parts) if skill_parts else ""

        self.last_system_prompt = system_prompt
        self.last_tools_context = tools_context
        self.last_user_prompt = user_prompt

        tools = self.tool_registry.to_openai_schemas()

        llm_calls: List[Dict] = []
        tool_calls: List[Dict] = []
        trace: List[Dict] = []

        max_retries = 2
        max_tool_seconds = 25.0  # 工具调用阶段最多 25 秒
        max_total_seconds = 600.0 if is_hermes else 60.0  # deepseek 给 60 秒，给工具调用留余量
        overall_start_time = time.time()

        # 构建 messages：system（人设）+ user（上下文含缓存）
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": user_prompt})

        # Hermes 走精简 system prompt，不传 tools
        if is_hermes:
            messages[0]["content"] = self._hermes_system_prompt()
            print("[Hermes] 使用精简 system prompt，不传 tools")
        tool_round_count = 0  # 已执行的 tool 轮数

        for attempt in range(max_retries + 1):
            start_time = time.time()

            try:
                while True:
                    elapsed = time.time() - start_time
                    total_elapsed = time.time() - overall_start_time
                    force_no_tools = elapsed > max_tool_seconds

                    # 总时间兜底
                    if total_elapsed > max_total_seconds:
                        force_no_tools = True

                    # 调用 LLM（matched_skills 时走 active_llm=hermes）
                    actual_tools = None if (force_no_tools or is_hermes) else (tools if tools else None)
                    llm_timeout = 600 if is_hermes else 30
                    _logger.info("[LLM] attempt=%d round=%d force_no_tools=%s tools=%s timeout=%s msg_count=%d",
                                 attempt + 1, tool_round_count, force_no_tools, bool(actual_tools), llm_timeout, len(messages))
                    t_llm_start = time.time()
                    raw = active_llm.chat(messages=messages, tools=actual_tools, max_tokens=2000, timeout=llm_timeout)
                    self.last_thinking = getattr(active_llm, "last_thinking", "") or ""
                    self.last_llm_messages = [dict(m) for m in messages]
                    t_llm_ms = (time.time() - t_llm_start) * 1000
                    raw_content = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
                    raw_tool_calls = getattr(raw, "tool_calls", None)
                    _logger.info("[LLM] attempt=%d round=%d 完成 耗时=%.0fms type=%s",
                                 attempt + 1, tool_round_count, t_llm_ms,
                                 "tool_calls" if raw_tool_calls else "text")

                    # 记录 LLM 调用（summary）
                    llm_calls.append({
                        "attempt": attempt + 1,
                        "elapsed": round(elapsed, 2),
                        "messages_count": len(messages),
                        "has_tools": bool(actual_tools),
                        "has_tool_calls": bool(raw_tool_calls),
                        "response_preview": raw_content[:500] if raw_content else "",
                    })

                    # 记录完整 trace（请求）
                    trace.append({
                        "round": len(trace) // 3 + 1,
                        "type": "llm_request",
                        "timestamp": time.time(),
                        "attempt": attempt + 1,
                        "messages": self._truncate_messages(messages),
                        "tools": actual_tools,
                        "force_no_tools": force_no_tools,
                    })

                    # 记录完整 trace（响应）
                    trace.append({
                        "round": len(trace) // 3 + 1,
                        "type": "llm_response",
                        "timestamp": time.time(),
                        "content": raw_content[:2000] if raw_content else "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                            for tc in (raw_tool_calls or [])
                        ],
                    })

                    # 工具调用处理
                    if raw_tool_calls and not force_no_tools:
                        assistant_msg = {
                            "role": "assistant",
                            "content": raw.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in raw_tool_calls
                            ],
                        }
                        # DeepSeek thinking mode 需要回传 reasoning_content，否则 round 1 会空返
                        if hasattr(raw, "reasoning_content") and raw.reasoning_content:
                            assistant_msg["reasoning_content"] = raw.reasoning_content
                        messages.append(assistant_msg)

                        for tc in raw_tool_calls:
                            tool_name = tc.function.name
                            tool_args = tc.function.arguments
                            _logger.info("[Tool] 执行开始: %s(%s)", tool_name, tool_args[:100] if isinstance(tool_args, str) else str(tool_args)[:100])
                            t_tool_start = time.time()
                            if self.tool_registry.has(tool_name):
                                result = self.tool_registry.get(tool_name).execute(tool_args)
                            else:
                                result = f"工具 {tool_name} 不存在"
                            t_tool_ms = (time.time() - t_tool_start) * 1000
                            _logger.info("[Tool] 执行完成: %s 耗时=%.0fms result_len=%d", tool_name, t_tool_ms, len(str(result)) if result else 0)

                            # 保存到 session memory（跨 tick 缓存）
                            try:
                                query_key = _extract_query_key(tool_name, tool_args)
                                self.session_memory.add_tool_result(chat_name, tool_name, query_key, str(result)[:1000])
                            except Exception as e:
                                _logger.warning("add tool result failed: %s", e)

                            # 记录工具调用（summary）
                            tool_calls.append({
                                "attempt": attempt + 1,
                                "tool_call_id": tc.id,
                                "tool_name": tool_name,
                                "arguments": tool_args,
                                "result_preview": str(result) if result else "",
                            })

                            # 记录完整 trace（工具执行）
                            trace.append({
                                "round": len(trace) // 3 + 1,
                                "type": "tool_execution",
                                "timestamp": time.time(),
                                "tool_call_id": tc.id,
                                "tool_name": tool_name,
                                "arguments": tool_args,
                                "result": str(result)[:2000] if result else "",
                            })

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result,
                            })

                        # 完成一轮 tool 调用
                        tool_round_count += 1
                        continue

                    # 文本回复（或强制无 tools 后的回复）
                    text = raw_content
                    self.last_raw_response = text
                    t_parse_start = time.time()
                    replies = self._parse_replies(text)
                    t_parse_ms = (time.time() - t_parse_start) * 1000
                    if replies:
                        model_name = "hermes" if active_llm is self.complex_llm_client else "deepseek"
                        t_total_ms = (time.time() - t_generate_start) * 1000
                        print(f"[Perf][Generate] total={t_total_ms:.0f}ms "
                              f"sp={t_sp_ms:.0f}ms tc={t_tc_ms:.0f}ms up={t_up_ms:.0f}ms "
                              f"route={t_route_ms:.0f}ms llm={sum(c.get('elapsed',0) for c in llm_calls)*1000:.0f}ms "
                              f"parse={t_parse_ms:.0f}ms replies={len(replies)}")
                        print(f"[Hermes] {model_name} 直接生成 replies={len(replies)} 条")
                        # 直接走 Hermes 时也记录 hermes debug 字段
                        if active_llm is self.complex_llm_client:
                            self.last_hermes_messages = [dict(m) for m in messages]
                            self.last_hermes_response = text or ""
                        for r in replies:
                            self.session_memory.add_reply(chat_name, r)
                        self.last_llm_calls = llm_calls
                        self.last_tool_calls = tool_calls
                        self.last_generation_trace.extend(trace)
                        self._submit_to_judge(tick_id, replies, unreplied, all_messages, is_group)
                        return replies

                    # deepseek 请求切换 hermes → 保留 tool 结果上下文，只换 system prompt
                    if text and '"use_hermes"' in text and self.complex_llm_client is not None:
                        print("[Hermes] deepseek 输出 use_hermes → 切 Hermes 重新生成")
                        self.last_hermes_fallback_triggered = True
                        # 基于当前 messages（含 tool 调用结果），替换 system prompt
                        hermes_system = self._hermes_system_prompt()
                        hermes_messages = [dict(m) for m in messages]
                        if hermes_messages and hermes_messages[0].get("role") == "system":
                            hermes_messages[0]["content"] = hermes_system
                        else:
                            hermes_messages.insert(0, {"role": "system", "content": hermes_system})

                        self.last_generation_trace.append({
                            "round": len(trace) // 3 + 1,
                            "type": "hermes_fallback_request",
                            "timestamp": time.time(),
                            "note": "deepseek 判定需复杂推理，切 hermes 重新生成",
                            "messages_count": len(hermes_messages),
                        })
                        print(f"[Hermes] 请求: messages={len(hermes_messages)} 条, 含 tool={any(m.get('role')=='tool' for m in hermes_messages)}")
                        hermes_raw = self.complex_llm_client.chat(messages=hermes_messages, tools=None, max_tokens=2000)
                        self.last_llm_messages = [dict(m) for m in hermes_messages]
                        hermes_text = hermes_raw if isinstance(hermes_raw, str) else getattr(hermes_raw, "content", str(hermes_raw))
                        print(f"[Hermes] 响应预览: {hermes_text[:100] if hermes_text else '(空)'}...")
                        self.last_hermes_messages = hermes_messages
                        self.last_hermes_response = hermes_text or ""
                        self.last_generation_trace.append({
                            "round": len(trace) // 3 + 1,
                            "type": "hermes_fallback_response",
                            "timestamp": time.time(),
                            "content": hermes_text[:2000] if hermes_text else "",
                        })
                        hermes_replies = self._parse_replies(hermes_text)
                        if hermes_replies:
                            print(f"[Hermes] 生成 replies={len(hermes_replies)} 条")
                            for r in hermes_replies:
                                self.session_memory.add_reply(chat_name, r)
                            self.last_llm_calls = llm_calls
                            self.last_tool_calls = tool_calls
                            self.last_raw_response = hermes_text
                            self.last_generation_trace.extend(trace)
                            self._submit_to_judge(tick_id, hermes_replies, unreplied, all_messages, is_group)
                            return hermes_replies
                        # hermes 也返回空 → fallback 到不回复
                        print("[Hermes] 返回空 replies")
                        self.last_llm_calls = llm_calls
                        self.last_tool_calls = tool_calls
                        self.last_generation_trace.extend(trace)
                        self._submit_to_judge(tick_id, [], unreplied, all_messages, is_group)
                        return []

                    # LLM 明确输出了 {"replies": []} → 正确决策（不想回复），不 retry
                    if text and '"replies"' in text:
                        print("[Hermes] LLM 输出空 replies → 正确决策不回复")
                        self.last_llm_calls = llm_calls
                        self.last_tool_calls = tool_calls
                        self.last_generation_trace.extend(trace)
                        self._submit_to_judge(tick_id, [], unreplied, all_messages, is_group)
                        return []

                    # 空回复处理（LLM 返回空字符串或无效内容）
                    if force_no_tools:
                        # 禁用 tools 后返回空，可能是 LLM 还在尝试调用工具
                        # 继续外层 retry，给 LLM 一次基于已有信息直接回复的机会
                        print("[Hermes] force_no_tools 空回复，继续 retry")
                        self.last_raw_response = f"[空回复且已禁用tools，attempt={attempt+1}]"
                        break  # 跳出 while，进入下一次 retry

                    self.last_raw_response = f"[空回复，attempt={attempt+1}]"
                    break  # 跳出 while，进入下一次 retry

            except Exception as e:
                self.last_raw_response = f"[ERROR attempt={attempt+1}: {type(e).__name__}: {e}]"
                llm_calls.append({
                    "attempt": attempt + 1,
                    "error": f"{type(e).__name__}: {e}",
                })
                trace.append({
                    "type": "error",
                    "timestamp": time.time(),
                    "attempt": attempt + 1,
                    "error": f"{type(e).__name__}: {e}",
                })
                if attempt < max_retries:
                    time.sleep(1)

        print("[Hermes] generate 最终返回空 ( exhausted retries )")
        self.last_llm_calls = llm_calls
        self.last_tool_calls = tool_calls
        self.last_generation_trace.extend(trace)
        self._submit_to_judge(tick_id, [], unreplied, all_messages, is_group)
        return []

    def _truncate_messages(self, messages: List[Dict]) -> List[Dict]:
        """截断 OpenAI message 数组，防止 debug JSON 过大。

        注意：此函数处理的是 List[Dict] 结构（含 content + tool_calls），
        与通用 text_utils._truncate_text（处理 str）不同，因此保留独立实现。
        阈值设高些，markdown 会单独保存完整版。
        """
        truncated = []
        for m in messages:
            cm = dict(m)
            if "content" in cm and isinstance(cm["content"], str) and len(cm["content"]) > 10000:
                cm["content"] = cm["content"][:10000] + "\n\n... [truncated, see markdown for full content]"
            if "tool_calls" in cm:
                cm["tool_calls"] = [{"id": tc.get("id"), "name": tc.get("function", {}).get("name")} for tc in cm["tool_calls"]]
            truncated.append(cm)
        return truncated

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """从 LLM 回复中提取 JSON 对象。支持 markdown 代码块和裸 JSON。
        使用 json.JSONDecoder.raw_decode() 精确解析，避免手动括号计数
        在字符串内遇到 } 时误判 JSON 边界的问题。
        """
        import json
        text = text.strip()
        if not text:
            return None
        # 去掉 markdown 代码块
        if "```" in text:
            parts = text.split("```", 2)
            if len(parts) >= 3:
                code_content = parts[1]
                # 去掉可能的 "json" 语言标记
                if code_content.lstrip().startswith("json"):
                    code_content = code_content.lstrip()[4:].lstrip()
                text = code_content
        # 找到第一个 { 的位置，尝试 raw_decode
        start = text.find("{")
        if start < 0:
            return None
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(text, start)
            return obj
        except json.JSONDecodeError:
            # 如果 raw_decode 失败，尝试清理常见问题后重试
            # 例如 LLM 在 JSON 前后加了多余文字
            for idx in range(start, len(text)):
                if text[idx] == '{':
                    try:
                        obj, _ = decoder.raw_decode(text, idx)
                        return obj
                    except json.JSONDecodeError:
                        continue
            return None

    def _parse_replies(self, text: str) -> List[str]:
        """解析 LLM 回复：{"replies": ["msg1", "msg2"]}。prompt 已要求此格式。"""
        if not text or not text.strip():
            return []
        data = self._extract_json(text)
        if data is not None:
            replies = data.get("replies", [])
            return [str(r).strip() for r in replies if str(r).strip() not in ("收到", "好的", "嗯", "OK", "1")][:3]
        # fallback: 按段落拆分，不再整段当一条发
        text = text.strip()
        for sep in ("\n\n", "\n"):
            parts = [p.strip().replace("\n", " ") for p in text.split(sep) if p.strip()]
            if len(parts) > 1:
                return [p for p in parts if p not in ("收到", "好的", "嗯", "OK", "1")][:3]
        return [text.replace("\n", " ")] if text not in ("收到", "好的", "嗯", "OK", "1") else []

    def _load_skill_manifest(self) -> List[Dict[str, str]]:
        """扫描 skills/ 目录，返回技能清单（name + trigger 描述），不含正文。"""
        skills_dir = Path(__file__).parent.parent.parent / "skills"
        if not skills_dir.exists():
            return []
        manifest = []
        for skill_dir in sorted(skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            md_file = skill_dir / "SKILL.md"
            if md_file.exists():
                text = md_file.read_text(encoding="utf-8")
                # 提取第一行标题作为描述
                first_line = text.strip().split("\n")[0].replace("#", "").strip()
                manifest.append({
                    "name": skill_dir.name,
                    "description": first_line,
                })
        return manifest

    def _load_skill_content(self, skill_name: str) -> str:
        """加载指定 skill 的完整 SKILL.md 内容。"""
        md_file = Path(__file__).parent.parent.parent / "skills" / skill_name / "SKILL.md"
        if md_file.exists():
            return md_file.read_text(encoding="utf-8").strip()
        return ""

    def _route_skills(self, user_text: str) -> List[str]:
        """模型辅助路由：根据用户消息判断需要加载哪些 skill。
        用一次轻量 LLM 调用，只消耗几十 token。
        """
        if not user_text or not self.llm_client:
            return []

        manifest = self._load_skill_manifest()
        if not manifest:
            return []

        # 构建轻量路由 prompt
        skill_list = "\n".join(
            f"{i+1}. {s['name']}：{s['description']}"
            for i, s in enumerate(manifest)
        )
        router_prompt = (
            "你是 SkillRouter，只负责判断用户消息需要哪些技能。\n\n"
            f"可用技能：\n{skill_list}\n\n"
            f"用户消息：\"{user_text}\"\n\n"
            "请输出 JSON，只包含技能 name 列表，不要其他内容：\n"
            '{"skills": ["skill_name1", "skill_name2"]}\n'
            "如果不需要任何技能，输出：{\"skills\": []}"
        )

        try:
            # 记录路由请求到 trace
            if hasattr(self, 'last_generation_trace') and isinstance(self.last_generation_trace, list):
                self.last_generation_trace.append({
                    "round": 0,
                    "type": "skill_router_request",
                    "timestamp": time.time(),
                    "messages": [{"role": "user", "content": router_prompt[:500] + "..." if len(router_prompt) > 500 else router_prompt}],
                    "temperature": 0.0,
                    "max_tokens": 256,
                })
            raw = self.llm_client.chat(
                messages=[{"role": "user", "content": router_prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            raw_str = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
            # 记录路由响应到 trace
            raw_preview = (raw_str[:200]) if raw_str else ""
            if hasattr(self, 'last_generation_trace') and isinstance(self.last_generation_trace, list):
                self.last_generation_trace.append({
                    "round": 0,
                    "type": "skill_router_response",
                    "timestamp": time.time(),
                    "content": raw_preview,
                })
            # 尝试解析 JSON
            if raw_str:
                data = self._extract_json(raw_str)
                if data is not None:
                    matched = data.get("skills", [])
                    # 过滤有效 skill
                    valid = {s["name"] for s in manifest}
                    result = [name for name in matched if name in valid]
                    print(f"[SkillRouter] 用户消息: {user_text[:30]}... -> 匹配技能: {result}")
                    return result
                else:
                    print(f"[SkillRouter] 用户消息: {user_text[:30]}... -> 未找到 JSON，原始响应: {raw_str[:100]}")
        except Exception as e:
            print(f"[SkillRouter] 路由异常: {type(e).__name__}: {e}")
            if hasattr(self, 'last_generation_trace') and isinstance(self.last_generation_trace, list):
                self.last_generation_trace.append({
                    "round": 0,
                    "type": "skill_router_error",
                    "timestamp": time.time(),
                    "error": f"{type(e).__name__}: {e}",
                })
        return []

    def _load_skill_one_liners(self) -> str:
        """加载所有 skill 的一句话摘要（始终放在 system prompt 中，极简）。
        从 SKILL.md 的'触发条件'段落提取第一句话。"""
        skills_dir = Path(__file__).parent.parent.parent / "skills"
        if not skills_dir.exists():
            return ""
        parts = []
        for skill_dir in sorted(skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            md_file = skill_dir / "SKILL.md"
            if md_file.exists():
                text = md_file.read_text(encoding="utf-8").strip()
                if text.strip():
                    name = skill_dir.name
                    lines = text.split("\n")
                    summary = ""
                    in_trigger = False
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith("## 触发条件"):
                            in_trigger = True
                            continue
                        if in_trigger:
                            if stripped.startswith("##") or not stripped:
                                break
                            summary = stripped
                            break
                    if summary:
                        # 截断到 40 字以内
                        if len(summary) > 40:
                            summary = summary[:37] + "..."
                        parts.append(f"- {name}：{summary}")
        if parts:
            return "\n可用技能（系统会根据对话内容自动下发详细框架）：\n" + "\n".join(parts) + "\n"
        return ""

    def _build_humor_query(self, all_messages: List[ChatMessage]) -> str:
        """为 humor_chat 检索构建 query：取最近 3 轮非自己发的文本消息拼接。"""
        if not all_messages:
            return ""
        # 从后往前找最近 3 条非自己且有文本的消息
        recent_others = []
        for m in reversed(all_messages):
            if m.sender_type != SenderType.SELF and (m.text or "").strip():
                recent_others.append(m)
            if len(recent_others) >= 3:
                break
        if not recent_others:
            return ""
        # 转回时间正序
        recent_others = list(reversed(recent_others))
        return " | ".join(m.text.strip() for m in recent_others)

    def _load_humor_retrieval_cases(self, query: str, sender_name: str, chat_type: str) -> str:
        """为 humor_chat skill 检索相似历史对话片段（含完整上下文），格式化为 few-shot。"""
        if not self._humor_vector_index:
            return ""
        try:
            results = self._humor_vector_index.search(
                query,
                sender_name=sender_name or "",
                chat_type="group" if chat_type == "group" else "single",
                top_k=3,
                context_radius=5,
            )
            if not results:
                return ""
            parts = []
            for i, result in enumerate(results, 1):
                context_text = self._humor_vector_index.format_context(result['context_messages'])
                parts.append(f"--- 历史对话片段 {i} ---")
                parts.append(context_text)
            return "\n\n".join(parts)
        except Exception as e:
            _logger.warning("[HumorRAG] search failed: %s", e)
            return ""

    def _hermes_system_prompt(self) -> str:
        """hermes 专用 system prompt：不含 tool 列表，只保留风格+格式。"""
        return (
            "核心人设与风格\n"
            "你没有固定人设。参照对方的说话风格自然融入对话。\n"
            '风格：casual，不用"您"，略带傲娇，适当幽默反转。\n'
            '口头禅：对方炫富/晒成就时可用"羡慕你们这些有钱人"、"被你装到了"等。\n'
            "不要重复自己之前说过的梗，每次回复都是新的。\n"
            "\n"
            "你可以使用 skills_list 查看可用技能，用 skill_view 加载需要的技能。"
            "遇到专业领域的问题，先加载对应 skill 再回答。"
            "回复开头请用【Skill: xxx】标注使用了哪些 skill。"
            "\n\n"
            "输出格式\n"
            "直接输出 JSON（不要 markdown 代码块）：\n"
            '{"replies": ["回复内容1", "回复内容2"]}\n'
            "replies 数组包含 0-5 条，不需要回复时为空数组 []。\n"
            "绝对不要输出思考过程。\n"
            "\n"
            "规则\n"
            "1. 每条回复简洁自然，不超过300字\n"
            "2. 不懂、不确定的话题，输出空 replies []\n"
            "3. 禁止敷衍词：收到、好的、嗯、OK、1\n"
            "4. 参照对方语气回复，不要延续自己的风格\n"
        )

    def _system_prompt(self, enable_reply_restraint: Optional[bool] = None,
                       unreplied: Optional[List[ChatMessage]] = None,
                       all_messages: Optional[List[ChatMessage]] = None,
                       is_group: bool = False) -> str:
        """核心 system prompt：读 prompts/persona.md（DT 风格），注入工具描述 + 检索案例。"""
        if enable_reply_restraint is None:
            enable_reply_restraint = self.enable_reply_restraint

        # 支持通过环境变量/参数切换 system prompt 版本
        if self.enable_mode_detection:
            prompt_file = "persona_mode_detection.md"
        else:
            prompt_file = "persona.md"
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / prompt_file
        if prompt_path.exists():
            prompt = prompt_path.read_text(encoding="utf-8")
        else:
            # fallback 到默认 prompt
            prompt = "你是王芊本人。用户不是在跟AI聊天，是在微信上给王芊发消息。"

        # 根据开关删除回复克制原则 section
        if not enable_reply_restraint:
            marker = "### 7. 回复克制原则"
            idx = prompt.find(marker)
            if idx != -1:
                next_section = prompt.find("### ", idx + 1)
                if next_section != -1:
                    prompt = prompt[:idx] + prompt[next_section:]
                else:
                    prompt = prompt[:idx]

        # 注入工具描述
        tools_desc = "\n".join(
            f"- {t.name}：{t.description}"
            for t in self.tool_registry._tools.values()
        )
        prompt = prompt.replace("{tools_description}", tools_desc)

        # 注入检索 few-shot
        dynamic_few_shot = "（无相关历史对话）"
        if "humor_chat" in self.last_loaded_skills:
            try:
                query = self._build_humor_query(all_messages or unreplied or [])
                sender_name = ""
                if unreplied:
                    sender_name = unreplied[-1].sender or ""
                chat_type = "group" if is_group else "single"
                if query:
                    cases = self._load_humor_retrieval_cases(query, sender_name, chat_type)
                    if cases:
                        dynamic_few_shot = cases
            except Exception as e:
                _logger.warning("[HumorRAG] retrieval failed: %s", e)
        prompt = prompt.replace("{dynamic_few_shot}", dynamic_few_shot)

        # 保留 skill hint
        skill_hint = self._load_skill_one_liners()
        if skill_hint:
            prompt += "\n\n" + skill_hint.strip()

        return prompt

    def _build_tools_context(self, chat_name: str) -> str:
        """构建工具上下文：已缓存数据 + 工具结果提示。"""
        cache_lines = self.session_memory.get_cache_lines(chat_name, include_expired=True)
        if not cache_lines:
            return ""
        lines_local = ["已缓存数据（来自之前查询，无需重复调用）"]
        lines_local.extend(cache_lines)
        lines_local.append("")
        return "\n".join(lines_local)

    @staticmethod
    def _format_time_tag(ts: Optional[int], now_ts: float) -> str:
        """根据时间戳生成绝对时间标签 YYYY-MM-DD HH:MM。"""
        if not ts:
            return ""
        tm = time.localtime(int(ts))
        return f"{tm.tm_year:04d}-{tm.tm_mon:02d}-{tm.tm_mday:02d} {tm.tm_hour:02d}:{tm.tm_min:02d}"

    @staticmethod
    def _format_message_line(m: ChatMessage, enable_timestamps: bool = True) -> str:
        """将单条消息渲染为 prompt 中的一行文本，含时间戳。"""
        sender_name = "我" if m.sender_type == SenderType.SELF else m.sender
        msg_type = m.message_type or "text"
        now_ts = time.time()

        # 时间标签（优先 create_time int，fallback timestamp str）
        time_tag = ""
        if enable_timestamps:
            ts = getattr(m, 'create_time', None)
            if not ts:
                ts_str = getattr(m, 'timestamp', None)
                if ts_str:
                    try:
                        from datetime import datetime
                        ts = int(datetime.strptime(ts_str.replace('  ', ' '), "%Y-%m-%d %H:%M:%S").timestamp())
                    except Exception as e:
                        _logger.warning("[Generator] 时间戳解析失败: %s (raw=%r)", e, ts_str)
            time_tag = ReplyGenerator._format_time_tag(ts, now_ts) if ts else ""

        def _line(body: str) -> str:
            if time_tag:
                return f"{sender_name}（{time_tag}）：{body}"
            return f"{sender_name}：{body}"

        if msg_type == "image":
            desc = m.image_description or "图片"
            text_part = m.image_text or m.text or ""
            if text_part:
                return _line(f"[图片] {desc}（图上文字：{text_part}）")
            return _line(f"[图片] {desc}")

        elif msg_type == "sticker":
            desc = m.image_description or "表情包"
            text_part = m.image_text or m.text or ""
            if text_part:
                return _line(f"[表情包] {desc}（配字：{text_part}）")
            return _line(f"[表情包] {desc}")

        elif msg_type == "mixed":
            desc = m.image_description or ""
            text_part = m.text or ""
            if desc:
                return _line(f"[图片+文字] {text_part} | 图片描述：{desc}")
            return _line(f"[图片+文字] {text_part}")

        elif msg_type == "link_card":
            desc = m.image_description or "链接卡片"
            return _line(f"[链接卡片] {desc}")

        elif msg_type == "video":
            desc = m.image_description or "视频"
            text_part = m.image_text or m.text or ""
            if text_part:
                return _line(f"[视频] {desc}（视频文字：{text_part}）")
            return _line(f"[视频] {desc}")

        else:
            body = m.text or ""
            if m.quoted_text:
                body = f"「引用：{m.quoted_text}」{body}"
            return _line(body)

    @staticmethod
    def _build_mode_hints(
        unreplied: List[ChatMessage],
        all_messages: List[ChatMessage],
        is_group: bool,
        hour: int,
    ) -> str:
        """基于当前消息和历史上下文，生成帮助 LLM 过五关的语境速查提示。"""
        hints: List[str] = []

        # 1. 读场子：从当前未读消息提取意图和情绪
        last_texts = " ".join(
            (m.text or m.image_text or "") for m in unreplied
        )

        celebration_signals = [
            "升职", "晋升", "涨薪", "加薪", "获奖", "通过了", "恭喜", "🎉",
            "喜提", "买房", "买车", "结婚", "订婚", "生娃", "上岸", "拿到",
            "offer", "签了", "中奖", "发财", "暴富", "脱单",
        ]
        complaint_signals = [
            "烦", "累", "倒霉", "亏了", "无语", "难受", "裂开", "崩溃",
            "什么鬼", "服了", "妈的", "烦死", "不想干了",
        ]
        question_signals = [
            "怎么办", "怎么", "吗", "呢", "请问", "谁知道", "推荐",
            "求", "有没有", "能不能", "为什么", "是啥", "是什么",
        ]
        confirmation_signals = [
            "确认", "收到", "麻烦", "核对", "登记", "预约", "扣款",
            "到账", "办理", "合同", "发票",
        ]

        if any(s in last_texts for s in celebration_signals):
            hints.append("【场子】对方刚分享好消息/喜事，优先考虑祝贺模式")
        elif any(s in last_texts for s in complaint_signals):
            hints.append("【场子】对方情绪负面，可能在吐槽/受挫，优先考虑安慰模式")
        elif any(s in last_texts for s in question_signals):
            hints.append("【场子】对方在提问/求助，优先考虑解答模式")
        elif any(s in last_texts for s in confirmation_signals):
            hints.append("【场子】对方在确认事务，优先考虑事务模式")
        elif unreplied and all(
            (m.message_type or "text") in MEDIA_MESSAGE_TYPES
            for m in unreplied
        ):
            hints.append("【场子】对方发了图片/表情包，先看图内容再决定接不接梗")

        # 1.5 记忆查询检测：对方在问 Bot 身边的人/旧事
        memory_query_signals = [
            "认识", "记得", "初中", "高中", "大学", "同学", "同事",
            "朋友", "亲戚", "表哥", "表姐", "表弟", "表妹", "堂哥",
            "堂姐", "堂弟", "堂妹", "舅舅", "姨妈", "姑姑", "叔叔",
            "伯伯", "爷爷", "奶奶", "外公", "外婆", "爸爸", "妈妈",
            "老公", "老婆", "媳妇", "男朋友", "女朋友", "兄弟", "姐妹",
            "谁", "哪人", "哪里的", "干什么的", "做什么工作",
        ]
        # 检测是否提到具体人名（2-4个字的常见中文名模式）
        import re
        # 简单人名检测：连续2-4个中文字符，且不在常见虚词列表中
        common_words = {"什么", "怎么", "为什么", "是不是", "有没有", "可以", "这个", "那个", "他们", "我们", "你们", "自己", "别人", "人家", "大家", "一起", "现在", "今天", "明天", "昨天", "上次", "下次", "一直", "总是", "真的", "假的", "当然", "可能", "应该", "大概", "好像", "似乎", "确实", "其实", "本来", "原来", "刚刚", "马上", "立刻", "终于", "已经", "曾经", "正在", "将要", "打算", "准备", "开始", "结束", "完成", "成功", "失败", "不错", "很好", "太差", "一般", "还行", "可以", "不行", "不能", "不会", "不要", "不用", "没空", "有事", "没事", "好的", "收到", "了解", "知道", "明白", "清楚", "确定", "确认", "同意", "反对", "支持", "拒绝", "接受", "放弃", "坚持", "努力", "加油", "注意", "小心", "当心", "放心", "担心", "害怕", "紧张", "开心", "高兴", "难过", "伤心", "生气", "愤怒", "惊讶", "意外", "失望", "满意", "舒服", "难受", "累", "困", "饿", "渴", "冷", "热", "疼", "痛", "痒", "酸", "麻", "胀", "晕", "傻", "笨", "聪明", "漂亮", "帅", "丑", "胖", "瘦", "高", "矮", "大", "小", "长", "短", "宽", "窄", "厚", "薄", "深", "浅", "远", "近", "快", "慢", "早", "晚", "新", "旧", "老", "年轻", "贵", "便宜", "多", "少", "满", "空", "干", "湿", "脏", "干净", "整齐", "乱", "安静", "吵", "亮", "暗", "黑", "白", "红", "黄", "蓝", "绿", "紫", "橙", "灰", "粉", "棕", "金", "银", "透明", "彩色", "单色", "漂亮", "美丽", "好看", "可爱", "帅", "酷", "棒", "厉害", "牛", "强", "弱", "好", "坏", "对", "错", "真", "假", "正", "反", "上", "下", "左", "右", "前", "后", "里", "外", "内", "中", "间", "边", "角", "头", "尾", "根", "底", "顶", "面", "背", "侧", "正", "反", "东", "西", "南", "北", "方向", "位置", "地方", "地区", "城市", "国家", "世界", "地球", "宇宙", "天空", "太阳", "月亮", "星星", "云", "风", "雨", "雪", "雷", "电", "雾", "霜", "露", "冰", "水", "火", "土", "木", "金", "石头", "沙子", "泥土", "灰尘", "空气", "氧气", "氢气", "氮气", "二氧化碳", "一氧化碳", "甲烷", "乙醇", "酒精", "汽油", "柴油", "煤油", "天然气", "石油", "煤炭", "矿石", "金属", "钢铁", "铜", "铁", "铝", "锌", "铅", "锡", "镍", "铬", "钛", "钨", "银", "金", "铂", "钻石", "宝石", "玉石", "珍珠", "玛瑙", "水晶", "玻璃", "塑料", "橡胶", "皮革", "木头", "竹子", "纸张", "布料", "丝绸", "棉花", "羊毛", "麻", "化纤", "尼龙", "涤纶", "腈纶", "维纶", "丙纶", "氯纶", "氨纶", "碳纤维", "玻璃纤维", "陶瓷", "瓷器", "陶器", "砖瓦", "水泥", "混凝土", "沥青", "油漆", "涂料", "染料", "颜料", "墨水", "油墨", "胶水", "胶带", "胶布", "橡皮", "橡皮筋", "绳子", "线", "铁丝", "钢丝", "铜丝", "铝丝", "钉子", "螺丝", "螺母", "螺栓", "垫圈", "弹簧", "轴承", "齿轮", "皮带", "链条", "轮子", "轮胎", "轮毂", "刹车", "油门", "离合器", "变速器", "发动机", "电动机", "发电机", "变压器", "电容器", "电阻器", "电感器", "二极管", "三极管", "集成电路", "芯片", "CPU", "GPU", "内存", "硬盘", "固态硬盘", "U盘", "光盘", "软盘", "磁带", "数据线", "充电线", "电源线", "网线", "光纤", "电缆", "电线", "电池", "电瓶", "充电宝", "充电器", "插头", "插座", "开关", "灯泡", "灯管", "灯带", "LED", "LCD", "OLED", "屏幕", "显示器", "电视", "投影仪", "摄像机", "相机", "手机", "电话", "电脑", "笔记本", "平板", "手表", "手环", "耳机", "音箱", "麦克风", "键盘", "鼠标", "触控板", "遥控器", "路由器", "交换机", "防火墙", "服务器", "主机", "客户端", "浏览器", "软件", "硬件", "系统", "程序", "代码", "脚本", "算法", "模型", "数据", "数据库", "表", "字段", "记录", "文件", "文件夹", "目录", "路径", "链接", "URL", "域名", "IP", "端口", "协议", "接口", "API", "SDK", "框架", "库", "包", "模块", "组件", "插件", "扩展", "主题", "模板", "样式", "布局", "设计", "架构", "结构", "模式", "规范", "标准", "文档", "说明", "注释", "日志", "报告", "图表", "图像", "照片", "视频", "音频", "音乐", "歌曲", "歌词", "乐器", "钢琴", "吉他", "小提琴", "大提琴", "二胡", "笛子", "箫", "唢呐", "琵琶", "古筝", "古琴", "鼓", "锣", "钹", "钟", "铃", "喇叭", "号角", "口琴", "手风琴", "电子琴", "合成器", "音箱", "功放", "调音台", "麦克风", "话筒", "耳机", "耳塞", "耳麦", "音响", "家庭影院", "KTV", "录音棚", "直播间", "舞台", "剧院", "电影院", "音乐厅", "美术馆", "博物馆", "图书馆", "书店", "学校", "大学", "中学", "小学", "幼儿园", "培训机构", "公司", "企业", "工厂", "车间", "仓库", "办公室", "会议室", "前台", "后台", "领导", "老板", "经理", "主管", "总监", "副总裁", "总裁", "CEO", "CTO", "CFO", "COO", "CMO", "CHO", "CIO", "CDO", "CSO", "CCO", "创始人", "合伙人", "股东", "董事", "监事", "员工", "同事", "下属", "上级", "客户", "用户", "顾客", "乘客", "游客", "观众", "读者", "听众", "粉丝", "关注者", "订阅者", "会员", "VIP", "嘉宾", "评委", "选手", "运动员", "教练", "裁判", "裁判", "解说", "记者", "编辑", "作者", "作家", "诗人", "画家", "音乐家", "演员", "导演", "制片人", "编剧", "摄影师", "设计师", "工程师", "科学家", "教授", "老师", "学生", "同学", "校友", "室友", "同桌", "班长", "团支书", "学习委员", "文艺委员", "体育委员", "生活委员", "劳动委员", "纪律委员", "组织委员", "宣传委员", "心理委员", "安全委员", "课代表", "组长", "队长", "社长", "会长", "主席", "部长", "干事", "志愿者", "义工", "慈善家", "捐赠者", "受助者", "患者", "医生", "护士", "药师", "技师", "护工", "保洁", "保安", "司机", "厨师", "服务员", "接待员", "销售员", "导购", "收银员", "会计", "出纳", "审计", "税务", "律师", "法官", "检察官", "警察", "军人", "消防员", "救护员", "救生员", "导游", "翻译", "秘书", "助理", "顾问", "咨询师", "分析师", "研究员", "调查员", "检测员", "质检员", "操作员", "管理员", "调度员", "快递员", "外卖员", "配送员", "搬运工", "装修工", "电工", "水工", "木工", "瓦工", "油漆工", "焊工", "车工", "铣工", "钳工", "铸工", "锻工", "模具工", "数控工", "编程员", "操作手", "驾驶员", "飞行员", "船长", "船员", "列车员", "乘务员", "空乘", "地勤", "塔台", "调度", "指挥", "控制", "监控", "巡检", "维护", "维修", "保养", "安装", "调试", "测试", "检验", "校准", "标定", "认证", "许可", "注册", "备案", "审批", "审核", "检查", "稽查", "督查", "巡视", "巡查", "巡逻", "守望", "守卫", "保卫", "守护", "监护", "看管", "看守", "监禁", "拘留", "逮捕", "关押", "服刑", "改造", "教育", "感化", "挽救", "救助", "救援", "急救", "抢救", "治疗", "康复", "护理", "照料", "照顾", "看护", "抚养", "养育", "培养", "教育", "教导", "指导", "辅导", "培训", "训练", "锻炼", "练习", "实习", "实践", "实验", "试验", "尝试", "探索", "研究", "钻研", "攻克", "突破", "创新", "发明", "发现", "发掘", "开发", "开拓", "拓展", "扩展", "扩张", "推广", "宣传", "传播", "流传", "传承", "继承", "发扬", "光大", "振兴", "复兴", "崛起", "腾飞", "飞跃", "跨越", "超越", "突破", "打破", "粉碎", "摧毁", "毁灭", "消灭", "消除", "清除", "清理", "整理", "整顿", "整治", "治理", "管理", "管制", "控制", "限制", "约束", "束缚", "捆绑", "束缚", "压迫", "压制", "镇压", "制止", "阻止", "阻挡", "阻拦", "拦截", "截获", "缴获", "没收", "征收", "征税", "收费", "罚款", "处罚", "惩罚", "制裁", "报复", "复仇", "反击", "反攻", "反抗", "抵抗", "抵御", "防御", "防守", "保卫", "保护", "维护", "维持", "保持", "保留", "保存", "储存", "存储", "收藏", "珍藏", "保管", "看守", "看护", "照料", "照顾", "伺候", "侍奉", "服侍", "服务", "效劳", "效力", "效忠", "忠诚", "忠心", "虔诚", "信仰", "信念", "信心", "信任", "信赖", "依靠", "依赖", "依附", "附着", "粘贴", "贴合", "紧靠", "靠近", "接近", "接触", "碰触", "碰撞", "撞击", "冲击", "打击", "攻击", "进攻", "侵犯", "侵略", "侵占", "占领", "占据", "占有", "拥有", "具有", "具备", "配备", "配置", "安置", "安排", "布置", "部署", "分布", "散布", "散播", "传播", "传染", "感染", "影响", "感动", "打动", "触动", "震撼", "震惊", "惊讶", "惊喜", "欢喜", "喜欢", "喜爱", "爱好", "热衷", "痴迷", "迷恋", "迷恋", "贪恋", "贪图", "图谋", "谋划", "策划", "筹划", "筹备", "准备", "预备", "预先", "提前", "趁早", "及时", "准时", "按时", "按期", "定期", "长期", "短期", "暂时", "临时", "永久", "永远", "永恒", "瞬间", "刹那", "片刻", "一会儿", "不久", "很快", "缓慢", "慢慢", "渐渐", "逐渐", "逐步", "逐步", "陆续", "连续", "持续", "不断", "不停", "不止", "不息", "不休", "不眠", "不休", "不懈", "不怠", "勤奋", "勤劳", "勤勉", "刻苦", "努力", "奋斗", "拼搏", "拼搏", "拼命", "拼死", "誓死", "决死", "决意", "决心", "决计", "决定", "决心", "决意", "决计", "决死", "拼死", "拼命", "拼搏", "奋斗", "努力", "刻苦", "勤奋", "勤劳", "勤勉", "不懈", "不怠", "不休", "不息", "不止", "不停", "不断", "持续", "连续", "陆续", "逐步", "逐渐", "渐渐", "慢慢", "缓慢", "很快", "不久", "一会儿", "片刻", "刹那", "瞬间", "永恒", "永远", "永久", "临时", "暂时", "短期", "长期", "定期", "按期", "按时", "准时", "及时", "趁早", "提前", "预先", "预备", "准备", "筹备", "筹划", "策划", "谋划", "图谋", "贪图", "贪恋", "迷恋", "痴迷", "热衷", "爱好", "喜爱", "喜欢", "欢喜", "惊喜", "惊讶", "震惊", "震撼", "触动", "打动", "感动", "影响", "感染", "传染", "传播", "散播", "散布", "分布", "部署", "布置", "安排", "安置", "配置", "配备", "具备", "具有", "拥有", "占有", "占据", "占领", "侵占", "侵略", "侵犯", "进攻", "攻击", "打击", "冲击", "撞击", "碰撞", "碰触", "接触", "接近", "靠近", "紧靠", "贴合", "粘贴", "附着", "依附", "依赖", "依靠", "信赖", "信任", "信心", "信念", "信仰", "虔诚", "忠心", "忠诚", "效忠", "效力", "效劳", "服务", "服侍", "侍奉", "伺候", "照顾", "照料", "看护", "看守", "保管", "珍藏", "收藏", "存储", "储存", "保存", "保留", "保持", "维持", "维护", "保护", "保卫", "防守", "防御", "抵御", "抵抗", "反抗", "反攻", "反击", "复仇", "报复", "制裁", "惩罚", "处罚", "罚款", "收费", "征税", "征收", "没收", "缴获", "截获", "拦截", "阻拦", "阻挡", "阻止", "制止", "镇压", "压制", "压迫", "束缚", "捆绑", "约束", "限制", "控制", "管制", "管理", "治理", "整治", "整顿", "整理", "清理", "清除", "消除", "消灭", "毁灭", "摧毁", "粉碎", "打破", "突破", "超越", "跨越", "飞跃", "腾飞", "崛起", "复兴", "振兴", "光大", "发扬", "继承", "传承", "流传", "传播", "宣传", "推广", "扩张", "扩展", "拓展", "开拓", "开发", "发掘", "发现", "发明", "创新", "突破", "攻克", "钻研", "研究", "探索", "尝试", "试验", "实验", "实践", "实习", "练习", "锻炼", "训练", "培训", "辅导", "指导", "教导", "教育", "培养", "养育", "抚养", "看护", "照料", "护理", "康复", "治疗", "抢救", "急救", "救援", "救助", "挽救", "感化", "教育", "改造", "服刑", "关押", "逮捕", "拘留", "监禁", "看守", "看管", "监护", "守护", "保卫", "守卫", "守望", "巡逻", "巡查", "巡视", "督查", "稽查", "检查", "审核", "审批", "备案", "注册", "许可", "认证", "标定", "校准", "检验", "测试", "调试", "安装", "保养", "维修", "维护", "巡检", "监控", "控制", "指挥", "调度", "塔台", "地勤", "空乘", "乘务员", "列车员", "船员", "船长", "飞行员", "驾驶员", "操作手", "编程员", "数控工", "模具工", "锻工", "铸工", "钳工", "铣工", "车工", "焊工", "油漆工", "瓦工", "木工", "水工", "电工", "装修工", "搬运工", "配送员", "外卖员", "快递员", "调度员", "管理员", "操作员", "检测员", "质检员", "调查员", "研究员", "分析师", "咨询师", "顾问", "助理", "秘书", "翻译", "导游", "救生员", "救护员", "消防员", "军人", "警察", "检察官", "法官", "律师", "税务", "审计", "出纳", "会计", "收银员", "导购", "销售员", "接待员", "服务员", "厨师", "司机", "保安", "保洁", "护工", "技师", "药师", "护士", "医生", "患者", "受助者", "捐赠者", "慈善家", "义工", "志愿者", "干事", "部长", "主席", "会长", "社长", "队长", "组长", "课代表", "安全委员", "心理委员", "宣传委员", "组织委员", "纪律委员", "劳动委员", "生活委员", "体育委员", "文艺委员", "学习委员", "团支书", "班长", "同桌", "室友", "校友", "同学", "学生", "老师", "教授", "科学家", "工程师", "设计师", "摄影师", "编剧", "制片人", "导演", "演员", "音乐家", "画家", "诗人", "作家", "作者", "编辑", "记者", "解说", "裁判", "教练", "运动员", "选手", "评委", "嘉宾", "VIP", "会员", "订阅者", "关注者", "粉丝", "听众", "读者", "观众", "游客", "乘客", "顾客", "用户", "客户", "下属", "上级", "同事", "员工", "监事", "董事", "股东", "合伙人", "创始人", "CCO", "CSO", "CDO", "CIO", "CHO", "CMO", "COO", "CFO", "CTO", "CEO", "总裁", "副总裁", "总监", "主管", "经理", "老板", "领导", "后台", "前台", "会议室", "办公室", "仓库", "车间", "工厂", "企业", "公司", "培训机构", "幼儿园", "小学", "中学", "大学", "学校", "书店", "图书馆", "博物馆", "美术馆", "音乐厅", "电影院", "剧院", "舞台", "直播间", "录音棚", "KTV", "家庭影院", "音响", "耳麦", "耳塞", "耳机", "话筒", "麦克风", "调音台", "功放", "音箱", "合成器", "电子琴", "手风琴", "口琴", "号角", "喇叭", "铃", "钟", "钹", "锣", "鼓", "古琴", "古筝", "琵琶", "唢呐", "箫", "笛子", "二胡", "大提琴", "小提琴", "吉他", "钢琴", "乐器", "歌词", "歌曲", "音乐", "音频", "视频", "照片", "图像", "图表", "报告", "日志", "注释", "说明", "文档", "标准", "规范", "模式", "结构", "架构", "设计", "布局", "样式", "模板", "主题", "扩展", "插件", "组件", "模块", "包", "库", "框架", "SDK", "API", "接口", "协议", "端口", "IP", "域名", "URL", "链接", "路径", "目录", "文件夹", "文件", "记录", "字段", "表", "数据库", "数据", "模型", "算法", "脚本", "代码", "程序", "系统", "硬件", "软件", "浏览器", "客户端", "主机", "服务器", "防火墙", "交换机", "路由器", "遥控器", "触控板", "鼠标", "键盘", "麦克风", "音箱", "耳机", "手表", "手环", "平板", "笔记本", "电脑", "电话", "手机", "相机", "摄像机", "投影仪", "电视", "显示器", "屏幕", "OLED", "LCD", "LED", "灯带", "灯管", "灯泡", "开关", "插座", "插头", "充电器", "充电宝", "电瓶", "电池", "电线", "电缆", "光纤", "网线", "电源线", "充电线", "数据线", "磁带", "软盘", "光盘", "U盘", "固态硬盘", "硬盘", "内存", "GPU", "CPU", "芯片", "集成电路", "三极管", "二极管", "电感器", "电阻器", "电容器", "变压器", "发电机", "电动机", "发动机", "变速器", "离合器", "油门", "刹车", "轮毂", "轮胎", "轮子", "链条", "皮带", "齿轮", "轴承", "弹簧", "垫圈", "螺栓", "螺母", "螺丝", "钉子", "铝丝", "铜丝", "钢丝", "铁丝", "线", "绳子", "橡皮筋", "橡皮", "胶布", "胶带", "胶水", "油墨", "墨水", "颜料", "染料", "涂料", "油漆", "沥青", "混凝土", "水泥", "砖瓦", "陶器", "瓷器", "陶瓷", "氨纶", "氯纶", "丙纶", "维纶", "腈纶", "涤纶", "尼龙", "化纤", "麻", "羊毛", "棉花", "丝绸", "布料", "纸张", "竹子", "木头", "皮革", "橡胶", "塑料", "玻璃", "水晶", "玛瑙", "珍珠", "玉石", "宝石", "钻石", "铂", "金", "银", "钨", "钛", "铬", "镍", "锡", "铅", "锌", "铝", "铁", "铜", "钢铁", "金属", "矿石", "煤炭", "石油", "天然气", "煤油", "柴油", "汽油", "乙醇", "酒精", "甲烷", "一氧化碳", "二氧化碳", "氮气", "氢气", "氧气", "空气", "灰尘", "泥土", "沙子", "石头", "木", "土", "火", "水", "冰", "露", "霜", "雾", "电", "雷", "雪", "雨", "风", "云", "星星", "月亮", "太阳", "天空", "宇宙", "地球", "世界", "国家", "城市", "地区", "地方", "位置", "方向", "北", "南", "西", "东", "反", "正", "侧", "背", "面", "顶", "底", "根", "尾", "头", "角", "边", "间", "中", "内", "外", "里", "后", "前", "右", "左", "下", "上", "错", "对", "坏", "好", "弱", "强", "棒", "酷", "帅", "可爱", "好看", "美丽", "漂亮", "彩色", "单色", "透明", "银", "金", "棕", "粉", "灰", "橙", "紫", "绿", "蓝", "黄", "红", "白", "黑", "暗", "亮", "吵", "安静", "乱", "整齐", "干净", "脏", "湿", "干", "空", "满", "少", "多", "便宜", "贵", "年轻", "老", "旧", "新", "晚", "早", "快", "慢", "近", "远", "浅", "深", "薄", "厚", "窄", "宽", "短", "长", "小", "大", "矮", "高", "瘦", "胖", "丑", "漂亮"}
        # 提取可能的人名
        potential_names = []
        for match in re.finditer(r'[\u4e00-\u9fff]{2,4}', last_texts):
            name = match.group()
            if name not in common_words and len(name) >= 2:
                potential_names.append(name)

        if potential_names and any(s in last_texts for s in memory_query_signals):
            names_str = "、".join(potential_names[:3])
            hints.append(f"【场子】对方在问身边的人/旧事（提到：{names_str}），如果不确定，必须调用 search_memory 查记忆，不要猜测不要编造")

        # 2. 读气氛：看最近 10 轮非我发的消息
        recent_others = [
            m for m in all_messages[-10:]
            if m.sender_type != SenderType.SELF
        ]
        recent_texts = [
            m.text for m in recent_others
            if m.text and len(m.text.strip()) > 1
        ]

        if recent_texts:
            avg_len = sum(len(t) for t in recent_texts) / len(recent_texts)
            if avg_len < 8:
                hints.append("【气氛】近期大家回复很短，适合短句/接梗")
            elif avg_len > 40:
                hints.append("【气氛】近期在正经长聊，回复可以适当展开")

            banter_signals = ["哈哈", "笑死", "牛逼", "废物", "妈的", "绝了", "艹"]
            banter_count = sum(
                1 for t in recent_texts
                if any(s in t for s in banter_signals)
            )
            if banter_count / len(recent_texts) >= 0.3:
                hints.append("【气氛】近期基调轻松调侃，大家都在互损接梗")

            late_night_signals = ["睡不着", "emo", "烦", "想", "回忆", "曾经"]
            if 22 <= hour <= 24 or 0 <= hour <= 3:
                if any(s in " ".join(recent_texts) for s in late_night_signals):
                    hints.append("【气氛】深夜感性聊天，避免理性分析和说教")

        # 3. 读场域：时间、群/私
        if is_group:
            hints.append("【场域】这是群聊，注意分寸，不要乱接话")
        else:
            hints.append("【场域】这是私聊，可以更个人化")

        if 22 <= hour or hour <= 6:
            hints.append("【时间】深夜时段，避免事务性回复和长文分析")
        elif 9 <= hour <= 18:
            hints.append("【时间】工作时间，回复可以简洁直接")

        return "\n".join(f"- {h}" for h in hints)

    def _build_user_prompt(self, unreplied: List[ChatMessage], all_messages: List[ChatMessage],
                           is_group: bool = False,
                           enable_time_awareness: Optional[bool] = None,
                           enable_unread_dedup: Optional[bool] = None,
                           enable_timestamps: Optional[bool] = None,
                           tools_context: str = "") -> str:
        """构建结构化 user prompt：会话信息 + 记忆 + 缓存 + 历史 + 未读。"""
        if enable_time_awareness is None:
            enable_time_awareness = self.enable_time_awareness
        if enable_unread_dedup is None:
            enable_unread_dedup = self.enable_unread_dedup
        if enable_timestamps is None:
            enable_timestamps = self.enable_timestamps

        from datetime import datetime
        chat_name = unreplied[-1].chat_name if unreplied else ""
        lines_local = []

        # 会话信息（含时间上下文）
        is_at = any(getattr(m, "is_at_me", False) for m in unreplied)
        chat_type = "群聊" if is_group else "私聊"
        now_dt = datetime.now()
        now = now_dt.strftime("%Y年%m月%d日 %H:%M")
        weekday = ["周一","周二","周三","周四","周五","周六","周日"][now_dt.weekday()]
        hour = now_dt.hour
        time_period = "凌晨" if hour < 6 else ("早上" if hour < 9 else ("上午" if hour < 12 else ("下午" if hour < 18 else ("晚上" if hour < 22 else "深夜"))))
        lines_local.append("[会话]")
        if enable_time_awareness:
            lines_local.append(f"当前时间：{now} {weekday} {time_period}")
        lines_local.append(f"聊天：{chat_name}")
        lines_local.append(f"类型：{chat_type}")
        lines_local.append(f"被@：{'是' if is_at else '否'}")
        lines_local.append("")
        if enable_time_awareness:
            lines_local.append("⚠️ 消息时间戳说明：每条消息后面标注的时间是消息发出的绝对时间（格式：YYYY-MM-DD HH:MM）。请根据时间戳推断语境，不要假设消息是刚刚发的。")
            lines_local.append("")

        # 语境速查（仅开启模式检测时注入，帮助 LLM 过五关）
        if self.enable_mode_detection and unreplied:
            mode_hints = self._build_mode_hints(unreplied, all_messages, is_group, hour)
            if mode_hints:
                lines_local.append("[语境速查]（帮你过五关，参考即可，不要原样复述）")
                lines_local.append(mode_hints)
                lines_local.append("")

        # wiki 记忆
        t_mem_ms = {}
        if self.memory_engine is not None and unreplied:
            # 固定注入 Bot 自己的 wiki，避免 LLM 被对方的 wiki 淹没后混淆身份
            t_m1 = time.time()
            self_memory = self.memory_engine.get_user_memory("王芊", max_chars=4000)
            t_mem_ms["self"] = (time.time() - t_m1) * 1000
            if self_memory:
                lines_local.append("[我的信息]（来自长期记忆，Bot 自己的身份背景）")
                lines_local.append(self_memory)
                lines_local.append("")

            # 对方信息：仅私聊加载，群聊里最后一条发送者不能代表"对方"
            if not is_group:
                last_sender = unreplied[-1].sender
                clean_sender = last_sender.split(" @")[0] if last_sender and " @" in last_sender else last_sender
                if clean_sender and clean_sender != "我":
                    t_m2 = time.time()
                    memory_text = self.memory_engine.get_user_memory(clean_sender, max_chars=6000)
                    t_mem_ms["other"] = (time.time() - t_m2) * 1000
                    if memory_text:
                        lines_local.append("[对方信息]（来自长期记忆，仅为该用户记忆的部分摘要）")
                        lines_local.append(memory_text)
                        lines_local.append("")
            if is_group and chat_name:
                t_m3 = time.time()
                group_text = self.memory_engine.get_group_memory(chat_name, max_chars=6000)
                t_mem_ms["group"] = (time.time() - t_m3) * 1000
                if group_text:
                    lines_local.append("[本群信息]（来自长期记忆）")
                    lines_local.append(group_text)
                    lines_local.append("")

            mem_summary = " ".join(f"{k}={v:.0f}ms" for k, v in t_mem_ms.items())
            print(f"[Perf][Memory] {mem_summary}")

        # 已缓存数据（session_memory 中的工具结果）
        if tools_context:
            lines_local.append(tools_context)
            lines_local.append("")

        # 历史消息：最近50条 或 最近30分钟内，取并集
        if all_messages:
            lines_local.append("[历史消息]（仅背景参考，按时间倒序）")

            def _msg_ts(m: ChatMessage) -> float:
                if m.sender_type == SenderType.SELF and m.reply_time:
                    return m.reply_time
                if m.create_time:
                    return float(m.create_time)
                return time.time()

            now_ts = time.time()
            cutoff_ts = now_ts - 1800  # 30分钟

            recent_50 = list(all_messages[-50:]) if len(all_messages) > 50 else list(all_messages)
            recent_30min = [m for m in all_messages if _msg_ts(m) >= cutoff_ts]

            union_ids = {id(m) for m in recent_50} | {id(m) for m in recent_30min}

            # 兜底：强制保留最近 5 条 bot 自己发的消息，防止被对方密集消息淹没
            self_msgs = [m for m in all_messages if m.sender_type == SenderType.SELF]
            for m in self_msgs[-5:]:
                union_ids.add(id(m))
            candidate = [m for m in all_messages if id(m) in union_ids]

            max_history = 80
            recent = list(candidate[-max_history:]) if len(candidate) > max_history else list(candidate)
            # 保持正序：旧的消息在前，新的消息在后，符合阅读习惯

            for m in recent:
                lines_local.append(self._format_message_line(m, enable_timestamps))

            if len(all_messages) > len(recent):
                lines_local.append(f"（共 {len(all_messages)} 条历史，显示 {len(recent)} 条：最近50条 + 30分钟内）")
            lines_local.append("")

        # 未读消息（带去重检查：如果历史中已有相似消息且 Bot 已回复，标记为'可能已处理'）
        lines_local.append("[未读消息]（重点回复）")
        # 从历史中提取 Bot 已回复的消息文本（用于去重判断）
        # 简化：如果[未读消息]中的某条在[历史消息]中能找到 Bot 的回复且 Bot 回复在未读消息时间之前
            # 则标记为"可能已回复"

        skipped_hint = []
        for i, m in enumerate(unreplied, 1):
            ts = getattr(m, 'create_time', None)
            # 检查历史中是否有 Bot 在未读消息时间之后回复的
            already_handled = False
            if ts and all_messages:
                for hm in all_messages:
                    if hm.sender_type == SenderType.SELF and hm.reply_time and hm.reply_time > ts:
                        # Bot 在未读消息之后回复了，说明这条可能已经处理过
                        already_handled = True
                        break
            tag = " ⚠️(历史中已有回复，可跳过)" if (enable_unread_dedup and already_handled) else ""
            lines_local.append(f"{i}. {self._format_message_line(m, enable_timestamps)}{tag}")
            if enable_unread_dedup and already_handled:
                skipped_hint.append(str(i))
        lines_local.append("")

        if enable_unread_dedup and skipped_hint:
            lines_local.append(f"提示：第{','.join(skipped_hint)}条未读消息在历史中已有回复，可能不需要再次回复。仅回复真正未处理的新消息。")
        lines_local.append("回复重点：仅回复真正需要回应的未读消息。纯表情/OK/好的等确认性消息可以不回复。")

        return "\n".join(lines_local)
