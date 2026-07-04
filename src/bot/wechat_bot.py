#!/usr/bin/env python3
"""L5 Bot Orchestrator - 主循环编排"""

import logging as _logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

from src.action.chat_list_clicker import ChatListClicker
from src.action.login_recovery import LoginRecoveryStatus, WeChatLoginHandler
from src.action.message_sender import WeChatMessageSender
from src.capture.window_capture import WeChatNotReadyError
from src.logging.bot_logger import BotLogger, get_logger
from src.memory import MemoryEngine
from src.models.base import ActionResult, ChatMessage, PerceptionResult, SenderType
from src.perception.vision_pipeline import VisionPipeline
from src.reply.generator import ReplyGenerator, _get_judge_worker
from src.reply.policy import ReplyPolicy
from src.session.global_store import GlobalStore
from src.tools import get_registry, register_builtin_tools
from src.utils.chat_utils import _is_group_chat_name, _normalize_chat_name
from src.utils.debug_logger import DebugLogger


def _raw_with_thinking(raw_response: str, thinking: str) -> str:
    """将思考过程合并到 raw_response 前，前端可以直接展示。"""
    if not thinking:
        return raw_response
    return f"[思考过程]\n{thinking}\n\n[回复]\n{raw_response}"


def _try_create_openclaw_client():
    """尝试创建 OpenClaw 客户端，失败时返回 None（退化为单模型模式）"""
    try:
        from src.llm.openclaw_client import OpenClawClient
        return OpenClawClient.from_openclaw_config()
    except Exception as e:
        _logging.warning("[bot] OpenClaw 客户端创建失败，退化为单模型模式: %s", e)
        return None


class WeChatBot:
    def __init__(self, profile=None, on_message: Optional[Callable] = None, llm_client=None,
                 complex_llm_client=None, debug_mode: bool = False, use_openclaw: bool = True, perception=None,
                 enable_chat_switch: bool = True):
        # 先初始化 logger，后续各步骤都可能需要记录日志
        self.logger: BotLogger = get_logger()
        if perception is not None:
            self.perception = perception
        else:
            self.perception = VisionPipeline(profile)
        self.global_store = GlobalStore()
        self.policy = ReplyPolicy(require_at_in_group=False)

        if llm_client is not None:
            actual_llm = llm_client
        elif use_openclaw:
            actual_llm = _try_create_openclaw_client()
        else:
            actual_llm = None

        # 启动时自动同步 knowledge_source.md → JSON / wiki
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from scripts.sync_knowledge import sync
            if sync():
                print("[knowledge] 已自动同步 knowledge_source.md")
        except Exception as e:
            print(f"[knowledge] 同步失败: {e}")

        # 先创建记忆引擎（ReplyGenerator 初始化时需要）
        self.memory_engine: MemoryEngine = MemoryEngine(llm_client=actual_llm)
        # 注册工具到全局注册表
        registry = get_registry()
        register_builtin_tools(registry)
        # 再创建 Generator，把 memory_engine 和 tool_registry 直接传入
        # JudgeWorker 默认关闭（deepseek token 消耗大），通过 ENABLE_JUDGE_WORKER=1 开启
        judge_worker = _get_judge_worker() if os.environ.get("ENABLE_JUDGE_WORKER", "0") == "1" else None
        if judge_worker is None:
            self.logger.info("JudgeWorker 已禁用（ENABLE_JUDGE_WORKER 未设置），跳过 badcase 审计以节省 deepseek token")
        self.generator = ReplyGenerator(
            llm_client=actual_llm,
            complex_llm_client=complex_llm_client,
            memory_engine=self.memory_engine,
            tool_registry=registry,
            judge_worker=judge_worker,
            enable_timestamps=True,
        )
        self.sender = WeChatMessageSender(silent_mode=os.environ.get("WECHAT_SILENT_MODE") == "1")

        # 动态注册 send_file 工具（需要 sender 实例）
        def _load_shareable_files() -> dict:
            import json
            from pathlib import Path
            path = Path(__file__).parent.parent.parent / "data" / "shareable_files.json"
            if not path.exists():
                return {}
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}

        def _send_file_tool(file_name: str = "", chat_name: str = "") -> str:
            if not file_name:
                return "file_name 不能为空"
            shareable_files = _load_shareable_files()
            file_path = shareable_files.get(file_name)
            if not file_path:
                available = "、".join(shareable_files.keys()) if shareable_files else "（暂无）"
                return f"找不到文件 '{file_name}'。可发送文件：{available}"
            abs_path = os.path.abspath(file_path)
            if not os.path.exists(abs_path):
                return f"文件不存在: {file_path}"
            result = self.sender.send_file(abs_path, chat_name=chat_name)
            if result.success:
                return f"已发送文件: {file_name}"
            return f"发送失败: {result.error}"

        shareable_files = _load_shareable_files()
        file_list = "\n".join(f"- {name}" for name in shareable_files.keys())
        send_file_description = (
            "给用户发送本地文件。仅当用户明确要求发送以下文件时调用：\n"
            f"{file_list}\n\n"
            "参数 file_name 必须是上面列表中的名称（如'简历'），chat_name 为当前聊天名称。\n"
            "注意：当对方要求发送简历时，默认理解为对方（招聘方、朋友、HR 等）需要你的简历资料，"
            "不是你本人要找工作。回复时不要提及'跳槽''找工作'等假设，避免让对方误解。"
        )

        registry.register(
            name="send_file",
            description=send_file_description,
            parameters={
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "要发送的文件名称，必须是可发送列表中的一项，如：简历",
                    },
                    "chat_name": {
                        "type": "string",
                        "description": "当前聊天名称（从会话信息中获取）",
                    },
                },
                "required": ["file_name", "chat_name"],
            },
            func=_send_file_tool,
        )

        self._login_handler = WeChatLoginHandler()
        self.on_message = on_message
        self.running = False
        self.session_id = __import__('time').strftime("%Y%m%d%H%M%S") + "_" + str(os.getpid())
        self._tick_id = int(time.time())  # 跨重启唯一
        self.debug_mode = debug_mode
        self.debug_logger = DebugLogger()
        # 免回复聊天列表：公众号、系统账号等不需要回复的聊天
        raw_no_reply = os.environ.get("WECHAT_NO_REPLY_CHATS", "腾讯新闻,文件传输助手,公众号,服务号")
        self.no_reply_chats = {c.strip() for c in raw_no_reply.split(",") if c.strip()}

        # 切换聊天防抖：10 秒内不重复切换同一个目标
        self._last_switch_target: str = ""
        self._last_switch_time: float = 0.0
        self._switch_debounce_seconds: float = 10.0

        # 服务号/公众号列表恢复冷却
        self._last_recovery_time: float = 0.0

        # 全局状态持久化目录
        project_root = Path(__file__).parent.parent.parent
        (project_root / "data").mkdir(parents=True, exist_ok=True)

        # ===== WeFlow 全量初始化 =====
        self._weflow_mode = os.getenv("WEFLOW_MODE", "ocr")
        print(f"[WeFlow] 模式检测: WEFLOW_MODE={self._weflow_mode}")
        if self._weflow_mode in ("weflow", "hybrid"):
            try:
                has_wf = hasattr(self.perception, '_weflow_pipeline')
                wf_val = getattr(self.perception, '_weflow_pipeline', None)
                print(f"[WeFlow] pipeline检查: hasattr={has_wf}, value={wf_val}")
                if has_wf and wf_val:
                    weflow = wf_val
                    init_ok = weflow.initialize()
                    print(f"[WeFlow] initialize() 返回: {init_ok}")
                    if init_ok:
                        self._inject_weflow_history(weflow)
                        print(f"[WeFlow] 全量初始化完成: {weflow.init_total_messages} 条消息")
                        # 历史注入完成后，切回 OCR 模式运行，不再依赖 WeFlow API
                        self._weflow_mode = "ocr"
                        # 同步修改 perception 层的模式，确保后续 tick 不走 WeFlow
                        perception = getattr(self, 'perception', None)
                        if perception and hasattr(perception, '_weflow_mode'):
                            perception._weflow_mode = "ocr"
                            print("[WeFlow] 历史注入完成，perception 层已切换为 OCR 模式")
                        else:
                            print("[WeFlow] 历史注入完成，后续切换为 OCR 模式运行")
                    else:
                        print(f"[WeFlow] 初始化失败: {weflow.get_init_status().get('error')}")
                else:
                    print("[WeFlow] perception 中没有 _weflow_pipeline，跳过历史注入")
            except Exception as e:
                print(f"[WeFlow] 启动初始化异常: {e}")
                import traceback
                traceback.print_exc()
                self._weflow_mode = "ocr"

    def _inject_weflow_history(self, weflow_pipeline) -> None:
        """将 WeFlow 全量历史注入 GlobalStore。"""
        try:
            history_map = weflow_pipeline.export_all_history()
            total = 0
            for talker, messages in history_map.items():
                if not messages:
                    continue
                # 用第一条消息的 chat_name 作为 key
                chat_name = messages[0].chat_name or talker
                is_group = _is_group_chat_name(chat_name)
                injected = self.global_store.inject_history(chat_name, messages, mode="weflow", is_group=is_group)
                total += injected
            print(f"[WeFlow] 历史注入完成: {total} 条消息注入 GlobalStore")
        except Exception as e:
            print(f"[WeFlow] 历史注入失败: {e}")

    def tick(self) -> None:
        """执行一轮: 感知 -> 去重 -> 决策 -> 回复."""
        self._tick_id += 1
        tick_id = self._tick_id
        self.logger.log_tick_start(tick_id, interval=getattr(self, '_interval', 5.0))
        result = None

        # 在 tick 一开始就初始化调试日志，即使 perceive() 失败也有记录
        self.debug_logger.start_tick(tick_id, "")

        try:
            try:
                result = self.perception.perceive()
            except WeChatNotReadyError:
                recovery = self._login_handler.handle()
                if recovery.status == LoginRecoveryStatus.SUCCESS:
                    result = self.perception.perceive()
                else:
                    self.logger.log_capture(tick_id, success=False, error=f"微信未就绪且恢复失败: {recovery.message}")
                    self.debug_logger.log_action("none", action_input="", success=False, error=f"微信未就绪: {recovery.message}")
                    return

            if result is None:
                self.logger.log_capture(tick_id, success=False, error="未能获取微信窗口画面，可能原因：微信未启动、窗口被最小化、或需要扫码登录")
                self.logger.warning(
                    "未能获取微信窗口画面，可能原因：微信未启动、窗口被最小化、或需要扫码登录"
                )
                self.debug_logger.log_action("none", action_input="", success=False, error="perceive 返回 None")
                return

            # 记录 Perception 层输出
            if self.debug_logger.current is not None:
                self.debug_logger.current.screenshot_path = result.screenshot_path or ""
                self.debug_logger.log_perception_output(
                    chat_name=result.chat_name,
                    messages_count=len(result.messages),
                    chat_list_count=len(result.chat_list_items),
                )
            # 复制 SmartPipeline 的 debug_info（OCR/Layout/API 中间结果）
            if result.debug_info and isinstance(result.debug_info, dict):
                for k, v in result.debug_info.items():
                    if hasattr(self.debug_logger.current, k):
                        if k in ("tick_id",):
                            continue
                        if k.startswith("bot_"):
                            continue
                        if k.startswith("session_"):
                            continue
                        if k.startswith("action_"):
                            continue
                        setattr(self.debug_logger.current, k, v)

            if result.screenshot_path:
                try:
                    saved_path = self.global_store.save_screenshot(
                        result.screenshot_path, session_id=str(tick_id)
                    )
                    self.logger.debug(f"截图已保存: {saved_path}")
                    result.screenshot_path = str(saved_path)
                    if self.debug_logger.current is not None:
                        self.debug_logger.current.screenshot_path = str(saved_path)
                except Exception as e:
                    self.logger.warning("截图保存失败: %s", e)

            messages = result.messages
            raw_chat_name = result.chat_name or ""
            chat_name = _normalize_chat_name(raw_chat_name)
            is_group = result.is_group

            if not chat_name:
                if messages:
                    # 右侧有消息但标题栏 OCR 失败，不切换避免误点当前聊天
                    self.logger.warning("当前聊天名为空但检测到消息，标题栏识别失败，跳过切换避免误点")
                    self.debug_logger.log_action("none", action_input="", success=False, error="标题栏识别失败，跳过避免误点")
                else:
                    self.logger.warning("当前聊天名为空且无消息，可能未打开任何聊天窗口")
                    # 先尝试从服务号/公众号/文章列表等异常视图返回
                    recovered = self._recover_from_non_chat_view(result)
                    if recovered:
                        self.debug_logger.log_action("recover", action_input="back_button", success=True, error="尝试从服务号/公众号列表返回")
                        return
                    # 恢复失败或本身就是普通视图，尝试切换到未读
                    self.logger.warning("尝试切换到未读")
                    switch_target = self._try_switch_to_unread_chat(result)
                    if switch_target:
                        self.debug_logger.log_action("switch", action_input=switch_target, success=True)
                    else:
                        self.debug_logger.log_action("none", action_input="", success=False, error="聊天名为空且无未读")
                return

            # API/Layout 识别到当前是服务号/订阅号/公众号列表，触发返回
            if chat_name in self._SERVICE_ACCOUNT_NAMES:
                self.logger.warning(f"当前聊天名为'{chat_name}'，处于服务号/公众号列表，尝试返回")
                recovered = self._recover_from_non_chat_view(result)
                if recovered:
                    self.debug_logger.log_action("recover", action_input="back_button", success=True, error=f"尝试从{chat_name}列表返回")
                    return

            self.logger.log_layout(
                tick_id=tick_id,
                chat_name=chat_name,
                title_elem_count=0,
                input_elem_count=0,
                timestamp_elem_count=0,
                self_bubble_count=sum(1 for m in messages if m.sender_type.value == "self"),
                message_candidate_count=len(messages),
            )

            state, unreplied = self.global_store.merge_tick(
                chat_name, messages, mode=self._weflow_mode, is_group=is_group
            )

            # 记录 Session 层输入输出
            msg_dicts = [
                {
                    "sender": m.sender,
                    "sender_type": m.sender_type.value,
                    "text": m.text,
                    "type": m.message_type,
                    "image_desc": m.image_description,
                    "image_dup": m.is_image_duplicate,
                    "replied": m.replied,
                }
                for m in messages
            ]
            unreplied_dicts = [
                {
                    "sender": m.sender,
                    "sender_type": m.sender_type.value,
                    "text": m.text,
                    "type": m.message_type,
                    "image_desc": m.image_description,
                    "image_dup": m.is_image_duplicate,
                    "replied": m.replied,
                }
                for m in unreplied
            ]
            try:
                total_stored = len(state.messages)
            except (TypeError, AttributeError):
                total_stored = 0
            self.debug_logger.log_session(
                input_chat_name=chat_name,
                input_messages=msg_dicts,
                output_unreplied=unreplied_dicts,
                total_stored=total_stored,
            )

            self.logger.log_messages(
                tick_id=tick_id,
                total_messages=len(messages),
                new_messages=len(unreplied),
                message_details=msg_dicts,
            )

            if not unreplied:
                self.logger.log_decision(tick_id, should_reply=False, reason="无未回复消息", latest_text="")
                self.debug_logger.log_action("none", action_input="", success=False, error="无未回复消息")
                # 当前聊天无未回复消息，尝试切换到其他未读聊天
                switch_target = self._try_switch_to_unread_chat(result)
                if switch_target:
                    self.debug_logger.log_action(f"switch:{switch_target}")
                return

            for msg in unreplied:
                if self.on_message:
                    self.on_message(msg, state)

            # 收集所有需要回复的未读消息
            to_reply = [msg for msg in unreplied if self.policy.should_reply(msg, state)]
            if not to_reply:
                skip_reason = "无符合条件的消息可回复"
                self.logger.log_decision(
                    tick_id, should_reply=False,
                    reason=skip_reason,
                    latest_text=unreplied[-1].text if unreplied else ""
                )
                self.debug_logger.log_bot_decision(
                    chat_name=chat_name,
                    new_messages_count=len(unreplied),
                    should_reply=False,
                    switch_reason=skip_reason,
                )
                self.debug_logger.log_action("none", action_input="", success=False, error=skip_reason)
                return

            # 传递完整消息上下文 + 所有未读消息，让 AI 生成多条回复
            all_messages = getattr(state, "messages", [])
            if not isinstance(all_messages, list):
                all_messages = []
            replies = self.generator.generate(to_reply, all_messages, is_group=is_group, tick_id=tick_id)
            reply_text = " | ".join(replies) if replies else ""
            # 写 tick_log
            conn = None
            try:
                import json as _json

                from src.badcase.case_db import get_db
                conn = get_db()._get_conn()
                # 提取工具执行结果（从 generation_trace 中过滤 tool_execution 事件）
                trace = getattr(self.generator, 'last_generation_trace', []) or []
                tool_results = [
                    {"tool": t.get("tool_name", ""), "args": t.get("arguments", ""), "result": str(t.get("result", ""))[:2000]}
                    for t in trace if t.get("type") == "tool_execution"
                ]

                # 序列化原始消息数据供实验复跑使用
                # NFR-5: input 只存最近 50 条上下文（足够复跑），不存全量 state.messages。
                # 全量历史会在每个 tick 重复存（单 tick 曾达 2078 条/927KB）导致膨胀。
                def _serialize_msgs(msgs):
                    from dataclasses import asdict
                    data = []
                    for m in msgs:
                        d = asdict(m)
                        d['sender_type'] = m.sender_type.value
                        d.pop('source_elements', None)
                        data.append(d)
                    return _json.dumps(data, ensure_ascii=False, default=str)

                conn.execute("""INSERT INTO tick_log
                    (session_id, tick_id, chat_name, is_group,
                     messages_count, new_messages_count,
                     system_prompt, user_prompt, raw_response, tool_calls_json, tool_results_json,
                     session_input_messages_json, session_output_unreplied_json,
                     should_reply, replies_sent_json, screenshot_path)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    self.session_id,
                    tick_id, chat_name, 1 if is_group else 0,
                    len(all_messages) if all_messages else 0,
                    len(unreplied),
                    getattr(self.generator, 'last_system_prompt', '') or '',
                    getattr(self.generator, 'last_user_prompt', '') or '',
                    _raw_with_thinking(getattr(self.generator, 'last_raw_response', '') or '', getattr(self.generator, 'last_thinking', '')),
                    _json.dumps(getattr(self.generator, 'last_tool_calls', []) or [], ensure_ascii=False),
                    _json.dumps(tool_results, ensure_ascii=False),
                    _serialize_msgs(all_messages[-50:]) if all_messages else '[]',
                    _serialize_msgs(unreplied) if unreplied else '[]',
                    1,
                    _json.dumps(replies, ensure_ascii=False),
                    result.screenshot_path or '',
                ))
                conn.commit()
            except Exception as e:
                self.logger.warning("tick_log 写入失败: %s", e)
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception as e:
                        self.logger.warning("close tick_log connection failed: %s", e)
            self.logger.log_decision(
                tick_id, should_reply=True,
                reason=f"触发回复条件 (未读 {len(unreplied)} 条，需回复 {len(to_reply)} 条，生成 {len(replies)} 条回复)",
                latest_text=unreplied[-1].text, reply_text=reply_text
            )
            # 记录 LLM 回复生成的完整链路（含多轮调用 + 工具调用）
            if self.debug_logger.current is not None:
                self.debug_logger.log_reply_generation(
                    system_prompt=getattr(self.generator, 'last_system_prompt', ''),
                    user_prompt=getattr(self.generator, 'last_user_prompt', ''),
                    raw_response=getattr(self.generator, 'last_raw_response', ''),
                    llm_calls=getattr(self.generator, 'last_llm_calls', []),
                    tool_calls=getattr(self.generator, 'last_tool_calls', []),
                    trace=getattr(self.generator, 'last_generation_trace', []),
                    loaded_skills=getattr(self.generator, 'last_loaded_skills', []),
                    skill_injected_content=getattr(self.generator, 'last_skill_injected_content', ''),
                    active_llm=getattr(self.generator, 'last_active_llm', ''),
                    hermes_fallback_triggered=getattr(self.generator, 'last_hermes_fallback_triggered', False),
                    hermes_messages=getattr(self.generator, 'last_hermes_messages', []),
                    hermes_response=getattr(self.generator, 'last_hermes_response', ''),
                )
            self.debug_logger.log_bot_decision(
                chat_name=chat_name,
                new_messages_count=len(unreplied),
                should_reply=True,
                reply_target=unreplied[-1].text if unreplied else "",
                reply_text=reply_text,
            )

            if not replies:
                # LLM 返回空 replies = 判断无需回复（群聊没@我/纯表情包等），
                # 属正常决策。补明确日志，避免被误读为"漏回"。
                self.logger.log_decision(
                    tick_id, should_reply=False,
                    reason=f"LLM 判断无需回复 (未读 {len(unreplied)} 条，均已处理)",
                    latest_text=unreplied[-1].text if unreplied else ""
                )
                self.debug_logger.log_action("none", action_input="", success=False, error="LLM 判断无需回复")
                # 即使不回复，也标记为已处理，避免下一轮又当成未读
                for msg in to_reply:
                    self.global_store.mark_replied(chat_name, msg, "(未回复)")
                return

            # 免回复聊天：跳过回复，尝试切换到其他未读聊天
            if chat_name in self.no_reply_chats:
                self.logger.log_decision(
                    tick_id, should_reply=False,
                    reason=f"当前聊天 '{chat_name}' 在免回复列表中",
                    latest_text=unreplied[-1].text if unreplied else ""
                )
                self.debug_logger.log_action("none", action_input=reply_text, success=False, error="免回复聊天")
                switch_target = self._try_switch_to_unread_chat(result)
                if switch_target:
                    self.debug_logger.log_action("switch", action_input=switch_target, success=True)
                return

            # 逐条发送回复，间隔 1.5 秒
            send_failed = False
            send_skipped = False  # 静默模式跳过（非真实失败），应视为已处理避免卡循环
            for i, reply in enumerate(replies):
                action_result = self.sender.send(reply, chat_name=chat_name)
                if action_result.success:
                    self.logger.log_send(tick_id, success=True, text=reply)
                    self.debug_logger.log_action("send", action_input=reply, success=True)
                    # Bot 自己发的消息不直接进 history，放入 pending 等感知层确认
                    self_msg = ChatMessage(
                        text=reply, sender="自己", sender_type=SenderType.SELF,
                        chat_name=chat_name, replied=True, reply_text=reply,
                        reply_time=time.time(), message_type="text"
                    )
                    chat_state = self.global_store.chats.get(chat_name)
                    if chat_state is not None:
                        chat_state.pending_self_messages.append(self_msg)
                        self.logger.info("[Pending] %s 记录 pending self 消息 (pending 队列长度=%d): %.60s",
                                             chat_name, len(chat_state.pending_self_messages), reply)
                elif action_result.skipped:
                    # 静默模式主动跳过（非白名单聊天）：不真实发送，但视为已处理，
                    # mark_replied 避免该聊天因"未回复"反复重试占满轮询、挡住白名单群。
                    self.logger.log_send(tick_id, success=False, text=reply, error=action_result.error)
                    self.debug_logger.log_action("send", action_input=reply, success=False, error=action_result.error or "")
                    send_skipped = True
                    break
                else:
                    self.logger.log_send(tick_id, success=False, text=reply, error=action_result.error)
                    self.debug_logger.log_action("send", action_input=reply, success=False, error=action_result.error or "")
                    send_failed = True
                    break
                if i < len(replies) - 1:
                    time.sleep(1.5)

            # 只在发送成功或静默跳过时标记 to_reply 为已回复。
            # 真实发送失败不 mark_replied，否则对方消息被永久跳过。
            # 静默跳过 mark_replied：本就不打算发，不卡循环让 bot 能轮到其他聊天。
            if not send_failed:
                for msg in to_reply:
                    self.global_store.mark_replied(chat_name, msg, reply_text)
                if send_skipped:
                    self.logger.info("[Bot] 静默跳过已 mark_replied，下轮不再重试 %s", chat_name)
            else:
                self.logger.info("[Bot] 发送失败，未 mark_replied，下轮仍会重试 %s", chat_name)

            # Judge 评分已由 generator._submit_to_judge 异步处理（唯一路径）

            # 触发记忆更新（异步，不阻塞）
            if self.memory_engine is not None:
                if is_group:
                    # 群聊：同时更新群 wiki 和最后发言者 wiki
                    self.memory_engine.update_group_wiki(
                        group_name=chat_name,
                        chat_name=chat_name,
                        messages=to_reply,
                        bot_replies=replies,
                    )
                    user_name = to_reply[-1].sender if to_reply else ""
                    if user_name:
                        self.memory_engine.update_user_wiki(
                            user_name=user_name,
                            chat_name=chat_name,
                            messages=to_reply,
                            bot_replies=replies,
                        )
                else:
                    # 私聊：只更新用户 wiki
                    self.memory_engine.update_user_wiki(
                        user_name=chat_name,
                        chat_name=chat_name,
                        messages=to_reply,
                        bot_replies=replies,
                    )
            return

        except Exception as exc:
            self.logger.log_exception(tick_id, phase="tick", exc=exc)
            if self.debug_logger.current:
                self.debug_logger.log_action("none", action_input="", success=False, error=f"异常: {exc}")
            raise
        finally:
            if self.debug_logger.current is not None:
                try:
                    path = self.debug_logger.save()
                    self.logger.debug(f"调试日志已保存: {path}")
                except Exception as e:
                    self.logger.warning("调试日志保存失败: %s", e)
                self.debug_logger.current = None
            self.save_sessions()

    def save_sessions(self) -> None:
        """保存全局状态到磁盘（增量保存，只保存有变化的聊天）。"""
        try:
            self.global_store.save()
        except Exception as e:
            self.logger.warning(f"保存全局状态失败: {e}")

    def run_auto(self, interval: float = 5.0) -> None:
        """自动运行主循环"""
        self.running = True
        self._interval = interval
        while self.running:
            try:
                self.tick()
            except Exception as e:
                self.logger.error(f"Tick #{self._tick_id} 未捕获异常: {e}", exc_info=True)
            # 每 60 个 tick（约 5 分钟，按 5s 间隔）输出一次 DeepSeek token 统计
            if self._tick_id > 0 and self._tick_id % 60 == 0:
                from src.utils.qwen_client import QwenClient
                QwenClient.log_token_stats(self.logger.runtime_logger)
            time.sleep(interval)

    _SERVICE_ACCOUNT_NAMES = {"服务号", "订阅号", "公众号"}

    def _recover_from_non_chat_view(self, result: PerceptionResult) -> bool:
        """当当前视图为服务号/订阅号/公众号列表时，点击左上角返回按钮回到聊天视图。

        检测依据：
        1. layout 层识别到标题栏包含"服务号/订阅号/公众号"字样；
        2. fallback：API/OCR 返回的 chat_name 本身就是这三个名称之一。
        只在明确识别为该视图时才点击返回，避免在其他非聊天视图误操作。
        """
        is_service = result.is_service_account_list
        if not is_service:
            # API 路径（如 SmartPipeline）可能更准确地识别出 chat_name='服务号'
            chat_name = _normalize_chat_name(result.chat_name or "")
            if chat_name in self._SERVICE_ACCOUNT_NAMES:
                is_service = True
        if not is_service:
            return False

        # 冷却：避免连续 tick 高频点击返回
        now = time.time()
        if getattr(self, '_last_recovery_time', 0) and now - self._last_recovery_time < 3.0:
            self.logger.info("[Recovery] 恢复冷却中，跳过本次返回")
            return False

        window_rect = result.window_rect
        scale_factor = result.scale_factor
        if window_rect is None:
            self.logger.warning("[Recovery] window_rect 为空，无法点击返回")
            return False

        self.logger.warning("[Recovery] 检测到服务号/公众号列表，点击左上角返回按钮")
        try:
            clicker = ChatListClicker(window_rect, scale_factor)
            if clicker.click_back_button():
                self._last_recovery_time = time.time()
                self.logger.info("[Recovery] 返回按钮已点击，等待下一轮重新感知")
                return True
            self.logger.warning("[Recovery] 点击返回按钮失败")
            return False
        except Exception as e:
            self.logger.warning(f"[Recovery] 点击返回按钮异常: {e}")
            return False

    def _try_switch_to_unread_chat(self, result: PerceptionResult) -> str:
        """检测到其他聊天有未读时，切换到未读数最多的那个。

        WeFlow 模式下优先使用 API 轮询检测未读（比 OCR 识别角标更可靠）。
        防抖：10 秒内不重复切换同一个目标，防止反复点击导致右侧折叠。
        """
        if not getattr(self, 'enable_chat_switch', True):
            return ""

        current_chat = _normalize_chat_name(result.chat_name)
        target_name = ""
        target_unread: int | str = 0

        # ===== WeFlow 模式：API 轮询检测未读 =====
        if self._weflow_mode in ("weflow", "hybrid"):
            try:
                weflow = getattr(self.perception, '_weflow_pipeline', None)
                self.logger.info(f"[WeFlow] 切换检测: weflow={weflow is not None}, initialized={getattr(weflow, '_initialized', False)}")
                if weflow and weflow._initialized:
                    unread_chats = weflow.check_unread_all()
                    self.logger.info(f"[WeFlow] check_unread_all 返回 {len(unread_chats)} 个未读")
                    # 过滤当前聊天和免回复列表
                    for item in unread_chats:
                        name = _normalize_chat_name(item["name"])
                        self.logger.info(f"[WeFlow] 未读候选: {name} ({item['unread_count']}条), current={current_chat}")
                        if name == current_chat:
                            continue
                        if name in {_normalize_chat_name(c) for c in self.no_reply_chats}:
                            continue
                        target_name = item["name"]
                        target_unread = item["unread_count"]
                        break
                    if target_name:
                        self.logger.info(f"[WeFlow] 检测到未读: {target_name} ({target_unread}条)")
            except Exception as e:
                self.logger.warning(f"[WeFlow] 未读检测失败: {e}")

        # ===== OCR fallback：截图识别未读角标 =====
        # WeFlow 模式下：API 已经检测过未读，如果 API 返回空（没有未读），
        # 直接信任 API，不 fallback 到 OCR（OCR 在小窗口中误识别率太高，会导致无限切换）
        # 但仅在 WeFlow 实际已初始化且执行了检测时才信任结果
        if self._weflow_mode in ("weflow", "hybrid"):
            weflow = getattr(self.perception, '_weflow_pipeline', None)
            if weflow and weflow._initialized and not target_name:
                return ""

        if not target_name:
            chat_list_items = result.chat_list_items
            if not chat_list_items:
                return ""

            unread_items = [
                item for item in chat_list_items
                if item.unread_count
                and _normalize_chat_name(item.nickname) != current_chat
                and _normalize_chat_name(item.nickname) not in {
                    _normalize_chat_name(c) for c in self.no_reply_chats
                }
            ]
            if not unread_items:
                return ""

            # 优先选择未读数最多的
            unread_items.sort(key=lambda item: int(item.unread_count) if item.unread_count.isdigit() else 0, reverse=True)
            target = unread_items[0]
            target_name = target.nickname
            target_unread = target.unread_count

        if not target_name:
            return ""

        target_norm = _normalize_chat_name(target_name)

        # 防抖：10 秒内不重复切换同一个目标
        now = time.time()
        if target_norm == self._last_switch_target and (now - self._last_switch_time) < self._switch_debounce_seconds:
            self.logger.info(f"[WeFlow] 防抖: {target_name} 最近已切换，跳过")
            return ""

        window_rect = result.window_rect
        scale_factor = result.scale_factor
        if window_rect is None:
            self.logger.info("[WeFlow] window_rect 为 None，无法切换")
            return ""

        clicker = ChatListClicker(window_rect, scale_factor)

        # WeFlow 模式：在截图的 chat_list_items 中找匹配的 rect
        target_item = None
        if self._weflow_mode in ("weflow", "hybrid"):
            weflow = getattr(self.perception, '_weflow_pipeline', None)
            for item in result.chat_list_items:
                # 1) 直接昵称匹配
                if _normalize_chat_name(item.nickname) == target_norm:
                    target_item = item
                    self.logger.info(f"[WeFlow] chat_list_items 昵称匹配 '{target_name}'")
                    break
                # 2) talker 匹配（API 返回的是 talker/wxid，OCR 显示的是昵称）
                if weflow:
                    item_talker = weflow._resolve_talker(item.nickname)
                    if item_talker and _normalize_chat_name(item_talker) == target_norm:
                        target_item = item
                        self.logger.info(f"[WeFlow] chat_list_items talker 匹配 '{target_name}' via '{item.nickname}'")
                        break
            if target_item is None:
                self.logger.info(f"[WeFlow] chat_list_items 中无匹配 '{target_name}' 的项，共 {len(result.chat_list_items)} 个")

        # fallback 1: OCR 列表中的第一个未读项
        if target_item is None:
            unread_items = [
                item for item in result.chat_list_items
                if item.unread_count and _normalize_chat_name(item.nickname) != current_chat
                and _normalize_chat_name(item.nickname) not in {_normalize_chat_name(c) for c in self.no_reply_chats}
            ]
            self.logger.info(f"[WeFlow] chat_list_items 未读项: {len(unread_items)}/{len(result.chat_list_items)}")
            if unread_items:
                target_item = unread_items[0]
                target_name = target_item.nickname
                target_unread = target_item.unread_count
            elif result.chat_list_items and self._weflow_mode in ("weflow", "hybrid"):
                # fallback 2: WeFlow 检测到未读但 OCR 列表中无未读标记，尝试点击列表第一项
                #（WeFlow 已知有未读，但 OCR 未识别到角标，可能是小窗口中角标太小）
                target_item = result.chat_list_items[0]
                self.logger.info(f"[WeFlow] 未读 '{target_name}' 无精确 rect，尝试点击列表第一项 '{target_item.nickname}'")

        if target_item is None:
            self.logger.info(f"[WeFlow] 无法找到可点击的列表项，chat_list_items={len(result.chat_list_items)}")
            return ""

        self.logger.info(f"[WeFlow] 准备点击 '{target_item.nickname}' at ({target_item.rect.x},{target_item.rect.y})")
        clicked = clicker.click_item(target_item)
        if not clicked:
            self.logger.info(f"[WeFlow] 点击失败: '{target_item.nickname}'")
            return ""

        # 不再做点击后二次截图验证，依赖 5 秒 tick 兜底：
        # 若误点进服务号/公众号列表，下一轮 tick 会识别并触发返回按钮。
        self._last_switch_target = target_norm
        self._last_switch_time = now
        self.logger.info(f"🔄 切换聊天: {target_name!r} (未读 {target_unread})")
        self.debug_logger.log_bot_decision(switch_target=target_name, switch_reason=f"未读 {target_unread}")
        return target_name

    def send_to_chat(self, chat_name: str, text: str) -> ActionResult:
        """外部系统调用此接口主动发消息到指定聊天。"""
        result = self.sender.send(text, chat_name=chat_name)
        if result.success:
            norm = _normalize_chat_name(chat_name)
            # 创建一条虚拟的已回复消息记录，放入 pending 等感知层确认
            from src.models.base import ChatMessage, SenderType
            msg = ChatMessage(
                text=text, sender="自己", sender_type=SenderType.SELF,
                chat_name=norm, replied=True, reply_text=text, reply_time=time.time()
            )
            state = self.global_store.chats.get(norm)
            if state is not None:
                state.pending_self_messages.append(msg)
                self.logger.info("[Pending] %s 记录 pending self 消息 (pending 队列长度=%d): %.60s",
                                     norm, len(state.pending_self_messages), text)
        return result
