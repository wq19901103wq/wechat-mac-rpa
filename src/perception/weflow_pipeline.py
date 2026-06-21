#!/usr/bin/env python3
"""WeFlow Pipeline — 全量初始化 + 增量同步感知管道

架构：
    启动: 并发拉取所有聊天的全部历史 → 存入 GlobalStore
    tick:  截图识别标题 → 增量同步新消息 → 返回 PerceptionResult

与现有 OCR 架构完全隔离，通过环境变量 WEFLOW_MODE 切换。
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

from src.capture.window_capture import WeChatNotReadyError, WindowCapture
from src.layout.layout_parser import LayoutParser
from src.layout.profile import LayoutProfile
from src.models.base import ChatMessage, PerceptionResult, SenderType
from src.ocr.vision_ocr import VisionOCREngine
from src.utils.chat_utils import _is_group_chat_name
from src.utils.xml_utils import _extract_xml_text

from .weflow_client import WeFlowClient, WeFlowMessage

_logger = logging.getLogger("src.weflow_pipeline")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

WEFLOW_TICK_LIMIT = int(os.getenv("WEFLOW_TICK_LIMIT", "20"))
WEFLOW_MAX_WORKERS = int(os.getenv("WEFLOW_MAX_WORKERS", "5"))
WEFLOW_FETCH_TIMEOUT = float(os.getenv("WEFLOW_FETCH_TIMEOUT", "30.0"))


# ---------------------------------------------------------------------------
# WeFlow 感知管道
# ---------------------------------------------------------------------------

class WeFlowPipeline:
    """WeFlow 感知管道。

    启动时全量预加载所有聊天历史，之后每个 tick 只做增量同步。
    截图降级为"状态检测器"：只识别标题栏 + 聊天列表，不识别消息区。
    """

    def __init__(
        self,
        profile: LayoutProfile,
        weflow_client: Optional[WeFlowClient] = None,
        tick_limit: int = WEFLOW_TICK_LIMIT,
    ):
        self.profile = profile
        self.capture = WindowCapture()
        self.ocr = VisionOCREngine()
        self.layout = LayoutParser(profile)
        self.weflow = weflow_client or WeFlowClient()

        self.tick_limit = tick_limit

        # 联系人缓存
        self._contacts_cache: list = []
        self._contacts_ts: float = 0
        self._contacts_ttl: float = 300.0  # 5 分钟

        # 每个聊天的 last_local_id（用于增量同步）
        self._last_local_ids: dict[str, int] = {}

        # 全量初始化缓存的历史消息（供 export_all_history 复用）
        self._history_cache: dict[str, list[WeFlowMessage]] = {}

        # 标题 → talker 映射缓存
        self._chat_name_to_talker: dict[str, str] = {}
        self._talker_to_contact: dict[str, dict] = {}

        # 全量初始化标志
        self._initialized: bool = False
        self._init_error: Optional[str] = None

        # 统计
        self.init_time_ms: float = 0.0
        self.init_total_messages: int = 0
        self.tick_count: int = 0
        self.tick_api_time_ms: float = 0.0

    # ------------------------------------------------------------------
    # 启动：全量初始化
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """启动时全量加载所有聊天的历史消息。返回是否成功。"""
        if self._initialized:
            return True

        _logger.info("[WeFlow] 开始全量初始化...")
        t0 = time.time()

        try:
            # 1. 加载联系人（重试 3 次，WeFlow 服务可能刚启动）
            for attempt in range(3):
                self._refresh_contacts()
                if self._contacts_cache:
                    break
                _logger.warning(f"[WeFlow] 联系人列表为空，重试 {attempt + 1}/3...")
                time.sleep(1.0)
            if not self._contacts_cache:
                self._init_error = "WeFlow 联系人列表为空"
                _logger.error(f"[WeFlow] {self._init_error}")
                return False

            # 2. 并发拉取每个聊天的历史
            self._fetch_all_history()

            self._initialized = True
            self.init_time_ms = (time.time() - t0) * 1000
            _logger.info(
                "[WeFlow] 全量初始化完成: %d 个聊天, %d 条消息, %.0fms",
                len(self._contacts_cache),
                self.init_total_messages,
                self.init_time_ms,
            )
            return True

        except Exception as e:
            self._init_error = str(e)
            _logger.exception("[WeFlow] 全量初始化失败")
            return False

    def _refresh_contacts(self) -> None:
        """加载并缓存联系人列表。"""
        now = time.time()
        if self._contacts_cache and now - self._contacts_ts < self._contacts_ttl:
            return

        contacts = self.weflow.get_contacts()
        self._contacts_cache = contacts
        self._contacts_ts = now

        # 构建映射表
        self._chat_name_to_talker = {}
        self._talker_to_contact = {}
        for c in contacts:
            talker = c.username
            # 用 nickname 优先，其次是 displayName
            name = c.nickname or c.display_name or talker
            self._chat_name_to_talker[name] = talker
            self._talker_to_contact[talker] = {
                "username": talker,
                "name": name,
                "type": c.type,
                "is_group": c.is_group,
            }
            # 也存 displayName 的映射（防 OCR 识别差异）
            if c.display_name and c.display_name != name:
                self._chat_name_to_talker[c.display_name] = talker

        _logger.info("[WeFlow] 联系人缓存已刷新: %d 个", len(contacts))

    def _fetch_all_history(self, limit: int = 10000) -> None:
        """串行拉取所有聊天的全部历史消息，分页直到拉完。

        Args:
            limit: 每页拉取条数，默认 10000。WeFlow API 可能有内部上限，
                   但设大点可以一次拉更多，减少分页次数。
        """
        results: dict[str, list[WeFlowMessage]] = {}

        for contact in self._contacts_cache:
            talker = contact.username
            name = contact.nickname or contact.display_name or talker
            msgs: list[WeFlowMessage] = []
            offset = 0
            page = 0

            while True:
                try:
                    batch, has_more = self.weflow.get_messages(talker, limit=limit, offset=offset)
                    page += 1
                    if batch:
                        msgs.extend(batch)
                    if not batch or not has_more:
                        break
                    offset += len(batch)
                except Exception as e:
                    _logger.warning("[WeFlow] 拉取 %s 历史失败: %s", name, e)
                    break

            if msgs:
                results[talker] = msgs
                self._history_cache[talker] = msgs
                self.init_total_messages += len(msgs)
                _logger.info("[WeFlow] 拉取 %s: %d 条 (%d 页)", name, len(msgs), page)

        _logger.info(
            "[WeFlow] 历史拉取完成: %d 个聊天, %d 条消息",
            len(results),
            self.init_total_messages,
        )

    # ------------------------------------------------------------------
    # tick：感知
    # ------------------------------------------------------------------

    def perceive(self) -> Optional[PerceptionResult]:
        """执行一轮感知：截图确认标题 + 增量同步消息。

        Returns:
            PerceptionResult: 结构化结果（消息来自 WeFlow 数据库）
            None: 窗口捕获失败或无法识别标题
        """
        self.tick_count += 1

        # 0. 确保已初始化
        if not self._initialized:
            if not self.initialize():
                _logger.warning("[WeFlow] 初始化失败，返回 None")
                return None

        # 1. 截图（只做状态检测）
        try:
            capture_result = self.capture.capture()
        except WeChatNotReadyError as e:
            _logger.warning("[WeFlow] 窗口捕获失败: %s", e)
            return None

        image_path = capture_result.image_path
        _logger.debug("[WeFlow] 截图: %s", Path(image_path).name)

        # 2. 轻量 OCR：只识别标题 + 聊天列表
        elements = self.ocr.recognize(image_path)
        layout = self.layout.parse(elements, image_path)

        chat_name = layout.chat_name

        # 3. 标题 → talker 映射（支持 fallback）
        talker = None
        if chat_name:
            talker = self._resolve_talker(chat_name)

        # fallback：如果标题 OCR 失败，用 WeFlow API 轮询推断当前聊天
        if not talker:
            _logger.debug("[WeFlow] 标题 OCR 失败('%s')，尝试 API 推断当前聊天", chat_name)
            # 检查最近活跃的聊天（有未读或最新消息的）
            try:
                unread = self.check_unread_all()
                if unread:
                    # 第一个未读聊天最可能是当前需要处理的
                    fallback = unread[0]
                    talker = fallback["talker"]
                    chat_name = fallback["name"]
                    _logger.info("[WeFlow] API 推断当前聊天: %s (未读 %d 条)", chat_name, fallback["unread_count"])
                else:
                    # 无未读，检查所有聊天的最新消息（跳过异常的）
                    for contact in self._contacts_cache:
                        t = contact.username
                        try:
                            msgs, _ = self.weflow.get_messages(t, limit=1)
                        except Exception:
                            continue
                        if msgs:
                            talker = t
                            chat_name = contact.nickname or contact.display_name or t
                            break
            except Exception as e:
                _logger.warning("[WeFlow] API 推断失败: %s", e)

        if not talker:
            _logger.warning("[WeFlow] 无法确定当前聊天 (OCR='%s')", chat_name)
            return PerceptionResult(
                chat_name=chat_name or "",
                messages=[],
                chat_list_items=layout.chat_list_items,
                screenshot_path=image_path,
                window_rect=capture_result.window_rect,
                scale_factor=capture_result.scale_factor,
                debug_info={"source": "weflow", "error": "unknown_talker", "chat_name": chat_name},
                is_service_account_list=layout.is_service_account_list,
            )

        # 4. 增量同步：拉取新消息
        print(f"[perceive] about to call _sync_incremental for {talker}")
        t0 = time.time()
        new_messages = self._sync_incremental(talker)
        print(f"[perceive] _sync_incremental returned {len(new_messages)} messages")
        self.tick_api_time_ms += (time.time() - t0) * 1000

        # 5. 转换为 ChatMessage
        contact = self._talker_to_contact.get(talker)
        chat_messages = self._convert_messages(new_messages, contact, chat_name)

        # 6. 组装 PerceptionResult
        return PerceptionResult(
            chat_name=chat_name,
            messages=chat_messages,
            chat_list_items=layout.chat_list_items,
            screenshot_path=image_path,
            window_rect=capture_result.window_rect,
            scale_factor=capture_result.scale_factor,
            debug_info={
                "source": "weflow",
                "talker": talker,
                "new_messages": len(new_messages),
                "total_history": self.init_total_messages,
            },
            is_service_account_list=layout.is_service_account_list,
        )

    def _sync_incremental(self, talker: str) -> list[WeFlowMessage]:
        """增量同步：拉取该聊天自 last_local_id 之后的新消息。"""
        print(f"[_sync_incremental] called for {talker}, _last_local_ids={self._last_local_ids.get(talker, 0)}")
        last_id = self._last_local_ids.get(talker, 0)

        # 首次处理该聊天：从历史缓存初始化 last_local_id
        if last_id == 0 and talker in self._history_cache and self._history_cache[talker]:
            last_id = max(m.local_id for m in self._history_cache[talker])
            self._last_local_ids[talker] = last_id
            _logger.debug("[WeFlow] %s 首次处理，从历史缓存初始化 last_id=%d", talker, last_id)

        # 策略：先拉 tick_limit 条，如果第一条的 localId > last_id，说明有新消息
        # 如果全部 localId <= last_id，则无新消息
        # 如果有 hasMore，继续拉直到 localId <= last_id
        msgs: list[WeFlowMessage] = []
        offset = 0
        limit = self.tick_limit
        max_rounds = 5

        for _ in range(max_rounds):
            try:
                batch, has_more = self.weflow.get_messages(talker, limit=limit, offset=offset)
            except Exception as e:
                _logger.warning("[WeFlow] 增量同步 %s 失败: %s", talker, e)
                break

            if not batch:
                break

            # 过滤：只保留 local_id > last_id 的消息
            new_batch = [m for m in batch if m.local_id > last_id]
            msgs.extend(new_batch)

            # 如果这批消息全部都已见过，停止
            if not new_batch and batch:
                # 所有消息的 localId 都 <= last_id
                break

            # 更新 last_local_id
            if new_batch:
                self._last_local_ids[talker] = max(m.local_id for m in new_batch)

            if not has_more:
                break

            offset += len(batch)

        if msgs:
            _logger.info(
                "[WeFlow] 增量同步 %s: %d 条新消息 (last_id=%d → %d)",
                talker,
                len(msgs),
                last_id,
                self._last_local_ids.get(talker, last_id),
            )
        else:
            _logger.info("[WeFlow] 增量同步 %s: 0 条新消息 (last_id=%d)", talker, self._last_local_ids.get(talker, last_id))

        return msgs

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _resolve_talker(self, chat_name: str) -> Optional[str]:
        """聊天名 → wxid/talker 映射。"""
        # 直接匹配
        if chat_name in self._chat_name_to_talker:
            return self._chat_name_to_talker[chat_name]

        # 尝试去掉群人数后缀再匹配（如 "柚子群2（128）" → "柚子群2"）
        from src.utils.chat_utils import _normalize_chat_name
        cleaned = _normalize_chat_name(chat_name)
        if cleaned != chat_name and cleaned in self._chat_name_to_talker:
            return self._chat_name_to_talker[cleaned]

        # 模糊匹配（应对 OCR 识别差异）
        for name, talker in self._chat_name_to_talker.items():
            # 简单包含匹配
            if name in chat_name or chat_name in name:
                return talker
            # 长短名匹配
            if len(name) > 2 and len(chat_name) > 2:
                # 计算公共子串比例
                shorter, longer = (name, chat_name) if len(name) < len(chat_name) else (chat_name, name)
                if shorter in longer:
                    return talker

        return None

    def _convert_messages(
        self,
        messages: list[WeFlowMessage],
        contact: Optional[dict],
        chat_name: str,
    ) -> list[ChatMessage]:
        """WeFlowMessage → ChatMessage。"""
        results = []
        # 优先使用 WeFlow 返回的 is_group 字段，若 contact 缺失或字段未提供，
        # 则回退到群名正则判断（避免 WeFlow contact 中 is_group 为 false 导致误判私聊）
        is_group = contact.get("is_group", _is_group_chat_name(chat_name)) if contact else _is_group_chat_name(chat_name)

        # 构建群成员昵称映射（如果有）
        member_names: dict[str, str] = {}
        if is_group:
            # TODO: 群聊成员昵称映射（WeFlow 暂不提供成员列表接口，先用 contacts 缓存）
            for c in self._contacts_cache:
                member_names[c.username] = c.nickname or c.display_name or c.username

        for m in messages:
            # sender 识别
            if m.is_send:
                sender = "自己"
                sender_type = SenderType.SELF
            else:
                if is_group:
                    # 群聊：优先用 WeFlow 返回的 senderDisplayName，没有再用 wxid 映射
                    sender = m.sender_display_name or member_names.get(m.sender_username, m.sender_username)
                else:
                    # 私聊：统一为对方昵称
                    sender = chat_name
                sender_type = SenderType.OTHER

            # 内容处理：XML 消息提取摘要
            content = m.content
            if m.local_type != 1 and content.startswith("<"):
                extracted = _extract_xml_text(content)
                if extracted:
                    content = extracted

            msg = ChatMessage(
                text=content,
                sender=sender,
                sender_type=sender_type,
                chat_name=chat_name,
                message_type="text" if m.local_type == 1 else "other",
            )
            # 附加 WeFlow 专属字段（不影响现有代码，因为都是可选/额外属性）
            msg.local_id = m.local_id
            msg.server_id = m.server_id
            msg.create_time = m.create_time
            msg.raw_type = m.local_type
            msg.sender_wxid = m.sender_username

            results.append(msg)

        return results

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """WeFlow API 是否可用。"""
        return self.weflow.health_check()

    def get_init_status(self) -> dict:
        """返回初始化状态（供调试）。"""
        return {
            "initialized": self._initialized,
            "error": self._init_error,
            "init_time_ms": self.init_time_ms,
            "init_total_messages": self.init_total_messages,
            "contacts_count": len(self._contacts_cache),
            "tick_count": self.tick_count,
            "avg_tick_api_ms": self.tick_api_time_ms / max(self.tick_count, 1),
        }

    def get_all_local_ids(self) -> dict[str, int]:
        """返回所有聊天的 last_local_id（供 GlobalStore 使用）。"""
        return dict(self._last_local_ids)

    def check_unread_all(self) -> list[dict]:
        """轮询所有聊天，检测哪些有未读消息。

        策略：拉取每个聊天的最新 5 条消息，对比 last_local_id。
        如果最新消息的 localId > last_local_id，说明有新消息。

        Returns:
            [{"talker": str, "name": str, "unread_count": int, "latest_msg": WeFlowMessage}, ...]
            按 unread_count 降序排列
        """
        unread_chats = []
        check_limit = 5
        checked = 0
        found_unread = 0

        for contact in self._contacts_cache:
            talker = contact.username
            name = contact.nickname or contact.display_name or talker
            last_id = self._last_local_ids.get(talker, 0)
            try:
                msgs, _ = self.weflow.get_messages(talker, limit=check_limit, offset=0)
            except Exception:
                continue
            checked += 1

            if not msgs:
                continue

            # 与 _sync_incremental 保持一致：last_id==0 时初始化
            if last_id == 0:
                if talker in self._history_cache and self._history_cache[talker]:
                    last_id = max(m.local_id for m in self._history_cache[talker])
                else:
                    # 历史缓存中没有，用当前 msgs 的最大 local_id 初始化
                    last_id = max(m.local_id for m in msgs)

            # 过滤：只统计 local_id > last_id 的消息
            new_msgs = [m for m in msgs if m.local_id > last_id]
            if new_msgs:
                found_unread += 1
                unread_chats.append({
                    "talker": talker,
                    "name": name,
                    "unread_count": len(new_msgs),
                    "latest_msg": new_msgs[0],
                })
                _logger.info("[WeFlow] check_unread_all %s: last_id=%d, msgs_max=%d, new=%d",
                             talker, last_id, max(m.local_id for m in msgs), len(new_msgs))

        if found_unread:
            _logger.info("[WeFlow] check_unread_all: 检查 %d 个联系人，发现 %d 个未读", checked, found_unread)
        else:
            _logger.info("[WeFlow] check_unread_all: 检查 %d 个联系人，无未读", checked)

        # 按未读数降序
        unread_chats.sort(key=lambda x: x["unread_count"], reverse=True)
        return unread_chats

    def export_all_history(self) -> dict[str, list[ChatMessage]]:
        """导出所有已加载的历史消息（供 GlobalStore 注入使用）。

        直接复用 _fetch_all_history() 缓存的数据，不再调用 API。

        Returns:
            {talker: [ChatMessage, ...], ...}
        """
        result: dict[str, list[ChatMessage]] = {}
        for talker, msgs in self._history_cache.items():
            if not msgs:
                continue
            contact = self._talker_to_contact.get(talker)
            chat_name = contact.get("name", talker) if contact else talker
            chat_messages = self._convert_messages(msgs, contact, chat_name)
            if chat_messages:
                result[talker] = chat_messages
        _logger.info("[WeFlow] 导出历史缓存: %d 个聊天, %d 条消息", len(result), sum(len(v) for v in result.values()))
        return result
