#!/usr/bin/env python3
"""WeFlow HTTP API 客户端 — 替代 OCR 直接读取微信数据库

用法:
    client = WeFlowClient()
    contacts = client.get_contacts()          # 所有联系人/群聊
    messages = client.get_messages(talker, limit=20)  # 某聊天消息
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.utils.xml_utils import _extract_xml_text

_logger = logging.getLogger("src.weflow_client")


@dataclass
class WeFlowMessage:
    local_id: int
    server_id: str
    local_type: int
    create_time: int          # 秒级时间戳
    sort_seq: int
    is_send: bool             # True=自己, False=对方
    sender_username: str
    content: str
    raw_content: str
    sender_display_name: str = ""  # WeFlow 返回的 sender 昵称

    @property
    def sender_type(self) -> str:
        return "self" if self.is_send else "other"

    @property
    def is_text(self) -> bool:
        return self.local_type == 1

    @property
    def is_xml(self) -> bool:
        return self.local_type in (33, 34, 47, 49, 66, 141733920817)


@dataclass
class WeFlowContact:
    username: str
    display_name: str
    type: str                 # friend | group | official
    nickname: str = ""
    alias: str = ""
    region: str = ""

    @property
    def is_group(self) -> bool:
        return self.type == "group" or self.username.endswith("@chatroom")

    @property
    def name(self) -> str:
        return self.nickname or self.display_name or self.username


class WeFlowClient:
    """WeFlow HTTP API 轻量客户端"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5031,
        access_token: str = "weflow_token_123",
        timeout: float = 10.0,
    ):
        self.base_url = f"http://{host}:{port}"
        self.access_token = access_token
        self.timeout = timeout
        self._contacts_cache: Optional[list[WeFlowContact]] = None
        self._contacts_ts: float = 0
        self._cache_ttl: float = 5.0  # 联系人缓存 5 秒

    # ------------------------------------------------------------------
    # 底层请求
    # ------------------------------------------------------------------
    def _request(self, path: str, params: Optional[dict] = None) -> dict:
        qs = {"access_token": self.access_token}
        if params:
            qs.update(params)
        url = f"{self.base_url}{path}?{urlencode(qs)}"
        req = Request(url, method="GET")
        req.add_header("Accept", "application/json")

        try:
            with urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"WeFlow API HTTP {e.code}: {body}") from e
        except Exception as e:
            raise RuntimeError(f"WeFlow API request failed: {e}") from e

        if not data.get("success") and "error" in data:
            raise RuntimeError(f"WeFlow API error: {data['error']}")
        return data

    # ------------------------------------------------------------------
    # 联系人
    # ------------------------------------------------------------------
    def get_contacts(self, use_cache: bool = True) -> list[WeFlowContact]:
        if use_cache and self._contacts_cache and time.time() - self._contacts_ts < self._cache_ttl:
            return self._contacts_cache

        data = self._request("/api/v1/contacts", {"limit": 10000})
        contacts = [
            WeFlowContact(
                username=c.get("username", ""),
                display_name=c.get("displayName", ""),
                type=c.get("type", ""),
                nickname=c.get("nickname", ""),
                alias=c.get("alias", ""),
                region=c.get("region", ""),
            )
            for c in data.get("contacts", [])
        ]
        self._contacts_cache = contacts
        self._contacts_ts = time.time()
        _logger.info("WeFlow contacts loaded: %d", len(contacts))
        return contacts

    def get_contact(self, username: str) -> Optional[WeFlowContact]:
        for c in self.get_contacts():
            if c.username == username:
                return c
        return None

    # ------------------------------------------------------------------
    # 消息
    # ------------------------------------------------------------------
    def get_messages(
        self,
        talker: str,
        limit: int = 20,
        offset: int = 0,
        contact_name: Optional[str] = None,
    ) -> tuple[list[WeFlowMessage], bool]:
        """获取消息，返回 (messages, has_more)"""
        data = self._request(
            "/api/v1/messages",
            {"talker": talker, "limit": limit, "offset": offset},
        )
        raw_msgs = data.get("messages", [])
        has_more = data.get("hasMore", False)

        messages = []
        for m in raw_msgs:
            # 群聊消息：sender_username 可能是 wxid，需要映射为昵称
            sender = m.get("senderUsername", "")
            sender_display_name = m.get("senderDisplayName", "")
            content = m.get("content", "")

            # 对 XML 类消息尝试提取文本摘要
            if m.get("localType", 1) != 1 and content and content.startswith("<"):
                content = _extract_xml_text(content) or content[:200]

            messages.append(
                WeFlowMessage(
                    local_id=m.get("localId", 0),
                    server_id=str(m.get("serverId", "")),
                    local_type=m.get("localType", 1),
                    create_time=m.get("createTime", 0),
                    sort_seq=m.get("sortSeq", 0),
                    is_send=bool(m.get("isSend", 0)),
                    sender_username=sender,
                    sender_display_name=sender_display_name,
                    content=content,
                    raw_content=m.get("rawContent", ""),
                )
            )

        _logger.debug(
            "WeFlow messages: talker=%s limit=%d offset=%d count=%d has_more=%s",
            talker, limit, offset, len(messages), has_more,
        )
        return messages, has_more

    def get_latest_messages(self, talker: str, limit: int = 5) -> list[WeFlowMessage]:
        """获取最新的 N 条消息（offset=0）"""
        msgs, _ = self.get_messages(talker, limit=limit, offset=0)
        return msgs

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------


    def health_check(self) -> bool:
        try:
            self._request("/api/v1/contacts")
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# 兼容 SmartPipeline 的转换函数
# ---------------------------------------------------------------------------

def weflow_messages_to_chat_messages(
    messages: list[WeFlowMessage],
    contact: WeFlowContact,
    my_wxid: str = "wxid_u89yxmv0ashf19",
) -> list[dict]:
    """把 WeFlowMessage 转成 SmartPipeline 能消费的格式

    输出格式与 vision_pipeline 的 API 结果兼容:
        [{"sender": "自己"|"对方"|昵称, "text": "...", "type": "text"}]
    """
    result = []
    for m in messages:
        if m.is_send:
            sender = "自己"
        elif contact.is_group:
            # 群聊：尝试找昵称
            sender = m.sender_username  # 先用 wxid，后面可映射
        else:
            sender = contact.name

        result.append({
            "sender": sender,
            "text": m.content,
            "type": "text" if m.is_text else "other",
            "raw_type": m.local_type,
            "create_time": m.create_time,
        })
    return result


if __name__ == "__main__":
    # 简单测试
    logging.basicConfig(level=logging.INFO)
    client = WeFlowClient()

    print("=== Health Check ===")
    print(client.health_check())

    print("\n=== Contacts ===")
    for c in client.get_contacts():
        print(f"  [{c.type:6s}] {c.name:20s} {c.username}")

    print("\n=== Latest Messages (王芊) ===")
    msgs, _ = client.get_messages("wxid_qxlscnesk92m21", limit=5)
    for m in reversed(msgs):  # 正序
        prefix = "[自己]" if m.is_send else "[对方]"
        print(f"  {prefix} {m.content[:60]}")
