"""微信消息内容解码（ZSTD / 明文）与发送者前缀解析。"""

import logging
import re
from typing import Optional

import zstandard

_logger = logging.getLogger("src.platform.windows.message_codec")

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_SENDER_RE = re.compile(
    r"^(wxid_[0-9a-zA-Z_]+|gh_[0-9a-zA-Z_]+|\d+@chatroom|.+@openim|[\w.-]+):\n"
)


def decompress_content(data: bytes, max_size: int = 100 * 1024 * 1024) -> bytes:
    """ZSTD 解压（仅当命中魔数），否则原样返回。"""
    if data[:4] == ZSTD_MAGIC:
        try:
            return zstandard.ZstdDecompressor().decompress(data, max_output_size=max_size)
        except Exception:
            _logger.debug("ZSTD 解压失败，按原文处理")
    return data


def decode_message_content(raw: object) -> str:
    """把 message_content 列解码为文本（bytes 先解压再 utf-8）。"""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return decompress_content(raw).decode("utf-8", errors="replace")
    return str(raw)


def split_sender_prefix(
    content: str,
    known_senders: Optional[set[str]] = None,
) -> tuple[Optional[str], str]:
    """如果内容形如 '<sender>:\\n<正文>'，返回 (sender, 正文)；否则 (None, 原文)。

    sender 需命中常见模式或存在于 Name2Id 已知发送者集合，避免误拆正文。
    """
    if "\n" not in content:
        return None, content
    head, _, body = content.partition(":\n")
    if not head:
        return None, content
    if known_senders is not None:
        if head in known_senders:
            return head, body
        return None, content
    if _SENDER_RE.match(content):
        return head, body
    return None, content
