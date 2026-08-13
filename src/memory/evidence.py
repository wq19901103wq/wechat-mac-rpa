#!/usr/bin/env python3
"""Reusable memory evidence hygiene helpers.

These helpers are **role/provenance based**, not keyword based, and are kept
independent of any specific entity names so they can be unit-tested with
synthetic data.

- :func:`is_self_message`        — role/provenance check for bot/self messages.
- :func:`format_evidence_conversation` — builds the transcript passed to wiki
  extraction, excluding bot/self content because the bot cannot be its own
  factual source.
- :func:`strip_unverified_lines` — removes ``[待验证]`` derived lines from a
  wiki body at runtime so untrusted derived facts never reach the LLM.
"""

import logging
from typing import Any, List


_logger = logging.getLogger(__name__)


def is_self_message(msg: Any) -> bool:
    """Return True if a message originates from the bot itself.

    Role/provenance based: inspects ``sender_type`` (``"self"``), never the
    message text. Tolerates both enum instances (``.value == "self"``) and
    plain strings.
    """
    st = getattr(msg, "sender_type", None)
    if st is None:
        return False
    if hasattr(st, "value"):
        return st.value == "self"
    return str(st) == "self"


def format_evidence_conversation(messages: List[Any], bot_replies: List[str]) -> str:
    """Build the evidence transcript passed to wiki extraction.

    Bot/self messages and ``bot_replies`` are excluded entirely because the
    bot's own words (and the bot's un-acknowledged replies) are not valid
    factual evidence for a user's identity. This is done by role/provenance,
    never by scanning message text for keywords.

    The bot_replies parameter is kept for call-site compatibility; it is
    intentionally not appended to the transcript.
    """
    del bot_replies  # bot replies are not factual evidence — excluded by design.
    lines: List[str] = []
    last_chat_name = None
    for msg in messages:
        if is_self_message(msg):
            continue

        chat_name = getattr(msg, "chat_name", "")
        if chat_name and chat_name != last_chat_name:
            lines.append(f"\n===== {chat_name} =====")
            last_chat_name = chat_name

        sender = getattr(msg, "sender", "")
        text = getattr(msg, "text", "")
        account = getattr(msg, "account", "")

        prefix = ""
        if account:
            prefix += f"[{account}]"
        ts = getattr(msg, "create_time", None)
        if ts:
            try:
                import time as _time
                ts_str = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(int(ts)))
                prefix += f"[{ts_str}]"
            except Exception:
                _logger.debug("invalid evidence timestamp: %r", ts, exc_info=True)
        if prefix:
            lines.append(f"{prefix}{sender}：{text}")
        else:
            lines.append(f"{sender}：{text}")
    return "\n".join(lines)


def strip_unverified_lines(text: str) -> str:
    """Remove untrusted ``[待验证]`` derived lines from a wiki body.

    Every line carrying the marker is removed, regardless of Markdown shape.
    The stored wiki files are left untouched — this is applied at runtime
    retrieval so untrusted derived lines never become evidence.
    """
    if not text or "[待验证]" not in text:
        return text
    kept: List[str] = []
    for line in text.split("\n"):
        if "[待验证]" in line:
            continue
        kept.append(line)
    return "\n".join(kept)
