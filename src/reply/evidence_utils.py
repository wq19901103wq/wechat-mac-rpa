#!/usr/bin/env python3
"""Evidence hygiene for reply self-refinement / fact-checking.

When the reply generator builds fact-check evidence, the raw prompt contains
``<history>`` blocks that may include the bot's own previous replies. The bot
must not corroborate itself, so assistant-history lines are removed *by
structural role* (the ``我：`` sender label used to render ``SenderType.SELF``),
never by scanning message text for keywords.

Tool/memory evidence (``<tool_results>``, ``<context>``, ``<session>``,
``<other_info>``, ``<group_info>``, ``<cached_data>``) is preserved — those are
legitimate external evidence.
"""

import re
from typing import List

# Structural rendering of a SELF-role message in a <history> block:
#   "我：..."  or  "我（时间标签）：..."
# The "我" sender label is how _format_message_line renders SenderType.SELF,
# so filtering by it is role/provenance based, not keyword based.
_SELF_LINE_RE = re.compile(r"^我([（(][^）)]*[）)])?[：:]\s*")
_SELF_MESSAGE_RE = re.compile(
    r'<message\b[^>]*\brole="self"[^>]*>.*?</message>\s*',
    re.DOTALL,
)

_HISTORY_BLOCK_RE = re.compile(r"(<history>.*?</history>)", re.DOTALL)


def _clean_history_block(block: str) -> str:
    block = _SELF_MESSAGE_RE.sub("", block)
    lines = block.split("\n")
    kept: List[str] = []
    for line in lines:
        stripped = line.strip()
        if _SELF_LINE_RE.match(stripped):
            continue
        kept.append(line)
    return "\n".join(kept)


def strip_assistant_history_lines(evidence: str) -> str:
    """Remove the bot's own previous replies from ``<history>`` blocks.

    Every other block (``<session>``, ``<context>``, ``<unread>``,
    ``<tool_results>``, memory info) is left intact. If the evidence contains
    no ``<history>`` block, it is returned unchanged.
    """
    if not evidence or "<history>" not in evidence:
        return evidence
    return _HISTORY_BLOCK_RE.sub(
        lambda m: _clean_history_block(m.group(0)), evidence
    )


def strip_assistant_history_from_parts(evidence_parts: List[str]) -> List[str]:
    """Map a list of evidence blocks through :func:`strip_assistant_history_lines`."""
    return [strip_assistant_history_lines(part) for part in evidence_parts]
