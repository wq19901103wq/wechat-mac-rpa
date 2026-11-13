#!/usr/bin/env python3
"""Build the full verified-human C0 development mother pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_persona_few_shots import Candidate, extract_candidates  # noqa: E402
from scripts.build_verified_same_chat_style_view import CUTOFF  # noqa: E402


DEFAULT_EXPORTS = ROOT / "data/exports/b"
DEFAULT_WIKI = ROOT / "data/memory/wiki/users"
DEFAULT_HOLDOUT = ROOT / "data/reports/private/adversarial_stage2_holdout_v1/manifest.json"
DEFAULT_OUTPUT = ROOT / "data/reports/private/c0_development_pool_v1/manifest.json"
MEDIA_MARKERS = ("[图片]", "[动画表情]", "[视频]", "<img", "refermsg", "[文件]")


def _normalized_pair(context: Any, reply: Any) -> str:
    context_list = context if isinstance(context, list) else [context]
    reply_list = reply if isinstance(reply, list) else [reply]
    return json.dumps([context_list, reply_list], ensure_ascii=False, separators=(",", ":"))


def _has_media_tail(context: list[str]) -> bool:
    tail = "\n".join(str(value) for value in context[-3:]).lower()
    return any(marker.lower() in tail for marker in MEDIA_MARKERS)


def _candidate_id(candidate: Candidate) -> str:
    payload = "\n".join([candidate.chat_id, *candidate.context, "=>", *candidate.replies])
    return "human_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _row(candidate: Candidate) -> dict[str, Any]:
    row = {
        "id": _candidate_id(candidate),
        "relationship": candidate.relation,
        "chat_id": candidate.chat_id,
        "context": [message["text"] for message in candidate.context_messages]
        if candidate.context_messages else candidate.context,
        "reply": candidate.replies,
        "timestamp": candidate.timestamp,
        "reply_shape": candidate.reply_shape,
        "source_provenance": candidate.source_provenance,
    }
    if candidate.context_messages:
        row["context_messages"] = candidate.context_messages
    if candidate.reply_messages:
        row["reply_messages"] = candidate.reply_messages
    return row


def build(
    exports: Path,
    wiki: Path,
    holdout: Path,
    output: Path,
    context_message_limit: int = 5,
) -> dict[str, Any]:
    candidates = extract_candidates(
        exports,
        wiki,
        human_before=CUTOFF,
        context_message_limit=context_message_limit,
    )
    holdout_rows = json.loads(holdout.read_text(encoding="utf-8")).get("cases", [])
    holdout_pairs = {
        _normalized_pair(row.get("context", []), row.get("reply", []))
        for row in holdout_rows
    }

    media_excluded = [candidate for candidate in candidates if _has_media_tail(candidate.context)]
    holdout_excluded = [
        candidate
        for candidate in candidates
        if _normalized_pair(candidate.context, candidate.replies) in holdout_pairs
    ]
    rows = [
        _row(candidate)
        for candidate in candidates
        if not _has_media_tail(candidate.context)
        and _normalized_pair(candidate.context, candidate.replies) not in holdout_pairs
    ]
    rows.sort(key=lambda row: (str(row["chat_id"]), int(row["timestamp"]), str(row["id"])))

    relationships = Counter(str(row["relationship"]) for row in rows)
    shapes = Counter(str(row["reply_shape"]) for row in rows)
    provenance = Counter(str(row["source_provenance"]) for row in rows)
    exact_overlap = sum(
        _normalized_pair(row["context"], row["reply"]) in holdout_pairs for row in rows
    )
    payload = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "full C0 development mother pool for case inspection and rotating regression; not formal evaluation",
        "selection_rule": "all eligible verified-human text turns before automation cutoff; no case-count cap",
        "context_message_limit": context_message_limit,
        "source": str(exports.relative_to(ROOT)),
        "frozen_holdout": str(holdout.relative_to(ROOT)),
        "holdout_exclusion": "exact normalized context + human reply pair",
        "counts": {
            "extracted_candidates": len(candidates),
            "development_pool": len(rows),
            "relationships": dict(sorted(relationships.items())),
            "reply_shapes": dict(sorted(shapes.items())),
            "source_provenance": dict(sorted(provenance.items())),
            "unique_chats": len({str(row["chat_id"]) for row in rows}),
            "media_tail_excluded": len(media_excluded),
            "holdout_pair_excluded": len(holdout_excluded),
            "exact_holdout_overlap": exact_overlap,
        },
        "usage": {
            "mother_pool": "all rows remain available for discovering recurring C0 style problems",
            "round_view": "each round draws representative cases; the view size is not the pool size",
            "per_round_score": "regression guard only",
            "formal_acceptance": "use the frozen 240-case holdout only after the candidate matures",
        },
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports", type=Path, default=DEFAULT_EXPORTS)
    parser.add_argument("--wiki", type=Path, default=DEFAULT_WIKI)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--context-message-limit", type=int, default=5)
    args = parser.parse_args()
    payload = build(
        args.exports,
        args.wiki,
        args.holdout,
        args.output,
        context_message_limit=args.context_message_limit,
    )
    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
