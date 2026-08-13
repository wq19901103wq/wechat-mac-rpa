#!/usr/bin/env python3
"""Reject common private artifacts and identifiers before they enter Git."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


PRIVATE_PATHS = re.compile(
    r"(^|/)(persona\.md$|data/|backups?/|chat_exports?/|wechat_exports?/|"
    r"conversation_dumps?/|private_few_?shots?/)|\.(db|sqlite3?|log)$",
    re.IGNORECASE,
)
IDENTIFIERS = {
    "personal home path": re.compile(rb"/Users/(?!your-?name|example-user)[^/\s\"']+"),
    "WeChat identifier": re.compile(
        rb"wxid_(?!example_|xxx\b|self\b|other\b|old\b|index\b|set\b|[ab]{1,3}\b)[A-Za-z0-9_]+"
    ),
    "phone number": re.compile(rb"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "email address": re.compile(
        rb"[A-Za-z0-9._%+-]+@(?!example\.com\b)[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    ),
}


def repository_files() -> list[str]:
    raw = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    )
    return [item.decode() for item in raw.split(b"\0") if item]


def main() -> int:
    failures: list[str] = []
    for name in repository_files():
        if name == "scripts/check_privacy.py":
            continue
        if PRIVATE_PATHS.search(name):
            failures.append(f"private path is tracked: {name}")
            continue
        path = Path(name)
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data[:4096]:
            continue
        for label, pattern in IDENTIFIERS.items():
            if pattern.search(data):
                failures.append(f"{label} found in: {name}")

    if failures:
        print("Privacy check failed:", file=sys.stderr)
        print("\n".join(f"- {item}" for item in failures), file=sys.stderr)
        return 1
    print("Privacy check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
