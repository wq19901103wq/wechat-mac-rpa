#!/usr/bin/env python3
"""Reject common private artifacts and identifiers before they enter Git."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


PRIVATE_PATHS = re.compile(
    r"(^|/)(\.env$|config/digital_human\.json$|persona\.md$|data/|backups?/|chat_exports?/|wechat_exports?/|"
    r"conversation_dumps?/|private_few_?shots?/)|\.(db|sqlite3?|log)$",
    re.IGNORECASE,
)
IDENTIFIERS = {
    "personal home path": re.compile(rb"/Users/(?!your-?name|example-user)[^/\s\"']+"),
    "WeChat identifier": re.compile(
        rb"wxid_(?!example_|xxx\b|self\b|other\b|friend\b|old\b|index\b|set\b|[ab]{1,3}\b)[A-Za-z0-9_]+"
    ),
    "WeChat group identifier": re.compile(rb"\b\d{6,}@chatroom\b"),
    "phone number": re.compile(rb"(?<![A-Za-z0-9])1[3-9]\d{9}(?![A-Za-z0-9])"),
    "email address": re.compile(
        rb"[A-Za-z0-9._%+-]+@(?!example\.(?:com|org|test)\b)[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    ),
    "API secret": re.compile(rb"\b(?:sk-|ghp_|github_pat_)[A-Za-z0-9_-]{20,}\b"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}

def repository_files(staged: bool = False) -> list[str]:
    if staged:
        raw = subprocess.check_output(["git", "ls-files", "--stage", "-z"])
        return [item.split(b"\t", 1)[1].decode() for item in raw.split(b"\0")
                if item.startswith((b"100644 ", b"100755 ", b"120000 "))]
    raw = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    )
    return [item.decode() for item in raw.split(b"\0") if item]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true', help='检查完整 Git index 的实际内容，包含未修改的已跟踪文件')
    args = parser.parse_args()
    failures: list[str] = []
    for name in repository_files(args.staged):
        if name == "scripts/check_privacy.py":
            continue
        if PRIVATE_PATHS.search(name):
            failures.append(f"private path is tracked: {name}")
            continue
        if args.staged:
            data = subprocess.check_output(['git', 'show', ':' + name])
        else:
            path = Path(name)
            if not path.is_file():
                continue
            data = path.read_bytes()
        if name == '.gitmodules':
            data = data.replace(b'git@github.com:', b'github.com:')
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
