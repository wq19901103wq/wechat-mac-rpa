#!/usr/bin/env python3
"""
文档自洽性轻量检查器。
读取 ARCHITECTURE.md 和 API_SURFACE.md，捕获明显的接口不一致、
未定义标识符、命名漂移和跨文档签名 mismatch。

使用方式:
    python3 scripts/doc_lint.py

退出码:
    0 = 全绿
    1 = 发现不一致
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCH_CANDIDATES = [
    PROJECT_ROOT / "docs" / "02-architecture" / "ARCHITECTURE.md",
    PROJECT_ROOT / "ARCHITECTURE.md",
]
API_CANDIDATES = [
    PROJECT_ROOT / "docs" / "02-architecture" / "API_SURFACE.md",
    PROJECT_ROOT / "API_SURFACE.md",
]

ARCH_PATH = None
for cand in ARCH_CANDIDATES:
    if cand.exists():
        ARCH_PATH = cand
        break

API_PATH = None
for cand in API_CANDIDATES:
    if cand.exists():
        API_PATH = cand
        break

# 已知的“黑盒函数”黑名单：如果出现在文档中，说明文档引用了未定义的东西
UNDEFINED_BLACKLIST = {
    "estimate_y",
    "nickname_x",
}

# 已知的命名漂移对：同一文档中不应同时出现
NAMING_DRIFT_PAIRS = [
    ("OCREngine", "VisionOCREngine"),
]


@dataclass
class Violation:
    file: str
    line: int
    message: str


def read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


WORD_RE = {token: re.compile(rf"\b{re.escape(token)}\b") for token in UNDEFINED_BLACKLIST}


def check_blacklist(lines: List[str], filename: str) -> List[Violation]:
    violations = []
    for i, line in enumerate(lines, 1):
        for bad in UNDEFINED_BLACKLIST:
            if WORD_RE[bad].search(line):
                violations.append(Violation(filename, i, f"发现未定义标识符: `{bad}`"))
    return violations


def check_naming_drift(lines: List[str], filename: str) -> List[Violation]:
    violations = []
    text = "\n".join(lines)
    for a, b in NAMING_DRIFT_PAIRS:
        if re.search(rf"\b{re.escape(a)}\b", text) and re.search(rf"\b{re.escape(b)}\b", text):
            violations.append(
                Violation(filename, 0, f"命名漂移: 文档中同时出现 `{a}` 和 `{b}`，必须统一")
            )
    return violations


# 正则：提取类定义/函数定义（支持类型注解和默认参数）
DEF_RE = re.compile(
    r"^\s*(?:class|def)\s+(\w+)\s*\((.*?)\)",
    re.MULTILINE | re.DOTALL,
)
# 正则：提取调用（简单启发式：word( ）
CALL_RE = re.compile(r"(\w+)\s*\((.*?)\)", re.DOTALL)


def extract_signatures(text: str) -> Dict[str, int]:
    """从文本中提取 class/def 签名，返回 {name: 参数个数}。"""
    sigs = {}
    for m in DEF_RE.finditer(text):
        name = m.group(1)
        params_str = m.group(2)
        # 简单参数计数：按逗号分，去掉 self/cls 和空字符串
        raw_parts = [p.strip() for p in params_str.split(",") if p.strip()]
        # 过滤掉仅包含 self/cls 的项（保留带类型的 self）
        parts = []
        for p in raw_parts:
            if p in ("self", "cls"):
                continue
            parts.append(p)
        sigs[name] = len(parts)
    return sigs


def extract_calls(text: str) -> List[Tuple[str, int, str]]:
    """提取函数调用，返回 [(name, 参数个数, 所在行)]。"""
    calls = []
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        for m in CALL_RE.finditer(line):
            name = m.group(1)
            args_str = m.group(2).strip()
            start = m.start()
            # 跳过 self.method() 中的 self
            if name == "self":
                continue
            # 跳过函数/类定义本身：def foo(...) / class Foo(...)
            prefix = line[:start].rstrip()
            if prefix.endswith("def") or prefix.endswith("class"):
                continue
            # 跳过外部模块调用如 subprocess.run(), time.sleep()
            if start > 0 and line[start - 1] == ".":
                continue
            # 排除 if/for/while/except 等关键字后的括号
            if name in ("if", "for", "while", "except", "class", "def", "print", "len", "range", "import", "from"):
                continue
            # 简单计数参数（不处理嵌套括号，够用即可）
            if not args_str:
                arg_count = 0
            else:
                # 用逗号分隔，但忽略字符串里的逗号
                # 简化：只要没引号就按逗号分
                if '"' not in args_str and "'" not in args_str:
                    arg_count = len([p for p in args_str.split(",") if p.strip()])
                else:
                    # 保守估计：把整个字符串当一个参数
                    arg_count = 1
            calls.append((name, arg_count, f"{name}({args_str})"))
    return calls


def check_internal_call_consistency(lines: List[str], filename: str) -> List[Violation]:
    """检查文档内函数定义和调用示例的参数数量是否明显 mismatch。"""
    text = "\n".join(lines)
    sigs = extract_signatures(text)
    calls = extract_calls(text)
    violations = []
    for name, arg_count, snippet in calls:
        if name not in sigs:
            continue
        expected = sigs[name]
        # 允许默认值导致的参数少传；但传多了一定是错误
        if arg_count > expected:
            violations.append(
                Violation(
                    filename,
                    0,
                    f"调用参数过多: `{snippet}` 传了 {arg_count} 个参数，但 `{name}` 定义只有 {expected} 个非 self/cls 参数",
                )
            )
    return violations


def extract_api_surface_signatures(text: str) -> Dict[str, str]:
    """从 API_SURFACE.md 中提取简洁签名字符串，用于跨文档比对。"""
    sigs = {}
    for m in DEF_RE.finditer(text):
        name = m.group(1)
        params = m.group(2).strip()
        sigs[name] = params
    return sigs


def check_cross_doc_consistency(arch_text: str, api_text: str) -> List[Violation]:
    """比对 ARCHITECTURE.md 和 API_SURFACE.md 的关键接口签名。"""
    violations = []

    arch_sigs = extract_api_surface_signatures(arch_text)
    api_sigs = extract_api_surface_signatures(api_text)

    # 关键接口白名单
    key_interfaces = [
        "VisionPipeline",
        "run",
        "switch_chat",
        "ChatSession",
        "ChatListItem",
        "filter_new",
        "record_sent",
        "_hash_messages",
        "_is_echo",
        "_is_seen_with_context",
        "is_in_cooldown",
        "UILayout",
        "MessageExtractor",
        "extract",
        "LayoutParser",
        "parse",
        "VisionOCREngine",
        "recognize",
        "WindowCapture",
        "capture",
        "MessageSender",
        "send",
        "send_image",
        "send_file",
        "WeChatMessageSender",
        "UIInteractor",
        "click_chat_item",
        "click_input_box",
        "_get_session",
        "send_to_chat",
    ]

    for name in key_interfaces:
        arch_sig = arch_sigs.get(name)
        api_sig = api_sigs.get(name)
        if arch_sig and api_sig:
            # 简单规范化：去掉空格、类型注解 -> 统一
            def norm(s: str) -> str:
                return s.replace(" ", "").replace("->", "").replace("None:", "").replace("...", "").replace(":", "")

            if norm(arch_sig) != norm(api_sig):
                violations.append(
                    Violation(
                        "ARCHITECTURE.md ↔ API_SURFACE.md",
                        0,
                        f"签名不一致: `{name}`\n  ARCH: ({arch_sig})\n  API : ({api_sig})",
                    )
                )

    return violations


def main() -> int:
    if not ARCH_PATH.exists():
        print(f"❌ 找不到 {ARCH_PATH}")
        return 1
    if not API_PATH.exists():
        print(f"❌ 找不到 {API_PATH}")
        return 1

    arch_lines = read_lines(ARCH_PATH)
    api_lines = read_lines(API_PATH)
    arch_text = "\n".join(arch_lines)
    api_text = "\n".join(api_lines)

    all_violations: List[Violation] = []

    # 1. 黑名单检查
    all_violations.extend(check_blacklist(arch_lines, "ARCHITECTURE.md"))
    all_violations.extend(check_blacklist(api_lines, "API_SURFACE.md"))

    # 2. 命名漂移检查
    all_violations.extend(check_naming_drift(arch_lines, "ARCHITECTURE.md"))
    all_violations.extend(check_naming_drift(api_lines, "API_SURFACE.md"))

    # 3. 文档内部调用一致性
    all_violations.extend(check_internal_call_consistency(arch_lines, "ARCHITECTURE.md"))

    # 4. 跨文档签名一致性
    all_violations.extend(check_cross_doc_consistency(arch_text, api_text))

    if not all_violations:
        print("✅ 文档自洽检查通过")
        return 0

    print(f"❌ 发现 {len(all_violations)} 处不一致:\n")
    for v in all_violations:
        loc = f"{v.file}:{v.line}" if v.line else v.file
        print(f"  [{loc}] {v.message}")
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
