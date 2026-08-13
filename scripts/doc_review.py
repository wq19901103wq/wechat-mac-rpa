#!/usr/bin/env python3
"""
文档增强审查脚本 — 覆盖 doc_lint.py 的盲区。
检查项：
1. API_SURFACE.md 的可复制粘贴性（import 完整性、类型定义）
2. ARCHITECTURE.md ↔ API_SURFACE.md 核心接口一致性
3. 文档内常量定义（如 TIMESTAMP_PATTERNS）
4. 回调/事件构造函数赋值与主循环调用的一致性
"""

import re
import sys
from pathlib import Path

# 默认在项目根目录运行；如果未找到，则尝试向上查找 wechat-mac-rpa
ROOT = Path.cwd()
ARCH_CANDIDATES = [
    ROOT / "docs" / "02-architecture" / "ARCHITECTURE.md",
    ROOT / "ARCHITECTURE.md",
]
API_CANDIDATES = [
    ROOT / "docs" / "02-architecture" / "API_SURFACE.md",
    ROOT / "API_SURFACE.md",
]

ARCH = None
for cand in ARCH_CANDIDATES:
    if cand.exists():
        ARCH = cand
        break
if ARCH is None:
    for parent in Path(__file__).resolve().parents:
        for cand in [
            parent / "docs" / "02-architecture" / "ARCHITECTURE.md",
            parent / "ARCHITECTURE.md",
        ]:
            if cand.exists():
                ROOT = parent
                ARCH = cand
                break
        if ARCH:
            break

API = None
for cand in API_CANDIDATES:
    if cand.exists():
        API = cand
        break
if API is None and ARCH:
    API = ARCH.parent / "API_SURFACE.md"

errors = []
warnings = []


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_api_surface_imports():
    """检查 API_SURFACE.md 是否有必要的 import。"""
    text = read_text(API)
    used = set()

    # 扫描使用的标识符
    for token in re.findall(r"\b(datetime|Callable|Dict|List|Optional|Tuple|Enum|dataclass)\b", text):
        used.add(token)

    has_import_block = bool(re.search(r"^from\s+\w+\s+import|^import\s+\w+", text, re.M))
    import_text = "\n".join(re.findall(r"^from\s+\w+\s+import.*$|^import\s+\w+.*$", text, re.M))

    if used and not has_import_block:
        errors.append("API_SURFACE.md 使用了类型标识符但没有 import 语句块。")
        return

    required = {
        "datetime": r"from\s+datetime\s+import",
        "Callable": r"from\s+typing\s+import.*Callable",
        "Dict": r"from\s+typing\s+import.*Dict",
        "List": r"from\s+typing\s+import.*List",
        "Optional": r"from\s+typing\s+import.*Optional",
        "Tuple": r"from\s+typing\s+import.*Tuple",
        "Enum": r"from\s+enum\s+import",
        "dataclass": r"from\s+dataclasses\s+import",
    }

    for token, pattern in required.items():
        if token in used:
            if not re.search(pattern, import_text):
                errors.append(f"API_SURFACE.md 使用了 `{token}` 但缺少对应的 import。")


def check_api_surface_types_defined():
    """检查 API_SURFACE.md 中使用的自定义类型是否都有定义。"""
    text = read_text(API)

    # 收集所有 class/dataclass 定义
    defined = set(re.findall(r"(?:^@dataclass\(?:.*\)\n)?^class\s+(\w+)", text, re.M))

    # 扫描所有类型引用（仅限代码块内）
    refs = set()
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```python"):
            in_code = True
            continue
        if line.strip().startswith("```"):
            in_code = False
            continue
        if not in_code:
            continue
        # : Type 或 -> Type
        for m in re.finditer(r"[:\(]\s*(\w+)(?:\[|\s|$)", line):
            refs.add(m.group(1))
        for m in re.finditer(r"->\s*(\w+)(?:\[|\s|:|$)", line):
            refs.add(m.group(1))

    primitives = {
        "int", "float", "str", "bool", "None", "object", "dict", "list", "tuple",
        "Exception", "Optional", "List", "Dict", "Tuple", "Callable", "datetime",
        "Enum", "self", "cls", "pass", "return", "import", "from", "def", "class",
        "True", "False",
    }

    for ref in sorted(refs):
        if ref in primitives:
            continue
        if ref not in defined:
            errors.append(f"API_SURFACE.md 代码块中使用了未定义的类型 `{ref}`。")


def check_constants_defined():
    """检查 ARCHITECTURE.md 代码块中引用的大写常量是否有定义。"""
    text = read_text(ARCH)

    # 只检查代码块内的大写标识符
    in_code = False
    constants = set()
    for line in text.splitlines():
        if line.strip().startswith("```python"):
            in_code = True
            continue
        if line.strip().startswith("```"):
            in_code = False
            continue
        if not in_code:
            continue
        constants |= set(re.findall(r"\b([A-Z][A-Z_0-9]+)\b", line))

    # 排除常见的 Python 内置/通用常量
    builtins = {"OCR", "UI", "API", "LLM", "IME", "JSONL", "CI", "URL", "TODO", "RGB", "L1", "L2", "L3", "L4", "L5"}
    for c in sorted(constants - builtins):
        # 检查是否有赋值定义：CONST = ... 或 TIMESTAMP_PATTERNS = [...]
        if not re.search(rf"\b{c}\s*[:=]", text):
            errors.append(f"ARCHITECTURE.md 代码块中引用了常量 `{c}` 但文档内没有定义。")


def check_cross_doc_signature_consistency():
    """检查两文档中核心类方法的签名是否一致。"""
    arch = read_text(ARCH)
    api = read_text(API)

    key_methods = [
        ("filter_new", r"def filter_new\(self.*?\).*?->.*?:"),
        ("record_sent", r"def record_sent\(self.*?\).*?->.*?:"),
        ("is_in_cooldown", r"def is_in_cooldown\(self.*?\).*?->.*?:"),
        ("send_to_chat", r"def send_to_chat\(self.*?\).*?->.*?:"),
        ("switch_chat", r"def switch_chat\(self.*?\).*?->.*?:"),
        ("_get_session", r"def _get_session\(self.*?\).*?->.*?:"),
    ]

    for name, pattern in key_methods:
        arch_match = re.search(pattern, arch)
        api_match = re.search(pattern, api)
        if arch_match and not api_match:
            errors.append(f"ARCHITECTURE.md 有 `{name}`，但 API_SURFACE.md 中缺失。")
        elif api_match and not arch_match:
            warnings.append(f"API_SURFACE.md 有 `{name}`，但 ARCHITECTURE.md 中缺失（可能是新增预留接口）。")


def check_callback_consistency():
    """检查 on_message 回调：构造函数赋值 + 主循环调用。"""
    arch = read_text(ARCH)

    has_init_param = bool(re.search(r"def __init__\(.*on_message", arch))
    has_self_assignment = bool(re.search(r"self\.on_message\s*=\s*on_message", arch))
    has_tick_call = bool(re.search(r"self\.on_message\([^)]+\)", arch))

    if has_init_param and not has_self_assignment:
        errors.append("ARCHITECTURE.md 中 `on_message` 是构造参数，但没有 `self.on_message = on_message` 赋值。")
    if has_self_assignment and not has_tick_call:
        errors.append("ARCHITECTURE.md 中 `on_message` 已赋值，但 `tick()` 主循环中没有调用它。")


def main():
    check_api_surface_imports()
    check_api_surface_types_defined()
    check_constants_defined()
    check_cross_doc_signature_consistency()
    check_callback_consistency()

    print("=" * 60)
    print("文档增强审查报告")
    print("=" * 60)

    if errors:
        print(f"\n🔴 错误（{len(errors)} 项）：")
        for e in errors:
            print(f"  - {e}")

    if warnings:
        print(f"\n🟡 警告（{len(warnings)} 项）：")
        for w in warnings:
            print(f"  - {w}")

    if not errors and not warnings:
        print("\n✅ 增强审查通过")

    print("=" * 60)

    if errors:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
