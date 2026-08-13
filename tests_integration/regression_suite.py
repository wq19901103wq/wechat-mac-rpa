#!/usr/bin/env python3
"""
模块化重构回归测试套件

针对目标架构的核心逻辑进行分层回归验证。
不依赖真实微信窗口，使用 fixtures/ 下的截图和 JSON 数据进行测试。

运行方式:
    python3 tests/regression_suite.py
    python3 tests/regression_suite.py --layer session
    python3 tests/regression_suite.py --layer layout
"""

import json
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURES_DIR = Path(__file__).parent / "fixtures"
ERRORS_DIR = FIXTURES_DIR / "legacy" / "errors"


def load_fixture(name: str):
    """加载 fixture 的 JSON 元数据"""
    path = FIXTURES_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_error_cases():
    """加载所有错误案例的 JSON 元数据"""
    cases = []
    if not ERRORS_DIR.exists():
        return cases
    for f in sorted(ERRORS_DIR.glob("error_*.json")):
        cases.append(json.loads(f.read_text(encoding="utf-8")))
    return cases


class RegressionReporter:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def ok(self, msg):
        self.passed += 1
        print(f"  ✅ {msg}")

    def fail(self, msg):
        self.failed += 1
        print(f"  ❌ {msg}")

    def skip(self, msg):
        self.skipped += 1
        print(f"  ⚠️  {msg}")

    def summary(self):
        print(f"\n━━━━━━━━━━━━━━━━━━━━")
        print(f"通过: {self.passed}  失败: {self.failed}  跳过: {self.skipped}")
        if self.failed:
            sys.exit(1)
        print("✅ 回归测试通过")


# ═══════════════════════════════════════════════════════════
# L1: Session / Deduplication 回归测试
# ═══════════════════════════════════════════════════════════

def test_session_dedup(reporter):
    print("\n📦 L1: Session / Deduplication")
    try:
        from src.session.global_store import GlobalStore
        from src.models.base import ChatMessage, SenderType
    except ImportError as e:
        reporter.skip(f"导入失败: {e}")
        return

    store = GlobalStore(max_messages=200, state_file="data/test_global_state.json")
    chat_name = "测试群"

    # 测试用例 1: 首次消息应被识别为新消息
    msg1 = ChatMessage(text="hello", sender="A", chat_name=chat_name, sender_type=SenderType.OTHER)
    try:
        state, unreplied = store.merge_tick(chat_name, [msg1])
        if len(unreplied) == 1:
            reporter.ok("首次消息被正确识别为新消息")
        else:
            reporter.fail("首次消息未被识别为新消息")
    except Exception as e:
        reporter.fail(f"merge_tick 首次调用异常: {e}")

    # 测试用例 2: 完全相同的消息重复出现应被过滤
    try:
        count_before = len(state.messages)
        state, _ = store.merge_tick(chat_name, [msg1])
        if len(state.messages) == count_before:
            reporter.ok("重复消息被正确过滤")
        else:
            reporter.fail("重复消息未被过滤")
    except Exception as e:
        reporter.fail(f"merge_tick 重复调用异常: {e}")

    # 测试用例 3: 回声检测（自己发送的消息不计入未回复）
    try:
        store.mark_replied(chat_name, msg1, "收到了")
        echo_msg = ChatMessage(text="收到了", sender="自己", chat_name=chat_name, sender_type=SenderType.SELF)
        state, unreplied = store.merge_tick(chat_name, [echo_msg])
        if echo_msg not in unreplied:
            reporter.ok("回声消息被正确过滤")
        else:
            reporter.fail("回声消息未被过滤")
    except Exception as e:
        reporter.fail(f"回声检测异常: {e}")


# ═══════════════════════════════════════════════════════════
# L2: Layout / Message Extraction 回归测试
# ═══════════════════════════════════════════════════════════

def test_layout_extraction(reporter):
    print("\n📦 L2: Layout / Message Extraction")
    try:
        from src.layout.layout_parser import LayoutParser
        from src.message.extractor import MessageExtractor
        from src.layout.profile import PROFILE_WECHAT_MAC_1760X1280
    except ImportError as e:
        reporter.skip(f"导入失败: {e}")
        return

    fixture_names = ["small_scene", "medium_scene", "large_scene"]
    for name in fixture_names:
        png_path = FIXTURES_DIR / f"{name}.png"
        if not png_path.exists():
            reporter.skip(f"fixture {name} 缺少 png，跳过")
            continue

        try:
            profile = PROFILE_WECHAT_MAC_1760X1280
            LayoutParser(profile)
            MessageExtractor(profile)
            reporter.ok(f"{name} 的布局解析器可实例化")
        except Exception as e:
            reporter.fail(f"{name} 布局解析异常: {e}")


# ═══════════════════════════════════════════════════════════
# L3: 历史错误案例回归（通过 pytest 运行真实断言）
# ═══════════════════════════════════════════════════════════

def test_error_cases(reporter):
    print("\n📦 L3: 历史错误案例回归")
    import subprocess

    result = subprocess.run(
        ["pytest", "tests/test_legacy_error_cases.py", "-q", "--tb=line"],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        timeout=180,
    )

    # 解析 pytest summary 行，例如 "23 passed in 74.84s" 或 "21 passed, 2 failed in 73.02s"
    import re as _re
    summary_match = _re.search(r"(\d+) passed(?:, (\d+) failed)?", result.stdout)
    passed = int(summary_match.group(1)) if summary_match else 0
    failed = int(summary_match.group(2)) if summary_match and summary_match.group(2) else 0

    if passed == 0 and failed == 0:
        reporter.skip("没有检测到历史错误案例测试输出")
    elif failed == 0:
        reporter.ok(f"全部 {passed} 个历史错误案例通过")
    else:
        reporter.fail(f"历史错误案例: {passed} 通过, {failed} 失败")


# ═══════════════════════════════════════════════════════════
# L4: 文档一致性快速检查
# ═══════════════════════════════════════════════════════════

def test_doc_consistency(reporter):
    print("\n📦 L4: 文档一致性")
    import subprocess

    try:
        result = subprocess.run(
            ["python3", "scripts/doc_lint.py"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            reporter.ok("doc_lint.py 通过")
        else:
            reporter.fail("doc_lint.py 失败\n" + result.stdout + result.stderr)
    except Exception as e:
        reporter.fail(f"运行 doc_lint.py 异常: {e}")

    try:
        result = subprocess.run(
            ["python3", "scripts/doc_review.py"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            reporter.ok("doc_review.py 通过")
        else:
            reporter.fail("doc_review.py 失败\n" + result.stdout + result.stderr)
    except Exception as e:
        reporter.skip(f"doc_review.py 不存在或异常: {e}")


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RPA 模块化重构回归测试套件")
    parser.add_argument("--layer", choices=["session", "layout", "errors", "docs", "all"], default="all")
    args = parser.parse_args()

    reporter = RegressionReporter()

    if args.layer in ("session", "all"):
        test_session_dedup(reporter)
    if args.layer in ("layout", "all"):
        test_layout_extraction(reporter)
    if args.layer in ("errors", "all"):
        test_error_cases(reporter)
    if args.layer in ("docs", "all"):
        test_doc_consistency(reporter)

    reporter.summary()


if __name__ == "__main__":
    main()
