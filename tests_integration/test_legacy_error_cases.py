#!/usr/bin/env python3
"""
历史错误案例回归测试

使用新架构 VisionPipeline 对 legacy/errors/ 下的所有历史错误截图进行端到端验证。
这些截图来自旧 V4 架构时期收集的真实错误案例，现在用新架构重新跑一遍并硬断言结果。
"""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.capture.window_capture import CaptureResult
from src.layout.profile import PROFILE_WECHAT_MAC_1760X1280
from src.models.base import Rect
from src.perception.vision_pipeline import VisionPipeline

ERRORS_DIR = Path(__file__).parent / "fixtures" / "legacy" / "errors"


def _load_case(name: str):
    png = ERRORS_DIR / f"{name}.png"
    meta = ERRORS_DIR / f"{name}.json"
    if not png.exists():
        pytest.skip(f"{name}.png not found")
    if not meta.exists():
        pytest.skip(f"{name}.json not found")
    return png, json.loads(meta.read_text(encoding="utf-8"))


def _run_pipeline(png_path: Path):
    pipeline = VisionPipeline(PROFILE_WECHAT_MAC_1760X1280)
    mock_capture = Mock()
    mock_capture.capture.return_value = CaptureResult(
        image_path=str(png_path),
        window_rect=Rect(0, 0, 1760, 1280),
        scale_factor=1.0,
    )
    pipeline.capture = mock_capture
    return pipeline.perceive()


# ═══════════════════════════════════════════════════════════
# OCR 容错与相似度比较
# ═══════════════════════════════════════════════════════════

OCR_ERROR_MAP = {
    "Al 助手": "AI 助手",
    "Al助手": "AI助手",
}

SIMILARITY_THRESHOLD = 0.90


def _normalize_ocr_errors(text: str) -> str:
    """对常见 OCR 错误进行归一化，预期 JSON 保持正确写法。"""
    for wrong, correct in OCR_ERROR_MAP.items():
        text = text.replace(wrong, correct)
    return text


def _text_similarity(a: str, b: str) -> float:
    """基于最长公共子序列的相似度 (0-1)。"""
    a = a.replace(" ", "").replace("\n", "")
    b = b.replace(" ", "").replace("\n", "")
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    return 2 * lcs / (m + n)


def _assert_forbidden(result, forbidden_texts):
    all_texts = [m.text for m in result.messages]
    for text in forbidden_texts:
        for actual in all_texts:
            assert text not in actual, f"禁止出现的文本出现在消息中: {text!r} (消息: {actual!r})"


def _assert_messages(result, expected_messages):
    assert len(result.messages) == len(expected_messages), (
        f"消息数量不匹配: 期望 {len(expected_messages)}, 实际 {len(result.messages)}"
    )
    for i, exp in enumerate(expected_messages):
        actual = result.messages[i]
        assert actual.sender_type.value == exp["sender_type"], (
            f"[{i}] 发送者类型不匹配: 期望 {exp['sender_type']}, 实际 {actual.sender_type.value}"
        )
        check = exp.get("check", "exact")
        expected_text = exp["text"]
        normalized_actual = _normalize_ocr_errors(actual.text)

        if check == "similarity":
            sim = _text_similarity(expected_text, normalized_actual)
            assert sim >= SIMILARITY_THRESHOLD, (
                f"[{i}] 消息相似度低于 {SIMILARITY_THRESHOLD:.0%}: "
                f"相似度={sim:.1%}, 期望={expected_text!r}, 实际={normalized_actual!r}"
            )
        elif check == "contains":
            assert expected_text in normalized_actual, (
                f"[{i}] 消息内容不包含期望文本: 期望包含 {expected_text!r}, 实际 {normalized_actual!r}"
            )
        else:
            assert normalized_actual == expected_text or expected_text in normalized_actual, (
                f"[{i}] 消息内容不匹配: 期望 {expected_text!r}, 实际 {normalized_actual!r}"
            )


# ═══════════════════════════════════════════════════════════
# 动态生成测试
# ═══════════════════════════════════════════════════════════


def _make_test(name: str):
    def _test():
        png, meta = _load_case(name)
        result = _run_pipeline(png)
        assert result is not None, "感知结果不应为 None"

        # 1. 聊天名称
        expected_chat_name = meta["chat_name"]
        assert result.chat_name == expected_chat_name, (
            f"聊天名称不匹配: 期望 {expected_chat_name!r}, 实际 {result.chat_name!r}"
        )

        # 2. 禁止出现的文本（噪声/乱码）
        _assert_forbidden(result, meta.get("forbidden_texts", []))

        # 3. 消息列表
        _assert_messages(result, meta.get("messages", []))

    _test.__name__ = f"test_{name}"
    _test.__doc__ = f"历史错误案例: {name}"
    return _test


# 自动注册所有历史错误案例
for png in sorted(ERRORS_DIR.glob("error_*.png")):
    case_name = png.stem
    globals()[f"test_{case_name}"] = _make_test(case_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
