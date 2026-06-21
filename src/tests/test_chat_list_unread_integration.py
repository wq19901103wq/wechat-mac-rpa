#!/usr/bin/env python3
"""ChatList 未读角标识别集成测试（真实 API 调用）

验证 prompt 修复后，qwen3.6-flash 不再把群聊拼贴头像误判为未读角标。

需要 DASHSCOPE_API_KEY 环境变量。
默认跳过（不消耗 API 额度），手动触发：
    pytest src/tests/test_chat_list_unread_integration.py -v --run-integration
"""

import os
from pathlib import Path

import pytest

from src.perception.smart_pipeline import _QwenAPIClient

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "unread_false_positive"


@pytest.mark.skip(
    reason="集成测试，手动触发：pytest --run-integration 或去掉本 skip"
)
def test_group_avatar_not_unread_badge_with_real_api():
    """
    用真实 qwen3.6-flash API 识别 fixture 截图，
    断言"王芊 @ai开发小分队"的未读角标为空字符串。
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        pytest.skip("需要 DASHSCOPE_API_KEY 环境变量")

    client = _QwenAPIClient(api_key=api_key)
    screenshot_path = FIXTURE_DIR / "screenshot.png"
    assert screenshot_path.exists(), f"截图不存在: {screenshot_path}"

    result = client.recognize(str(screenshot_path))

    chat_list = result.get("chat_list", [])
    assert chat_list, "API 未返回 chat_list"

    target = next(
        (item for item in chat_list if "王芊" in item.get("nickname", "")),
        None,
    )
    assert target is not None, (
        f"未在 API 返回的 chat_list 中找到目标聊天。"
        f"返回的 chat_list: {[i.get('nickname') for i in chat_list]}"
    )

    assert target.get("unread_count", "") == "", (
        f"API 把群聊拼贴头像误判为未读角标: "
        f"nickname={target.get('nickname')!r}, "
        f"unread_count={target.get('unread_count')!r}, 期望=''"
    )
