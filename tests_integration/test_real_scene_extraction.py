#!/usr/bin/env python3
"""
真实场景回归测试：基于 wechat_capture.png 的消息提取准确性验证。

这个测试不依赖真实微信窗口，使用 fixtures/real_login_recovered_scene.png
断言消息提取结果必须符合预期，防止时间戳/噪声被误识别为消息。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr.vision_ocr import VisionOCREngine
from src.layout.layout_parser import LayoutParser
from src.layout.profile import PROFILE_WECHAT_MAC_1760X1280
from src.message.extractor import MessageExtractor

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "real_login_recovered_scene.png"


def test_real_scene_message_extraction():
    assert FIXTURE_PATH.exists(), f"fixture 不存在: {FIXTURE_PATH}"

    ocr = VisionOCREngine()
    elements = ocr.recognize(str(FIXTURE_PATH))

    profile = PROFILE_WECHAT_MAC_1760X1280
    layout = LayoutParser(profile).parse(elements, str(FIXTURE_PATH))
    messages = MessageExtractor(profile).extract(layout)

    # 1. 聊天名称必须正确
    assert layout.chat_name == "周末兴趣群（5)", f"聊天名称错误: {layout.chat_name!r}"

    # 2. 时间戳不能被识别为消息
    for msg in messages:
        assert "星期" not in msg.text, f"时间戳被误识别为消息: {msg.text!r}"
        assert "23:58" not in msg.text, f"时间戳被误识别为消息: {msg.text!r}"

    # 3. 标题栏/界面噪声不能被识别为消息
    all_texts = [m.text for m in messages]
    assert "®v (D." not in all_texts, "标题栏噪声被误识别为消息"
    assert "-" not in all_texts, "界面噪声被误识别为消息"

    # 4. 必须正确提取 wanglc 的消息
    wanglc_msgs = [m for m in messages if "是不是忙着切号呢" in m.text]
    assert len(wanglc_msgs) == 1, "wanglc 的消息丢失或重复"
    assert wanglc_msgs[0].sender == "wanglc", f"wanglc 发件人识别错误: {wanglc_msgs[0].sender!r}"

    # 5. 必须正确提取自己的长消息（不应被截断）
    self_msgs = [m for m in messages if m.sender == "自己"]
    long_self = [m for m in self_msgs if "哈哈误会啦！" in m.text]
    assert len(long_self) == 1, "自己的长消息丢失或重复"
    assert "回复速度取决于网络/服务器负载" in long_self[0].text, "长消息内容被截断"
    assert "还有登录问题需要解决吗？随时叫我！" in long_self[0].text, "长消息尾部被截断"

    # 6. 必须包含最后一条自己的短消息
    short_self = [m for m in self_msgs if "laayaua5aapangaaaaa~" in m.text]
    assert len(short_self) == 1, "最后一条自己的短消息丢失"

    # 7. 总消息数必须正确（不应包含噪声/时间戳）
    # 期望: wanglc 消息 + 自己长消息 + 自己短消息 = 3
    assert len(messages) == 3, f"消息数异常，期望 3，实际 {len(messages)}: {[m.text[:30] for m in messages]}"


if __name__ == "__main__":
    test_real_scene_message_extraction()
    print("✅ test_real_scene_message_extraction 通过")
