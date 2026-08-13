#!/usr/bin/env python3
"""L1 Domain Models 单元测试"""

import pytest

from src.models.base import (
    ActionResult,
    ChatListItem,
    ChatMessage,
    OCRTextElement,
    PerceptionResult,
    Point,
    Rect,
    SenderType,
)


class TestPoint:
    def test_creation(self):
        p = Point(x=10, y=20)
        assert p.x == 10
        assert p.y == 20

    def test_immutable(self):
        p = Point(x=10, y=20)
        with pytest.raises(AttributeError):
            p.x = 30


class TestRect:
    def test_creation(self):
        r = Rect(x=0, y=0, width=100, height=50)
        assert r.width == 100
        assert r.height == 50

    def test_immutable(self):
        r = Rect(x=0, y=0, width=100, height=50)
        with pytest.raises(AttributeError):
            r.width = 200


class TestOCRTextElement:
    def test_creation(self):
        elem = OCRTextElement(
            text="hello",
            bbox=Rect(0, 0, 50, 20),
            center=Point(25, 10),
            confidence=0.95
        )
        assert elem.text == "hello"
        assert elem.confidence == pytest.approx(0.95)


class TestChatMessage:
    def test_creation_minimal(self):
        msg = ChatMessage(
            text="hi",
            sender="Alice",
            sender_type=SenderType.OTHER,
            chat_name="测试群"
        )
        assert msg.text == "hi"
        assert msg.is_at_me is False
        assert msg.timestamp is None

    def test_creation_full(self):
        elem = OCRTextElement(
            text="hi", bbox=Rect(0,0,10,10), center=Point(5,5), confidence=0.9
        )
        msg = ChatMessage(
            text="hi",
            sender="Alice",
            sender_type=SenderType.OTHER,
            chat_name="测试群",
            is_at_me=True,
            timestamp="12:34",
            source_elements=[elem]
        )
        assert msg.is_at_me is True
        assert msg.timestamp == "12:34"
        assert len(msg.source_elements) == 1


class TestActionResult:
    def test_success(self):
        ar = ActionResult(success=True, sent_text="收到")
        assert ar.success is True
        assert ar.sent_text == "收到"
        assert ar.error is None

    def test_failure(self):
        ar = ActionResult(success=False, error="timeout")
        assert ar.success is False
        assert ar.error == "timeout"


class TestChatListItem:
    def test_creation(self):
        item = ChatListItem(
            nickname="示例用户酉",
            last_message_preview="在吗",
            unread_count="1",
            timestamp="12:34",
            rect=Rect(0, 0, 300, 60)
        )
        assert item.nickname == "示例用户酉"
        assert item.rect.width == 300


class TestPerceptionResult:
    def test_creation(self, tmp_path):
        image_path = str(tmp_path / "test.png")
        msg = ChatMessage(
            text="hi", sender="Alice", sender_type=SenderType.OTHER, chat_name="测试群"
        )
        item = ChatListItem(
            nickname="示例用户酉", last_message_preview="...", unread_count="", timestamp="",
            rect=Rect(0,0,300,60)
        )
        pr = PerceptionResult(
            chat_name="测试群",
            messages=[msg],
            chat_list_items=[item],
            screenshot_path=image_path
        )
        assert pr.chat_name == "测试群"
        assert len(pr.messages) == 1
        assert len(pr.chat_list_items) == 1
        assert pr.screenshot_path == image_path
