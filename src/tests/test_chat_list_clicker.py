#!/usr/bin/env python3
"""ChatListClicker 单元测试"""

from unittest.mock import MagicMock, patch

import pytest

from src.action.chat_list_clicker import ChatListClicker
from src.action.system_automation import SystemAutomation
from src.models.base import ChatListItem, Rect


@pytest.fixture(autouse=True)
def reset_click_cooldown(monkeypatch):
    monkeypatch.setattr(ChatListClicker, "_last_click_time", 0.0)


def _make_mock_automation() -> SystemAutomation:
    """返回一个所有 UI 操作都成功的 mock SystemAutomation。"""
    mock = MagicMock(spec=SystemAutomation)
    mock.get_frontmost_app.return_value = (True, "WeChat")
    mock.click_at.return_value = True
    return mock


class TestChatListClicker:
    def test_click_cooldown_is_shared_between_instances(self):
        window_rect = Rect(x=0, y=0, width=100, height=100)
        automation = _make_mock_automation()
        first = ChatListClicker(window_rect, automation=automation)
        second = ChatListClicker(window_rect, automation=automation)
        item = ChatListItem(nickname="test", last_message_preview="", unread_count="", timestamp="", rect=Rect(x=10, y=10, width=20, height=20))

        with patch("wechat_reader.action.chat_list_clicker.time") as clock:
            clock.time.return_value = 100.0
            assert first.click_item(item) is True
            clock.time.return_value = 100.5
            assert second.click_item(item) is False
            automation.click_at.assert_called_once()
            clock.time.return_value = 101.0
            assert second.click_item(item) is True

        assert automation.click_at.call_count == 2

    def test_click_item_coordinates(self):
        """验证坐标计算：屏幕绝对坐标 = 窗口逻辑坐标 + 截图像素坐标 / scale_factor"""
        window_rect = Rect(x=100, y=200, width=800, height=600)
        automation = _make_mock_automation()
        clicker = ChatListClicker(window_rect, scale_factor=2.0, automation=automation)

        item = ChatListItem(
            nickname="测试",
            last_message_preview="",
            unread_count="",
            timestamp="",
            rect=Rect(x=200, y=300, width=100, height=50),
        )

        clicker.click_item(item)

        # click_x = 200 + 100//2 = 250, click_y = 300 + 50//2 = 325
        # abs_x = 100 + 250/2 = 225, abs_y = 200 + 325/2 = 362
        automation.click_at.assert_called_once_with(225, 362)

    def test_click_item_returns_true_on_success(self):
        window_rect = Rect(x=0, y=0, width=100, height=100)
        automation = _make_mock_automation()
        clicker = ChatListClicker(window_rect, scale_factor=2.0, automation=automation)
        item = ChatListItem(nickname="test", last_message_preview="", unread_count="", timestamp="", rect=Rect(x=10, y=10, width=20, height=20))

        result = clicker.click_item(item)
        assert result is True

    def test_click_item_returns_false_on_failure(self):
        window_rect = Rect(x=0, y=0, width=100, height=100)
        automation = _make_mock_automation()
        automation.click_at.return_value = False
        clicker = ChatListClicker(window_rect, scale_factor=2.0, automation=automation)
        item = ChatListItem(nickname="test", last_message_preview="", unread_count="", timestamp="", rect=Rect(x=10, y=10, width=20, height=20))

        result = clicker.click_item(item)
        assert result is False

    def test_click_by_index_valid(self):
        window_rect = Rect(x=0, y=0, width=100, height=100)
        automation = _make_mock_automation()
        clicker = ChatListClicker(window_rect, scale_factor=2.0, automation=automation)
        items = [
            ChatListItem(nickname="a", last_message_preview="", unread_count="", timestamp="", rect=Rect(x=0, y=0, width=10, height=10)),
            ChatListItem(nickname="b", last_message_preview="", unread_count="", timestamp="", rect=Rect(x=0, y=10, width=10, height=10)),
        ]
        result = clicker.click_by_index(items, 1)
        assert result is True

    def test_click_by_index_out_of_range(self):
        window_rect = Rect(x=0, y=0, width=100, height=100)
        automation = _make_mock_automation()
        clicker = ChatListClicker(window_rect, scale_factor=2.0, automation=automation)
        items = [ChatListItem(nickname="a", last_message_preview="", unread_count="", timestamp="", rect=Rect(x=0, y=0, width=10, height=10))]
        result = clicker.click_by_index(items, 5)
        assert result is False

    def test_click_first_unread_found(self):
        window_rect = Rect(x=0, y=0, width=100, height=100)
        automation = _make_mock_automation()
        clicker = ChatListClicker(window_rect, scale_factor=2.0, automation=automation)
        items = [
            ChatListItem(nickname="a", last_message_preview="", unread_count="", timestamp="", rect=Rect(x=0, y=0, width=10, height=10)),
            ChatListItem(nickname="b", last_message_preview="", unread_count="3", timestamp="", rect=Rect(x=0, y=10, width=10, height=10)),
        ]
        result = clicker.click_first_unread(items)
        assert result is not None
        assert result.nickname == "b"

    def test_click_first_unread_excludes_current(self):
        window_rect = Rect(x=0, y=0, width=100, height=100)
        automation = _make_mock_automation()
        clicker = ChatListClicker(window_rect, scale_factor=2.0, automation=automation)
        items = [
            ChatListItem(nickname="current", last_message_preview="", unread_count="3", timestamp="", rect=Rect(x=0, y=0, width=10, height=10)),
            ChatListItem(nickname="other", last_message_preview="", unread_count="5", timestamp="", rect=Rect(x=0, y=10, width=10, height=10)),
        ]
        result = clicker.click_first_unread(items, exclude_nickname="current")
        assert result is not None
        assert result.nickname == "other"

    def test_click_first_unread_none_found(self):
        window_rect = Rect(x=0, y=0, width=100, height=100)
        automation = _make_mock_automation()
        clicker = ChatListClicker(window_rect, scale_factor=2.0, automation=automation)
        items = [
            ChatListItem(nickname="a", last_message_preview="", unread_count="", timestamp="", rect=Rect(x=0, y=0, width=10, height=10)),
        ]
        result = clicker.click_first_unread(items)
        assert result is None
