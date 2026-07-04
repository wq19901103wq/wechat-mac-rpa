#!/usr/bin/env python3
"""GlobalStore 单元测试"""

import os
import tempfile

import pytest

from src.models.base import ChatMessage, SenderType
from src.session.global_store import ChatState, GlobalStore, _is_group_chat_name, _msg_id


class TestMsgId:
    def test_msg_id_deterministic(self):
        """相同消息生成相同 ID"""
        msg = ChatMessage(text="hello", sender="A", sender_type=SenderType.OTHER, chat_name="群1")
        assert _msg_id("群1", msg) == _msg_id("群1", msg)

    def test_msg_id_differs_by_chat(self):
        """不同聊天生成不同 ID"""
        m1 = ChatMessage(text="hi", sender="A", sender_type=SenderType.OTHER, chat_name="群1")
        m2 = ChatMessage(text="hi", sender="A", sender_type=SenderType.OTHER, chat_name="群2")
        assert _msg_id("群1", m1) != _msg_id("群2", m2)


class TestGlobalStore:
    @pytest.fixture
    def store(self):
        # 使用独立临时目录，避免 save() 的分片文件（chats/index.json）
        # 污染系统临时目录导致测试间状态串扰
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            yield GlobalStore(state_file=path)

    def test_merge_tick_new_messages(self, store):
        """首次 merge 返回所有消息为未回复"""
        msg = ChatMessage(text="hi", sender="A", sender_type=SenderType.OTHER, chat_name="群1")
        state, unreplied = store.merge_tick("群1", [msg])
        assert len(unreplied) == 1
        assert unreplied[0].text == "hi"
        assert isinstance(state, ChatState)

    def test_merge_tick_deduplication(self, store):
        """相同消息多次 merge 不重复堆积（消息体只存一份，但未回复的仍会返回）"""
        msg = ChatMessage(text="hi", sender="A", sender_type=SenderType.OTHER, chat_name="群1")
        store.merge_tick("群1", [msg])
        state, unreplied = store.merge_tick("群1", [msg])
        # 消息不重复堆积
        assert len(state.messages) == 1
        # 但未回复的遗留消息仍会返回
        assert len(unreplied) == 1
        assert unreplied[0].text == "hi"

    def test_merge_tick_same_tick_duplicates(self, store):
        """同一 tick 内传入重复消息只保留一条"""
        msg = ChatMessage(text="hi", sender="A", sender_type=SenderType.OTHER, chat_name="群1")
        state, unreplied = store.merge_tick("群1", [msg, msg, msg])
        assert len(unreplied) == 1
        assert len(state.messages) == 1

    def test_merge_tick_excludes_self(self, store):
        """自己消息不算未回复"""
        msg = ChatMessage(text="ok", sender="me", sender_type=SenderType.SELF, chat_name="群1")
        state, unreplied = store.merge_tick("群1", [msg])
        assert len(unreplied) == 0

    def test_mark_replied(self, store):
        """标记回复后消息不再出现在未回复列表"""
        msg = ChatMessage(text="hi", sender="A", sender_type=SenderType.OTHER, chat_name="群1")
        store.merge_tick("群1", [msg])
        store.mark_replied("群1", msg, "收到")

        state, unreplied = store.merge_tick("群1", [])
        assert len(unreplied) == 0
        assert state.messages[0].replied is True
        assert state.messages[0].reply_text == "收到"

    def test_get_unreplied_ordered(self, store):
        """未回复消息按时间顺序返回"""
        m1 = ChatMessage(text="a", sender="A", sender_type=SenderType.OTHER, chat_name="群1")
        m2 = ChatMessage(text="b", sender="B", sender_type=SenderType.OTHER, chat_name="群1")
        store.merge_tick("群1", [m1])
        store.merge_tick("群1", [m2])
        unreplied = store.get_unreplied("群1")
        assert [m.text for m in unreplied] == ["a", "b"]

    def test_max_messages_limit(self, store):
        """超过 max_messages 裁剪旧消息"""
        store.max_messages = 3
        for i in range(5):
            msg = ChatMessage(text=str(i), sender="A", sender_type=SenderType.OTHER, chat_name="群1")
            store.merge_tick("群1", [msg])
        state, _ = store.merge_tick("群1", [])
        assert len(state.messages) == 3
        assert state.messages[0].text == "2"

    def test_persistence_roundtrip(self, store):
        """持久化后加载能恢复状态"""
        msg = ChatMessage(text="hi", sender="A", sender_type=SenderType.OTHER, chat_name="群1")
        store.merge_tick("群1", [msg])
        store.mark_replied("群1", msg, "ok")
        store.save()

        # 重新加载
        store2 = GlobalStore(state_file=store._state_file)
        state, unreplied = store2.merge_tick("群1", [])
        assert len(unreplied) == 0
        assert len(state.messages) == 1
        assert state.messages[0].replied is True
        assert state.messages[0].reply_text == "ok"

    def test_reply_count_and_last_reply_time(self, store):
        """从消息推导回复统计"""
        msg = ChatMessage(text="hi", sender="A", sender_type=SenderType.OTHER, chat_name="群1")
        store.merge_tick("群1", [msg])
        assert store.reply_count("群1") == 0
        assert store.last_reply_time("群1") is None

        store.mark_replied("群1", msg, "收到")
        assert store.reply_count("群1") == 1
        assert store.last_reply_time("群1") is not None

    def test_multiple_chats_isolated(self, store):
        """不同聊天互不影响"""
        m1 = ChatMessage(text="a", sender="A", sender_type=SenderType.OTHER, chat_name="群1")
        m2 = ChatMessage(text="b", sender="B", sender_type=SenderType.OTHER, chat_name="群2")
        store.merge_tick("群1", [m1])
        store.merge_tick("群2", [m2])
        assert len(store.get_unreplied("群1")) == 1
        assert len(store.get_unreplied("群2")) == 1

    def test_merge_tick_sender_normalization(self, store):
        """tick 中 sender='对方' 能匹配历史中 sender='昵称'"""
        # 历史存的是昵称
        hist = ChatMessage(text="hello", sender="秋水文章", sender_type=SenderType.OTHER, chat_name="秋水文章")
        store.merge_tick("秋水文章", [hist])
        assert len(store.chats["秋水文章"].messages) == 1

        # tick 中 API 返回的是 "对方"
        tick = ChatMessage(text="hello", sender="对方", sender_type=SenderType.OTHER, chat_name="秋水文章")
        state, unreplied = store.merge_tick("秋水文章", [tick])
        # 不应重复添加
        assert len(state.messages) == 1
        # 未回复列表仍返回该消息（因为它本来就未回复）
        assert len(unreplied) == 1

    def test_merge_tick_scroll_no_new_messages(self, store):
        """用户向上滚动，tick 显示历史中间段，不应产生新消息"""
        # 构建历史：10 条消息
        history = []
        for i in range(10):
            msg = ChatMessage(
                text=f"msg{i}",
                sender="秋水文章" if i % 2 == 0 else "自己",
                sender_type=SenderType.OTHER if i % 2 == 0 else SenderType.SELF,
                chat_name="秋水文章",
            )
            history.append(msg)
        store.merge_tick("秋水文章", history)
        assert len(store.chats["秋水文章"].messages) == 10

        # tick 只显示历史中间 3 条（索引 3-5）
        tick = history[3:6]
        state, unreplied = store.merge_tick("秋水文章", tick)
        # 不应添加任何新消息
        assert len(state.messages) == 10

    def test_merge_tick_prefix_match_new_suffix(self, store):
        """tick 前缀匹配历史末尾，后缀是新消息"""
        # 历史：5 条旧消息
        history = []
        for i in range(5):
            msg = ChatMessage(
                text=f"msg{i}",
                sender="秋水文章" if i % 2 == 0 else "自己",
                sender_type=SenderType.OTHER if i % 2 == 0 else SenderType.SELF,
                chat_name="秋水文章",
            )
            history.append(msg)
        store.merge_tick("秋水文章", history)

        # tick：前 3 条是历史末尾，后 2 条是新的
        tick = [
            ChatMessage(text="msg3", sender="自己", sender_type=SenderType.SELF, chat_name="秋水文章"),
            ChatMessage(text="msg4", sender="秋水文章", sender_type=SenderType.OTHER, chat_name="秋水文章"),
            ChatMessage(text="new1", sender="秋水文章", sender_type=SenderType.OTHER, chat_name="秋水文章"),
            ChatMessage(text="new2", sender="自己", sender_type=SenderType.SELF, chat_name="秋水文章"),
        ]
        state, unreplied = store.merge_tick("秋水文章", tick)
        # 应添加 2 条新消息
        assert len(state.messages) == 7
        assert state.messages[-2].text == "new1"
        assert state.messages[-1].text == "new2"
        # 未回复列表包含所有对方消息（包括旧的历史中未回复的）
        assert len(unreplied) == 4  # msg0, msg2, msg4, new1

    def test_merge_tick_weflow_bot_reply_time_update(self, store):
        """_merge_tick_weflow 对 bot 重复消息应更新 reply_time 而不是简单跳过"""
        # 使用独立 chat_name，避免与其他测试残留数据冲突
        # 模拟 WeFlow 返回的 bot 历史消息（有 local_id，无 reply_time）
        weflow_bot = ChatMessage(
            text="same",
            sender="自己",
            sender_type=SenderType.SELF,
            chat_name="weflow_test",
            local_id=100,
        )
        state, _ = store.merge_tick("weflow_test", [weflow_bot], mode="weflow")
        assert len(state.messages) == 1
        assert state.messages[0].reply_time is None

        # 模拟 bot 手动注入的相同内容消息（无 local_id，有 reply_time）
        manual_bot = ChatMessage(
            text="same",
            sender="bot",
            sender_type=SenderType.SELF,
            chat_name="weflow_test",
            reply_time=99999.0,
        )
        state, _ = store.merge_tick("weflow_test", [manual_bot], mode="weflow")
        # 不应新增消息，但应更新已有消息的 reply_time
        assert len(state.messages) == 1
        assert state.messages[0].reply_time == 99999.0


class TestIsGroupChatName:
    def test_chinese_parentheses(self):
        assert _is_group_chat_name("ai开发小分队（128）") is True
        assert _is_group_chat_name("王老板们和小天才（5）") is True

    def test_english_parentheses(self):
        assert _is_group_chat_name("王老板们和小天才 (5)") is True
        assert _is_group_chat_name("ai开发小分队 (128)") is True

    def test_private_chat_returns_false(self):
        assert _is_group_chat_name("W1han") is False
        assert _is_group_chat_name("秋水文章") is False
        assert _is_group_chat_name("") is False


class TestGlobalStoreDbSync:
    """GlobalStore 同步到 SQLite 的回归测试。"""

    def test_weflow_mode_syncs_with_chatroom_id(self):
        """WeFlow 消息自带 chatroom_id，直接按 chatroom_id 入库。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = os.path.abspath(tmpdir)
            state_file = os.path.join(tmpdir, "state.json")
            db_path = os.path.join(tmpdir, "chat_history.db")

            store = GlobalStore(state_file=state_file)
            from src.db import ChatHistoryRepository, init_db
            init_db(db_path)
            store._chat_repo = ChatHistoryRepository(db_path=db_path)

            msg = ChatMessage(
                text="hello",
                sender="Alice",
                sender_type=SenderType.OTHER,
                chat_name="TestGroup",
                chatroom_id="room_real@chatroom",
            )
            store.merge_tick("TestGroup", [msg], is_group=True)
            store.save()

            repo = ChatHistoryRepository(db_path=db_path)
            result = repo.get_messages("room_real@chatroom")
            assert len(result) == 1
            assert result[0].content == "hello"

    def test_ocr_mode_falls_back_to_db_display_name(self):
        """OCR 消息没有 chatroom_id 时，按 display_name 从 DB 反查。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = os.path.abspath(tmpdir)
            state_file = os.path.join(tmpdir, "state.json")
            db_path = os.path.join(tmpdir, "chat_history.db")

            from src.db import ChatHistoryRepository, init_db
            init_db(db_path)
            repo = ChatHistoryRepository(db_path=db_path)
            repo.bulk_sync_chat(
                chatroom_id="room_real@chatroom",
                display_name="TestGroup",
                chat_type="group",
                messages=[{
                    "content": "old msg",
                    "wxid": "wxid_old",
                    "sender_display_name": "Old",
                    "create_time": 1700000000.0,
                    "message_type": "text",
                }],
            )

            store = GlobalStore(state_file=state_file)
            store._chat_repo = ChatHistoryRepository(db_path=db_path)

            # OCR 消息：没有 chatroom_id，也没有 sender_wxid
            msg = ChatMessage(
                text="new msg",
                sender="Alice",
                sender_type=SenderType.OTHER,
                chat_name="TestGroup",
            )
            store.merge_tick("TestGroup", [msg], is_group=True)
            store.save()

            result = repo.get_messages("room_real@chatroom")
            contents = [m.content for m in result]
            assert "old msg" in contents
            assert "new msg" in contents

    def test_db_sync_uses_bulk_insert_not_per_message_select(self):
        """GlobalStore.save() 应该调 bulk_sync_chat，而不是逐条 SELECT。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = os.path.abspath(tmpdir)
            state_file = os.path.join(tmpdir, "state.json")
            db_path = os.path.join(tmpdir, "chat_history.db")

            from src.db import ChatHistoryRepository, init_db
            init_db(db_path)

            store = GlobalStore(state_file=state_file)
            repo = ChatHistoryRepository(db_path=db_path)
            store._chat_repo = repo

            # 模拟：替换 repo.sync_chat 和 repo.bulk_sync_chat，断言调用的是后者
            calls = []
            repo.sync_chat = lambda **kwargs: calls.append("sync_chat") or {"messages": 0}
            repo.bulk_sync_chat = lambda **kwargs: calls.append("bulk_sync_chat") or {"messages": len(kwargs.get("messages", []))}

            msg = ChatMessage(
                text="bulk msg",
                sender="Alice",
                sender_type=SenderType.OTHER,
                chat_name="BulkGroup",
                chatroom_id="bulk_room@chatroom",
            )
            store.merge_tick("BulkGroup", [msg], is_group=True)
            store.save()

            assert "bulk_sync_chat" in calls
            assert "sync_chat" not in calls
