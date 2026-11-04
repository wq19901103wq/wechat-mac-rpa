#!/usr/bin/env python3
"""GlobalStore 单元测试"""

import os
import tempfile
import time
from dataclasses import replace

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

    def test_merge_tick_does_not_deduplicate_against_distant_history(self, store):
        name = "好友"
        history = [
            ChatMessage("难受", name, SenderType.OTHER, name),
            ChatMessage("难受", "自己", SenderType.SELF, name),
            ChatMessage("", name, SenderType.OTHER, name, message_type="image",
                        image_description="一个切开的榴莲，露出黄色的果肉"),
        ] + [ChatMessage(f"旧记录{i}", name, SenderType.OTHER, name) for i in range(60)]
        store.chats[name] = ChatState("friend", name, messages=list(history),
                                     _msg_ids={_msg_id(name, msg) for msg in history})
        tick = [
            ChatMessage("今天去新开的书店吗", name, SenderType.OTHER, name, observed_at=100),
            ChatMessage("难受", "自己", SenderType.SELF, name, observed_at=100),
            ChatMessage("刚好买到了想看的书", name, SenderType.OTHER, name, observed_at=100),
            ChatMessage("没事的", name, SenderType.OTHER, name, observed_at=100),
            ChatMessage("之后有空再去看看", name, SenderType.OTHER, name, observed_at=100),
            ChatMessage("下次再聊", name, SenderType.OTHER, name, observed_at=100),
            ChatMessage("", name, SenderType.OTHER, name, observed_at=100, message_type="image",
                        image_description="一个黄色的卡通牛头人形象，背景模糊"),
            ChatMessage("好了么", name, SenderType.OTHER, name, observed_at=100),
        ]

        state, _ = store.merge_tick(name, tick)
        assert state.messages[-8:] == tick
        state, _ = store.merge_tick(name, tick)
        assert len(state.messages) == len(history) + 8
        store.save()
        restored = GlobalStore(state_file=store._state_file)
        assert [(m.text, m.sender_type, m.image_description) for m in restored.chats[name].messages
                if m.observed_at == 100] == [(m.text, m.sender_type, m.image_description) for m in tick]

    @pytest.mark.parametrize("message_type", ["text", "image"])
    @pytest.mark.parametrize("sender_type", [SenderType.SELF, SenderType.OTHER])
    def test_merge_tick_fuzzy_dedup_respects_sender_role(self, store, message_type, sender_type):
        opposite = SenderType.OTHER if sender_type == SenderType.SELF else SenderType.SELF
        old = ChatMessage("难受", "自己" if opposite == SenderType.SELF else "好友", opposite, "好友",
                          message_type=message_type, image_description="一个黄色的卡通牛头人形象，背景模糊")
        new = ChatMessage("难受", "自己" if sender_type == SenderType.SELF else "好友", sender_type, "好友",
                          message_type=message_type, image_description=old.image_description)
        store.merge_tick("好友", [old])
        state, _ = store.merge_tick("好友", [new])
        assert [m.sender_type for m in state.messages] == [opposite, sender_type]

    @pytest.mark.parametrize('reverse', [False, True])
    def test_sticker_text_field_drift_does_not_trigger_another_reply(self, store, reverse):
        old = ChatMessage('', 'A', SenderType.OTHER, '群1', message_type='sticker',
                          image_description="一只柴犬/金毛犬的头像表情包，戴着墨镜，配文'顺带路过'",
                          image_text='顺带路过', observed_at=100)
        repeated = replace(old, text='顺带路过', image_text='',
                           image_description='一个戴墨镜的黄色卡通猫头表情包', observed_at=200)
        if reverse:
            old, repeated = repeated, old
        state, unreplied = store.merge_tick('群1', [old], is_group=True)
        store.mark_replied('群1', unreplied[0], '收到')
        state, unreplied = store.merge_tick('群1', [repeated], is_group=True)
        assert len(state.messages) == 1
        assert state.messages[0].replied
        assert unreplied == []

    @pytest.mark.parametrize('message_type', ['text', 'sticker'])
    def test_fuzzy_dedup_preserves_different_group_speakers(self, store, message_type):
        old = ChatMessage('顺带路过', 'A', SenderType.OTHER, '群1', message_type=message_type,
                          image_description='一只戴墨镜的黄色小狗', image_text='顺带路过')
        new = replace(old, sender='B', image_description='一个戴墨镜的黄色卡通猫头表情包')
        store.merge_tick('群1', [old], is_group=True)
        state, unreplied = store.merge_tick('群1', [new], is_group=True)
        assert [msg.sender for msg in state.messages] == ['A', 'B']
        assert [msg.sender for msg in unreplied] == ['A', 'B']

    def test_same_sticker_can_be_sent_again_after_later_messages(self, store):
        old = ChatMessage('', 'A', SenderType.OTHER, '群1', message_type='sticker',
                          image_description='一只戴墨镜的小狗', image_text='顺带路过')
        first = ChatMessage('到哪了', 'B', SenderType.OTHER, '群1')
        second = ChatMessage('正在楼下等你', '自己', SenderType.SELF, '群1')
        state, unreplied = store.merge_tick('群1', [old, first, second], is_group=True)
        store.mark_replied('群1', unreplied[0], '收到')
        store.mark_replied('群1', unreplied[1], '正在楼下等你')
        repeated = replace(old, text='顺带路过', image_text='', image_description='戴墨镜的猫头')
        state, unreplied = store.merge_tick('群1', [first, second, repeated], is_group=True)
        assert len(state.messages) == 4
        assert len(unreplied) == 1
        assert unreplied[0].image_description == '戴墨镜的猫头'

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

    @pytest.mark.parametrize('create_time', [None, 50])
    def test_first_observation_survives_repeat_and_restart(self, store, create_time):
        msg = ChatMessage('hi', 'A', SenderType.OTHER, '群1', observed_at=100.25, create_time=create_time)
        store.merge_tick('群1', [msg])
        store.save()
        restored = GlobalStore(state_file=store._state_file)
        repeated = ChatMessage('hi', 'A', SenderType.OTHER, '群1', observed_at=200)
        state, _ = restored.merge_tick('群1', [repeated])
        assert len(state.messages) == 1
        assert state.messages[0].observed_at == 100.25
        assert state.messages[0].create_time is None
        restored.save()
        assert GlobalStore(state_file=store._state_file).chats['群1'].messages[0].observed_at == 100.25

    def test_legacy_database_adds_observation_column_without_inventing_time(self, store):
        from sqlalchemy import text
        from src.db import init_db

        store.merge_tick('群1', [ChatMessage('hi', 'A', SenderType.OTHER, '群1', create_time=50)])
        store.save()
        with store._chat_repo._session() as session:
            engine = session.get_bind()
        with engine.begin() as conn:
            conn.execute(text('ALTER TABLE messages DROP COLUMN observed_at'))
        init_db(engine.url.database)
        init_db(engine.url.database)
        msg = GlobalStore(state_file=store._state_file).chats['群1'].messages[0]
        assert msg.observed_at is None
        assert msg.create_time == 50

    def test_self_sender_type_survives_persistence(self, store):
        msg = ChatMessage(
            text="我发的",
            sender="自己",
            sender_type=SenderType.SELF,
            chat_name="群1",
            chatroom_id="room1",
            sender_wxid="wxid_self",
            create_time=1700000000,
        )
        store.merge_tick("群1", [msg])
        store.save()

        store2 = GlobalStore(state_file=store._state_file)

        assert store2.chats["群1"].messages[0].sender_type == SenderType.SELF

    def test_failed_db_sync_keeps_dirty_for_retry(self, store):
        msg = ChatMessage(
            text="hi",
            sender="A",
            sender_type=SenderType.OTHER,
            chat_name="群1",
            create_time=1700000000,
        )
        store.merge_tick("群1", [msg])

        class FailingRepo:
            def resolve_chatroom_id(self, chat_name, messages):
                return "room1"

            def bulk_sync_chat(self, **kwargs):
                raise RuntimeError("db locked")

        store._chat_repo = FailingRepo()
        store.save()

        assert "群1" in store._dirty

    def test_save_cleans_expired_screenshots_without_new_capture(self, store):
        screenshot = store._screenshot_store._screenshots_dir / "old.png"
        screenshot.write_bytes(b"old")
        old = time.time() - 90000
        os.utime(screenshot, (old, old))
        store._screenshot_store._last_cleanup = 0

        store.save()

        assert not screenshot.exists()

    def test_replied_state_updates_existing_db_message(self, store):
        """消息先以未回复入库后，二次保存应更新状态，重启不能复活。"""
        msg = ChatMessage(
            text="hi",
            sender="A",
            sender_type=SenderType.OTHER,
            chat_name="群1",
            create_time=1700000000.0,
        )
        store.merge_tick("群1", [msg])
        store.save()

        store.mark_replied("群1", msg, "ok")
        store.save()

        store2 = GlobalStore(state_file=store._state_file)
        state, unreplied = store2.merge_tick("群1", [])
        assert unreplied == []
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
        hist = ChatMessage(text="hello", sender="示例用户午", sender_type=SenderType.OTHER, chat_name="示例用户午")
        store.merge_tick("示例用户午", [hist])
        assert len(store.chats["示例用户午"].messages) == 1

        # tick 中 API 返回的是 "对方"
        tick = ChatMessage(text="hello", sender="对方", sender_type=SenderType.OTHER, chat_name="示例用户午")
        state, unreplied = store.merge_tick("示例用户午", [tick])
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
                sender="示例用户午" if i % 2 == 0 else "自己",
                sender_type=SenderType.OTHER if i % 2 == 0 else SenderType.SELF,
                chat_name="示例用户午",
            )
            history.append(msg)
        store.merge_tick("示例用户午", history)
        assert len(store.chats["示例用户午"].messages) == 10

        # tick 只显示历史中间 3 条（索引 3-5）
        tick = history[3:6]
        state, unreplied = store.merge_tick("示例用户午", tick)
        # 不应添加任何新消息
        assert len(state.messages) == 10

    def test_merge_tick_prefix_match_new_suffix(self, store):
        """tick 前缀匹配历史末尾，后缀是新消息"""
        # 历史：5 条旧消息
        history = []
        for i in range(5):
            msg = ChatMessage(
                text=f"msg{i}",
                sender="示例用户午" if i % 2 == 0 else "自己",
                sender_type=SenderType.OTHER if i % 2 == 0 else SenderType.SELF,
                chat_name="示例用户午",
            )
            history.append(msg)
        store.merge_tick("示例用户午", history)

        # tick：前 3 条是历史末尾，后 2 条是新的
        tick = [
            ChatMessage(text="msg3", sender="自己", sender_type=SenderType.SELF, chat_name="示例用户午"),
            ChatMessage(text="msg4", sender="示例用户午", sender_type=SenderType.OTHER, chat_name="示例用户午"),
            ChatMessage(text="new1", sender="示例用户午", sender_type=SenderType.OTHER, chat_name="示例用户午"),
            ChatMessage(text="new2", sender="自己", sender_type=SenderType.SELF, chat_name="示例用户午"),
        ]
        state, unreplied = store.merge_tick("示例用户午", tick)
        # 应添加 2 条新消息
        assert len(state.messages) == 7
        assert state.messages[-2].text == "new1"
        assert state.messages[-1].text == "new2"
        # 未回复列表包含所有对方消息（包括旧的历史中未回复的）
        assert len(unreplied) == 4  # msg0, msg2, msg4, new1


class TestIsGroupChatName:
    def test_chinese_parentheses(self):
        assert _is_group_chat_name("示例交流群（128）") is True
        assert _is_group_chat_name("周末兴趣群（5）") is True

    def test_english_parentheses(self):
        assert _is_group_chat_name("周末兴趣群 (5)") is True
        assert _is_group_chat_name("示例交流群 (128)") is True

    def test_private_chat_returns_false(self):
        assert _is_group_chat_name("示例用户甲") is False
        assert _is_group_chat_name("示例用户午") is False
        assert _is_group_chat_name("") is False


class TestGlobalStoreDbSync:
    """GlobalStore 同步到 SQLite 的回归测试。"""

    def test_source_chatroom_id_syncs_to_db(self):
        """消息自带 chatroom_id，直接按 chatroom_id 入库。"""
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
