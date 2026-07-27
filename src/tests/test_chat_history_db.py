"""聊天记录数据库 Phase 1 MVP 测试。"""

import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db import ChatHistoryRepository, init_db
from src.db.models import ChatMember, Chatroom, Message


@pytest.fixture
def repo():
    """提供一个内存数据库仓库。"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    init_db(db_path)
    repository = ChatHistoryRepository(db_path=db_path)
    yield repository
    db_path.unlink(missing_ok=True)


def test_init_db_creates_tables(repo: ChatHistoryRepository):
    engine = create_engine(f"sqlite:///{repo.db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        assert session.query(Chatroom).count() == 0
        assert session.query(Message).count() == 0
        assert session.query(ChatMember).count() == 0
    finally:
        session.close()


def test_sync_chat_creates_records(repo: ChatHistoryRepository):
    messages = [
        {
            "content": "hello",
            "wxid": "wxid_aaa",
            "sender_display_name": "Alice",
            "create_time": 1700000000.0,
            "message_type": "text",
        },
        {
            "content": "hi",
            "wxid": "wxid_bbb",
            "sender_display_name": "Bob",
            "create_time": 1700000001.0,
            "message_type": "text",
        },
    ]
    stats = repo.sync_chat(
        chatroom_id="20934380170@chatroom",
        display_name="共同富裕群",
        chat_type="group",
        messages=messages,
    )
    assert stats["chatrooms"] == 1
    assert stats["messages"] == 2
    assert stats["members"] == 2


def test_duplicate_messages_are_skipped(repo: ChatHistoryRepository):
    messages = [
        {
            "content": "same message",
            "wxid": "wxid_aaa",
            "sender_display_name": "Alice",
            "create_time": 1700000000.0,
            "message_type": "text",
        },
    ]
    repo.sync_chat(
        chatroom_id="20934380170@chatroom",
        display_name="共同富裕群",
        chat_type="group",
        messages=messages,
    )
    stats = repo.sync_chat(
        chatroom_id="20934380170@chatroom",
        display_name="共同富裕群",
        chat_type="group",
        messages=messages,
    )
    assert stats["messages"] == 0
    assert stats["skipped"] == 1


def test_bulk_sync_preserves_existing_reply_state(repo: ChatHistoryRepository):
    base = {
        "content": "same message",
        "wxid": "wxid_aaa",
        "sender_display_name": "Alice",
        "create_time": 1700000000.0,
        "message_type": "text",
        "replied": True,
        "reply_text": "done",
        "reply_time": 1700000001.0,
    }
    repo.bulk_sync_chat("room", "Alice", "single", [base])
    repo.bulk_sync_chat(
        "room",
        "Alice",
        "single",
        [{**base, "replied": False, "reply_text": None, "reply_time": None}],
    )

    stored = repo.get_messages("room")[0]
    assert stored.replied is True
    assert stored.reply_text == "done"
    assert stored.reply_time == 1700000001.0


def test_bulk_sync_accepts_sender_type_self(repo: ChatHistoryRepository):
    repo.bulk_sync_chat(
        "room",
        "Alice",
        "single",
        [{
            "content": "self message",
            "wxid": "wxid_self",
            "sender_display_name": "自己",
            "sender_type": "self",
            "create_time": 1700000000.0,
        }],
    )

    assert repo.get_messages("room")[0].is_self is True


def test_init_db_repairs_runtime_self_messages(repo: ChatHistoryRepository):
    repo.bulk_sync_chat(
        "room",
        "Alice",
        "single",
        [{
            "content": "legacy self message",
            "wxid": "wxid_self",
            "sender_display_name": "自己",
            "is_self": False,
            "create_time": 1700000000.0,
        }],
    )

    init_db(repo.db_path)

    assert repo.get_messages("room")[0].is_self is True


def test_same_name_groups_are_separated(repo: ChatHistoryRepository):
    """两个同名群按 chatroom_id 分开存储。"""
    group_a_msgs = [
        {
            "content": "有西西",
            "wxid": "wxid_xixi",
            "sender_display_name": "西西",
            "create_time": 1700000000.0,
            "message_type": "text",
        },
    ]
    group_b_msgs = [
        {
            "content": "没西西",
            "wxid": "wxid_other",
            "sender_display_name": "Other",
            "create_time": 1700000000.0,
            "message_type": "text",
        },
    ]
    repo.sync_chat(
        chatroom_id="room_a@chatroom",
        display_name="共同富裕群",
        chat_type="group",
        messages=group_a_msgs,
    )
    repo.sync_chat(
        chatroom_id="room_b@chatroom",
        display_name="共同富裕群",
        chat_type="group",
        messages=group_b_msgs,
    )

    a_msgs = repo.get_messages("room_a@chatroom")
    b_msgs = repo.get_messages("room_b@chatroom")
    assert len(a_msgs) == 1
    assert len(b_msgs) == 1
    assert a_msgs[0].content == "有西西"
    assert b_msgs[0].content == "没西西"


def test_get_messages_ordered_by_time(repo: ChatHistoryRepository):
    messages = [
        {
            "content": "second",
            "wxid": "wxid_aaa",
            "create_time": 1700000002.0,
            "message_type": "text",
        },
        {
            "content": "first",
            "wxid": "wxid_aaa",
            "create_time": 1700000001.0,
            "message_type": "text",
        },
        {
            "content": "third",
            "wxid": "wxid_aaa",
            "create_time": 1700000003.0,
            "message_type": "text",
        },
    ]
    repo.sync_chat(
        chatroom_id="room@chatroom",
        display_name="Test",
        chat_type="group",
        messages=messages,
    )
    result = repo.get_messages("room@chatroom")
    assert [m.content for m in result] == ["first", "second", "third"]


def test_sync_chat_without_chatroom_id_is_skipped(repo: ChatHistoryRepository):
    stats = repo.sync_chat(
        chatroom_id="",
        display_name="Test",
        chat_type="group",
        messages=[{"content": "hi", "wxid": "a", "create_time": 1.0, "message_type": "text"}],
    )
    assert stats["chatrooms"] == 0
    assert stats["messages"] == 0
