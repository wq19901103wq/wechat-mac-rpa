"""Windows 本地数据读取层测试。

使用构造的明文 SQLite fixture（无需微信进程/真实密钥），覆盖：
list_contacts / get_sessions / get_messages 的查询、排序、分页、跨库合并、
ZSTD 解码与发送者解析，以及 config / message_codec 的纯逻辑。
"""

import hashlib
import sqlite3
from pathlib import Path

import pytest
import zstandard

from src.platform.windows.config import ENV_CACHE_DIR, ENV_DATA_DIR, get_cache_dir, get_db_storage_dir
from src.platform.windows.message_codec import (
    decode_message_content,
    decompress_content,
    split_sender_prefix,
)
from src.platform.windows.wechat_data import WeChatData

TALKER = "wxid_fixture_test"
TABLE = "Msg_" + hashlib.md5(TALKER.encode("utf-8"), usedforsecurity=False).hexdigest()
ZSTD = zstandard.ZstdCompressor().compress


def _zstd(text: str) -> bytes:
    return ZSTD(text.encode("utf-8"))


@pytest.fixture()
def decrypted_dir(tmp_path: Path) -> Path:
    """构造与真实微信 4.1 schema 一致的明文库目录。"""
    decrypted = tmp_path / "decrypted"
    (decrypted / "contact").mkdir(parents=True)
    (decrypted / "session").mkdir(parents=True)
    (decrypted / "message").mkdir(parents=True)

    conn = sqlite3.connect(decrypted / "contact" / "contact.db")
    conn.execute(
        "CREATE TABLE contact (username TEXT, alias TEXT, remark TEXT, nick_name TEXT, local_type INTEGER)"
    )
    conn.executemany(
        "INSERT INTO contact VALUES (?,?,?,?,?)",
        [
            ("wxid_fixture_test", "", "备注A", "昵称A", 0),
            ("group1@chatroom", "", "", "群聊一", 2),
            ("gh_abc123", "aliasX", "", "公众号", 1),
            ("", "", "", "空用户名", 0),
        ],
    )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(decrypted / "session" / "session.db")
    conn.execute(
        "CREATE TABLE SessionTable (username TEXT, unread_count INTEGER, summary TEXT, "
        "last_timestamp INTEGER, sort_timestamp INTEGER, last_msg_locald_id INTEGER, "
        "last_msg_type INTEGER, last_msg_sub_type INTEGER, last_msg_sender TEXT, "
        "last_sender_display_name TEXT)"
    )
    conn.execute(
        "INSERT INTO SessionTable VALUES ('wxid_fixture_test', 3, '你好', 1000, 2000, "
        "7, 1, 0, 'wxid_fixture_test', '昵称A')"
    )
    conn.commit()
    conn.close()

    # message_0：明文 + ZSTD 压缩；message_1：跨分片一条（排序验证）
    message_0 = decrypted / "message" / "message_0.db"
    conn = sqlite3.connect(message_0)
    conn.execute("CREATE TABLE Name2Id (user_name TEXT, is_session INTEGER)")
    conn.execute("INSERT INTO Name2Id VALUES ('sender_a', 0)")  # rowid=1
    conn.execute(
        f"CREATE TABLE {TABLE} (local_id INTEGER, server_id INTEGER, local_type INTEGER, "
        "sort_seq INTEGER, real_sender_id INTEGER, create_time INTEGER, "
        "message_content BLOB, packed_info_data BLOB)"
    )
    conn.executemany(
        f"INSERT INTO {TABLE} VALUES (?,?,?,?,?,?,?,?)",  # nosec B608
        [
            (1, 1001, 1, 100, 1, 1000, "sender_a:\n你好世界".encode("utf-8"), b"\x08\x01"),
            (2, 1002, 1, 200, 1, 2000, _zstd("sender_a:\n压缩消息"), None),
        ],
    )
    conn.commit()
    conn.close()

    message_1 = decrypted / "message" / "message_1.db"
    conn = sqlite3.connect(message_1)
    conn.execute("CREATE TABLE Name2Id (user_name TEXT, is_session INTEGER)")
    conn.execute("INSERT INTO Name2Id VALUES ('sender_b', 0)")  # rowid=1
    conn.execute(
        f"CREATE TABLE {TABLE} (local_id INTEGER, server_id INTEGER, local_type INTEGER, "
        "sort_seq INTEGER, real_sender_id INTEGER, create_time INTEGER, "
        "message_content BLOB, packed_info_data BLOB)"
    )
    conn.execute(
        f"INSERT INTO {TABLE} VALUES (?,?,?,?,?,?,?,?)",  # nosec B608
        (3, 1003, 1, 300, 1, 3000, "sender_b:\n跨库消息".encode("utf-8"), None),
    )
    conn.commit()
    conn.close()
    return decrypted


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------

def test_get_db_storage_dir_env_override(tmp_path, monkeypatch):
    target = tmp_path / "db_storage"
    target.mkdir()
    monkeypatch.setenv(ENV_DATA_DIR, str(target))
    assert get_db_storage_dir() == target


def test_get_db_storage_dir_env_invalid_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path / "not_exist"))
    detected = tmp_path / "detected"
    detected.mkdir()
    monkeypatch.setattr(
        "src.platform.windows.config.auto_detect_db_storage_dir", lambda: detected
    )
    assert get_db_storage_dir() == detected


def test_get_cache_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_CACHE_DIR, str(tmp_path / "cache"))
    assert get_cache_dir() == tmp_path / "cache"


# ----------------------------------------------------------------------
# message_codec
# ----------------------------------------------------------------------

def test_decompress_content_only_zstd():
    raw = b"plain text"
    assert decompress_content(raw) == raw
    compressed = _zstd("hello")
    assert decompress_content(compressed) == b"hello"


def test_decode_message_content_bytes_and_none():
    assert decode_message_content(None) == ""
    assert decode_message_content(b"abc") == "abc"
    assert decode_message_content(_zstd("你好")) == "你好"
    assert decode_message_content("str") == "str"


def test_split_sender_prefix_with_known_senders():
    known = {"sender_a"}
    sender, body = split_sender_prefix("sender_a:\nhello", known)
    assert (sender, body) == ("sender_a", "hello")
    sender, body = split_sender_prefix("正文含冒号:\n但不拆", known)
    assert sender is None and body == "正文含冒号:\n但不拆"


def test_split_sender_prefix_common_pattern():
    sender, body = split_sender_prefix("wxid_abc123:\n你好", None)
    assert (sender, body) == ("wxid_abc123", "你好")


# ----------------------------------------------------------------------
# wechat_data 查询层（decrypted_dir_override 直查，不触发提 key）
# ----------------------------------------------------------------------

def test_list_contacts(decrypted_dir):
    client = WeChatData(decrypted_dir_override=decrypted_dir)
    contacts = client.list_contacts()
    by_name = {c.username: c for c in contacts}
    assert by_name["wxid_fixture_test"].display_name == "备注A"  # remark 优先
    assert by_name["group1@chatroom"].is_group
    assert by_name["gh_abc123"].alias == "aliasX"
    assert by_name["gh_abc123"].local_type == 1
    assert "" not in by_name  # 空用户名被过滤


def test_get_sessions(decrypted_dir):
    client = WeChatData(decrypted_dir_override=decrypted_dir)
    sessions = client.get_sessions()
    assert len(sessions) == 1
    s = sessions[0]
    assert s.username == "wxid_fixture_test"
    assert s.unread_count == 3
    assert s.summary == "你好"
    assert s.name == "昵称A"


def test_get_messages_sorted_and_decoded(decrypted_dir):
    client = WeChatData(decrypted_dir_override=decrypted_dir)
    messages = client.get_messages(TALKER)
    assert [m.create_time for m in messages] == [1000, 2000, 3000]  # 跨库按时间升序
    assert messages[0].content == "你好世界"
    assert messages[0].raw_content == "sender_a:\n你好世界"
    assert messages[0].sender_username == "sender_a"
    assert messages[1].content == "压缩消息"  # ZSTD 已解压
    assert messages[2].content == "跨库消息"
    assert messages[2].sender_username == "sender_b"


def test_get_messages_pagination(decrypted_dir):
    client = WeChatData(decrypted_dir_override=decrypted_dir)
    page = client.get_messages(TALKER, limit=2, offset=1)
    assert [m.create_time for m in page] == [2000, 3000]
    assert client.get_messages(TALKER, limit=1, offset=10) == []


def test_get_messages_unknown_talker(decrypted_dir):
    client = WeChatData(decrypted_dir_override=decrypted_dir)
    assert client.get_messages("nobody@chatroom") == []


def test_get_messages_negative_limit_rejected(decrypted_dir):
    client = WeChatData(decrypted_dir_override=decrypted_dir)
    with pytest.raises(ValueError):
        client.get_messages(TALKER, limit=-1)
