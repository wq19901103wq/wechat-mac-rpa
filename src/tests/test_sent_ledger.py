import json
import sqlite3

import pytest

from src.reply.sent_ledger import SentReplyLedger, load_sent_replies


@pytest.fixture
def ledger(tmp_path):
    path = tmp_path / 'cases.db'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE tick_log (id INTEGER PRIMARY KEY, created_at TEXT, '
                           'session_input_messages_json TEXT, replies_sent_json TEXT, send_success INTEGER)')

    def insert(identity, replies, success=1, created='2026-07-06 12:00:00'):
        inputs = identity if isinstance(identity, list) else [dict(chatroom_id=identity)]
        with sqlite3.connect(path) as connection:
            connection.execute('INSERT INTO tick_log (created_at,session_input_messages_json,'
                               'replies_sent_json,send_success) VALUES (?,?,?,?)',
                               (created, json.dumps(inputs), json.dumps(replies), success))
        return path

    return path, insert


def test_confirmed_and_partial_sends_are_ordered_without_mutating_database(ledger):
    path, insert = ledger
    insert('group-1', ['later'], created='2026-07-06 12:01:00')
    insert('group-1', ['first', 'second'], success=0)
    insert('group-1', ['not confirmed'], success=None)
    insert('group-1', [])
    before = path.read_bytes()
    sends = load_sent_replies(path)
    assert [s['text'] for s in sends] == ['first', 'second', 'later']
    assert all(s['source_chat_id'] == 'group-1' for s in sends)
    assert sends[0]['timestamp'] == 1783310400
    assert sends[-1]['timestamp'] - sends[0]['timestamp'] == 60
    assert path.read_bytes() == before


def test_ocr_mapping_uses_identity_and_does_not_merge_same_named_groups(ledger):
    path, insert = ledger
    insert('_chats_same_name_a', ['mapped'])
    insert('_chats_same_name_b', ['unmapped'])
    insert([dict(chatroom_id='group-1'), dict(chatroom_id='group-2')], ['ambiguous'])
    insert('group-2', ['raw'])
    assert [(s['source_chat_id'], s['text']) for s in
            load_sent_replies(path, {'_chats_same_name_a': 'group-1'})] == [
                ('group-1', 'mapped'), ('group-2', 'raw')]


@pytest.mark.parametrize('identity,replies', [(123, ['text']), ('group-1', [None]),
                                            ('group-1', [' ']), ('group-1', {}),
                                            ('group-1', ['partial', None])])
def test_malformed_send_evidence_does_not_discard_valid_rows(ledger, identity, replies, caplog):
    path, insert = ledger
    insert('group-1', ['before'])
    insert(identity, replies)
    insert('group-1', ['after'])
    assert [item['text'] for item in load_sent_replies(path)] == ['before', 'after']
    assert 'id=2' in caplog.text
    assert 'partial' not in caplog.text


def test_incremental_reads_capture_latest_send_finalization_once(ledger, history):
    path, insert = ledger
    history_path, observe = history
    message = dict(observed_at=1783310300.25, text='observed', sender='Alice', sender_type='other')
    observe('group-1', message)
    insert([message], [])
    reader = SentReplyLedger(path, history_database=history_path)
    assert reader.read() == []
    history_path.unlink()
    with sqlite3.connect(path) as connection:
        connection.execute('UPDATE tick_log SET replies_sent_json=?,send_success=0 WHERE id=1',
                           (json.dumps(['partial send']),))
    assert [item['text'] for item in reader.read()] == ['partial send']
    assert reader.read() == []
    insert('group-1', [])
    insert('group-2', ['new send'])
    assert [item['text'] for item in reader.read()] == ['new send']
    assert reader.cursor == 3
    assert reader.read() == []


def test_missing_database_is_not_created(tmp_path):
    path = tmp_path / 'missing.db'
    with pytest.raises(sqlite3.OperationalError):
        load_sent_replies(path)
    assert not path.exists()


@pytest.fixture
def history(tmp_path):
    path = tmp_path / 'history.db'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE chatrooms (id INTEGER PRIMARY KEY, chatroom_id TEXT)')
        connection.execute('CREATE TABLE messages (chatroom_id INTEGER, observed_at REAL, create_time REAL, '
                           'content TEXT, sender_display_name TEXT, is_self INTEGER)')

    def insert(identity, observation):
        with sqlite3.connect(path) as connection:
            room = connection.execute('INSERT INTO chatrooms (chatroom_id) VALUES (?)', (identity,)).lastrowid
            connection.execute('INSERT INTO messages VALUES (?,?,?,?,?,?)', (room, observation.get('observed_at'),
                observation.get('create_time'),
                observation['text'], observation['sender'], observation['sender_type'] == 'self'))
        return path

    return path, insert


@pytest.mark.parametrize('time_field', ['observed_at', 'create_time'])
def test_missing_identity_recovers_only_exact_observation_and_explicit_mapping(ledger, history, time_field):
    path, insert = ledger
    history_path, observe = history
    message = dict(text='same text', sender='Alice', sender_type='other', **{time_field: 1783310300.25})
    observe('_chats_same_name_a', message)
    insert([message], ['known bot'])
    before = path.read_bytes(), history_path.read_bytes()
    assert load_sent_replies(path, history_database=history_path) == []
    sends = load_sent_replies(path, {'_chats_same_name_a': 'group-1'}, history_database=history_path)
    assert sends == [dict(source_chat_id='group-1', timestamp=1783310400, text='known bot')]
    assert (path.read_bytes(), history_path.read_bytes()) == before


@pytest.mark.parametrize('change', [dict(observed_at=1783310300.26), dict(text='different'),
    dict(sender='Bob'), dict(sender_type='self'), dict(is_self=True), dict(observed_at=None)])
def test_incomplete_or_mismatched_observation_does_not_infer_identity(ledger, history, change):
    path, insert = ledger
    history_path, observe = history
    message = dict(observed_at=1783310300.25, text='same text', sender='Alice', sender_type='other')
    observe('group-1', message)
    insert([{**message, **change}], ['unknown source'])
    assert load_sent_replies(path, history_database=history_path) == []


@pytest.mark.parametrize('time_field', ['observed_at', 'create_time'])
@pytest.mark.parametrize('conflict', ['history', 'explicit', 'inputs'])
def test_conflicting_chat_sources_are_not_guessed(ledger, history, conflict, time_field):
    path, insert = ledger
    history_path, observe = history
    message = dict(text='same text', sender='Alice', sender_type='other', **{time_field: 1783310300.25})
    observe('group-1', message)
    inputs = [message]
    if conflict == 'history':
        observe('group-2', message)
    elif conflict == 'explicit':
        inputs = [{**message, 'chatroom_id': 'group-2'}]
    else:
        other = {**message, 'text': 'different'}
        observe('group-2', other)
        inputs.append(other)
    insert(inputs, ['ambiguous'])
    assert load_sent_replies(path, history_database=history_path) == []


def test_missing_history_database_is_not_created(ledger, tmp_path):
    history_path = tmp_path / 'absent-history.db'
    with pytest.raises(sqlite3.OperationalError):
        load_sent_replies(ledger[0], history_database=history_path)
    assert not history_path.exists()


@pytest.mark.parametrize('change', [dict(create_time=1783310301), dict(text='different'),
    dict(sender='Bob'), dict(sender_type='self'), dict(is_self=True), dict(create_time=None),
    dict(observed_at=1783310300), dict(observed_at=False)])
def test_legacy_identity_requires_exact_fields_and_time_basis(ledger, history, change):
    path, insert = ledger
    history_path, observe = history
    message = dict(create_time=1783310300, text='same text', sender='Alice', sender_type='other')
    observe('group-1', message)
    insert([{**message, **change}], ['unknown source'])
    assert load_sent_replies(path, history_database=history_path) == []


def test_recorded_time_does_not_replace_available_observation_time(ledger, history):
    path, insert = ledger
    history_path, observe = history
    message = dict(create_time=1783310300, observed_at=1783310300.25,
                   text='same text', sender='Alice', sender_type='other')
    observe('group-1', message)
    insert([{**message, 'observed_at': None}], ['unknown source'])
    assert load_sent_replies(path, history_database=history_path) == []
