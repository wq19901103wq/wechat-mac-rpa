import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from src.reply.history_export import export_history


def test_cli_exports_history_and_sends_using_exact_observation(tmp_path):
    database, ledger = tmp_path / 'history.db', tmp_path / 'sends.db'
    output, sent_output = tmp_path / 'history.jsonl', tmp_path / 'sends.json'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE chatrooms (id,chatroom_id,display_name,chat_type)')
        db.execute("INSERT INTO chatrooms VALUES (1,'alias','room','group')")
        db.execute('CREATE TABLE messages (id,chatroom_id,wxid,sender_display_name,is_self,content,'
                   'message_type,is_revoked,create_time,observed_at)')
        db.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?)',
                   (1, 1, 'peer', 'friend', 0, 'question', 'text', 0, 1783310300, 1783310300.25))
    with sqlite3.connect(ledger) as db:
        db.execute('CREATE TABLE tick_log (id,created_at,session_input_messages_json,'
                   'replies_sent_json,send_success)')
        inputs = [dict(observed_at=1783310300.25, text='question', sender='friend', sender_type='other')]
        db.execute('INSERT INTO tick_log VALUES (?,?,?,?,?)',
                   (1, '2026-07-06 12:00:00', json.dumps(inputs), json.dumps(['sent reply']), 1))
    before = database.read_bytes(), ledger.read_bytes()
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, str(root / 'scripts/export_dh_history.py'),
        '--database', str(database), '--source-chat-ids', '{"alias":"source-room"}',
        '--self-account', 'owner', '--output', str(output),
        '--sent-replies-db', str(ledger), '--sent-output', str(sent_output)],
        check=True, text=True, capture_output=True)
    assert json.loads(result.stdout)['sent_replies'] == 1
    assert json.loads(output.read_text())['source_chat_id'] == 'source-room'
    assert json.loads(sent_output.read_text()) == [dict(source_chat_id='source-room',
        timestamp=1783310400, text='sent reply')]
    assert (database.read_bytes(), ledger.read_bytes()) == before


def test_exact_chat_export_preserves_context_and_ignores_generated_columns(tmp_path):
    database, output = tmp_path / 'history.db', tmp_path / 'history.jsonl'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE chatrooms (id,chatroom_id,display_name,chat_type)')
        db.executemany('INSERT INTO chatrooms VALUES (?,?,?,?)',
                       [(1,'alias-a','same name','group'),(2,'alias-b','same name','group')])
        db.execute('CREATE TABLE messages (id,chatroom_id,wxid,sender_display_name,is_self,content,'
                   'message_type,is_revoked,create_time,observed_at,reply_text)')
        db.executemany('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?)', [
            (1,1,'other','friend',0,'question','text',0,100,None,'generated but not sent'),
            (2,1,'me','self',1,'manual reply','text',0,101,None,None),
            (3,1,'other','friend',0,'unreliable OCR','image',0,102,105,None),
            (4,2,'me','self',1,'different group','text',0,100,None,None)])
    assert export_history(database, {'alias-a':'exact-a'}, output, self_account='owner')['messages'] == 3
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert {r['source_chat_id'] for r in rows} == {'exact-a'}
    assert {r['chat_id'] for r in rows} == {'group:owner:exact-a'}
    assert rows[1]['text'] == 'manual reply' and rows[1]['is_self'] is True
    assert rows[2]['timestamp'] == 105 and rows[2]['time_basis'] == 'first_observed'
    assert rows[2]['event']['reply_eligible'] is False
    assert 'OCR' not in output.read_text() and 'generated' not in output.read_text()
    with pytest.raises(FileExistsError):
        export_history(database, {'alias-a':'exact-a'}, output, self_account='owner')
    with pytest.raises(ValueError, match='一对一'):
        export_history(database, {'alias-a':'same','alias-b':'same'}, tmp_path/'bad', self_account='owner')
    with pytest.raises(ValueError, match='缺失'):
        export_history(database, {'same name':'exact-a'}, tmp_path/'bad', self_account='owner')
    for account in ('', ' ', 'wrong:owner', None):
        with pytest.raises(ValueError, match='本人账号'):
            export_history(database, {'alias-a':'exact-a'}, tmp_path/'bad', self_account=account)


def test_unknown_time_keeps_only_the_complete_tail_without_joining_across_gaps(tmp_path):
    database, output = tmp_path / 'history.db', tmp_path / 'history.jsonl'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE chatrooms (id,chatroom_id,display_name,chat_type)')
        db.execute("INSERT INTO chatrooms VALUES (1,'alias','group','group')")
        db.execute('CREATE TABLE messages (id,chatroom_id,wxid,sender_display_name,is_self,content,'
                   'message_type,is_revoked,create_time,observed_at)')
        db.executemany('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?)', [
            (1,1,'other','friend',0,'must not join later reply','text',0,100,None),
            (2,1,'other','friend',0,'unknown time','text',0,0,None),
            (3,1,'me','self',1,'must not become a reply across gap','text',0,101,None),
            (4,1,'other','friend',0,'another unknown time','text',0,0,None),
            (5,1,'other','friend',0,'reliable question','text',0,102,None),
            (6,1,'me','self',1,'reliable reply','text',0,103,None)])
    result = export_history(database, {'alias':'exact'}, output, self_account='owner')
    assert result['unknown_time_messages'] == 2 and result['omitted_before_reliable_tail'] == 4
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row['text'] for row in rows] == ['reliable question', 'reliable reply']
    assert [row['timestamp'] for row in rows] == [102, 103]
    with sqlite3.connect(database) as db:
        db.execute("UPDATE messages SET create_time=0 WHERE id=6")
    with pytest.raises(ValueError, match='没有时间完整'):
        export_history(database, {'alias':'exact'}, tmp_path / 'empty.jsonl', self_account='owner')
    assert not (tmp_path / 'empty.jsonl').exists()
