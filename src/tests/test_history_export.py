import json
import sqlite3

import pytest

from src.reply.history_export import export_history


def test_export_cli_uses_history_for_sent_reply_identity(tmp_path, monkeypatch, capsys):
    from scripts import export_dh_history
    database, sent_database = tmp_path / 'history.db', tmp_path / 'sent.db'
    evidence = [{'source_chat_id': 'exact', 'timestamp': 100, 'text': 'sent'}]
    received = []
    monkeypatch.setattr(export_dh_history, 'load_sent_replies',
        lambda db, mapping, **options: received.append((db, mapping, options)) or evidence)
    monkeypatch.setattr(export_dh_history, 'export_history', lambda *args, **kwargs: {'messages': 1})
    monkeypatch.setattr('sys.argv', ['export_dh_history', '--database', str(database),
        '--source-chat-ids', '{"alias":"exact"}', '--self-account', 'owner',
        '--output', str(tmp_path / 'history.jsonl'), '--sent-replies-db', str(sent_database),
        '--sent-output', str(tmp_path / 'sent.json')])
    export_dh_history.main()
    assert received == [(sent_database, {'alias': 'exact'}, {'history_database': database})]
    assert json.loads((tmp_path / 'sent.json').read_text()) == evidence
    assert json.loads(capsys.readouterr().out)['sent_replies'] == 1


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
