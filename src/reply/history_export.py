"""Read explicitly identified RPA chats as DH unified history; never infer identity by name."""

import json
import math
from pathlib import Path
import sqlite3


def export_history(database, source_chat_ids, output, *, self_account):
    if not isinstance(self_account, str) or not self_account.strip() or ':' in self_account:
        raise ValueError("需要明确的本人账号")
    if (not isinstance(source_chat_ids, dict) or not source_chat_ids
            or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v
                   for k, v in source_chat_ids.items())
            or len(set(source_chat_ids.values())) != len(source_chat_ids)):
        raise ValueError("需要一对一的明确聊天 ID 映射")
    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    messages = []
    unknown_time_messages = omitted_before_reliable_tail = 0
    try:
        connection.execute("BEGIN")
        for alias, source in source_chat_ids.items():
            chats = connection.execute("SELECT id, display_name, chat_type FROM chatrooms "
                                       "WHERE chatroom_id=?", (alias,)).fetchall()
            if len(chats) != 1 or chats[0]['chat_type'] not in ('group', 'private'):
                raise ValueError("聊天 ID 缺失、不唯一或类型不明确")
            chat = chats[0]
            rows = connection.execute("SELECT id, wxid, sender_display_name, is_self, content, "
                "message_type, is_revoked, create_time, observed_at FROM messages "
                "WHERE chatroom_id=? ORDER BY id", (chat['id'],))
            chat_messages = []
            for row in rows:
                timestamp = row['observed_at'] if row['observed_at'] is not None else row['create_time']
                if row['is_self'] not in (0, 1):
                    raise ValueError("历史消息缺少可靠的本人标记")
                if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp <= 0:
                    unknown_time_messages += 1
                    omitted_before_reliable_tail += len(chat_messages) + 1
                    chat_messages.clear()
                    continue
                kind = 'system' if row['is_revoked'] else row['message_type']
                text = row['content'] or ''
                eligible = kind == 'text' and bool(text.strip())
                if not eligible:
                    text = f'[{kind}：内容未验证]'
                sender = row['sender_display_name'] or row['wxid'] or ('本人' if row['is_self'] else '')
                if not sender:
                    raise ValueError("历史消息缺少发言人")
                chat_messages.append(dict(chat_id=f"{chat['chat_type']}:{self_account}:{source}",
                    source_chat_id=source, chat_type=chat['chat_type'], chat_name=chat['display_name'],
                    sender=sender, is_self=bool(row['is_self']), timestamp=timestamp, text=text,
                    time_basis='first_observed' if row['observed_at'] is not None else 'recorded',
                    event=dict(schema=1, id=f"rpa:{row['id']}", kind=kind,
                        source_type_name=row['message_type'], sender_id=row['wxid'] or '',
                        reply_eligible=eligible, context_boundary=kind not in
                        ('text', 'image', 'mixed', 'sticker', 'link_card', 'video'))))
            if not chat_messages:
                raise ValueError("聊天没有时间完整的连续记录可供导出")
            messages.extend(sorted(chat_messages, key=lambda message: message['timestamp']))
    finally:
        connection.close()
    with Path(output).open('x', encoding='utf-8') as stream:
        for message in messages:
            stream.write(json.dumps(message, ensure_ascii=False, allow_nan=False) + '\n')
    return dict(messages=len(messages), self_messages=sum(m['is_self'] for m in messages),
                source_chat_ids=list(source_chat_ids.values()), unknown_time_messages=unknown_time_messages,
                omitted_before_reliable_tail=omitted_before_reliable_tail)
