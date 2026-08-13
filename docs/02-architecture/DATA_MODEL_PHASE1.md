# 聊天记录持久化 Phase 1 MVP 设计

## 1. 范围

Phase 1 已演进为 **DB-only 架构**：`GlobalStore` 不再使用 JSON 分片，SQLite 成为聊天记录的唯一权威源。解决**聊天记录丢失**、**同名群无法区分**和**多副本不一致**三个问题。只建 3 张表：

- `chatrooms`
- `messages`
- `chat_members`

别名、事实、wiki 版本控制等放到后续 Phase。

## 2. 数据模型

### 2.1 `chatrooms`

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | 自增 |
| chatroom_id | TEXT UNIQUE NOT NULL | 微信 chatroom_id 或私聊 wxid |
| display_name | TEXT | 当前显示名 |
| chat_type | TEXT | `group` / `single` |
| first_seen_at | REAL | 首次发现时间 |
| last_active_at | REAL | 最近活跃时间 |
| created_at | REAL | 记录创建时间 |
| updated_at | REAL | 记录更新时间 |

### 2.2 `messages`

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | 自增 |
| chatroom_id | INTEGER FK → chatrooms.id | 聊天 ID |
| local_id | INTEGER | 导出 localId |
| server_id | TEXT | 微信 server_msg_id |
| wxid | TEXT | 发送者 wxid |
| sender_display_name | TEXT | 发送者显示名 |
| is_self | BOOLEAN | 是否林岚发送 |
| content | TEXT | 文本内容 |
| message_type | TEXT | `text` / `image` / `emoji` / `system` / ... |
| image_description | TEXT | 图片描述 |
| is_at_me | BOOLEAN | 是否 @ 我 |
| is_revoked | BOOLEAN | 是否撤回 |
| replied | BOOLEAN | Bot 是否已回复 |
| reply_text | TEXT | Bot 回复内容 |
| reply_time | REAL | Bot 回复时间 |
| create_time | REAL | 消息原始时间戳 |
| raw_type | INTEGER | 微信原始类型码 |
| source_file | TEXT | 来源导出文件路径 |
| content_hash | TEXT | 内容哈希 |
| created_at | REAL | 入库时间 |

唯一约束：`UNIQUE(chatroom_id, wxid, create_time, content_hash)`

索引：
- `idx_messages_chatroom_create_time`
- `idx_messages_wxid`
- `idx_messages_content_hash`

### 2.3 `chat_members`

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | 自增 |
| chatroom_id | INTEGER FK → chatrooms.id | 群 ID |
| wxid | TEXT NOT NULL | 成员 wxid |
| group_nickname | TEXT | 群内昵称 |
| joined_at | REAL | 首次发现时间 |
| is_active | BOOLEAN DEFAULT 1 | 是否仍在群中 |
| created_at | REAL | 记录创建时间 |

唯一约束：`UNIQUE(chatroom_id, wxid)`

## 3. 架构

```
WeChat 抓取 / 导出文件
       │
       ▼
┌──────────────┐    upsert    ┌──────────────────┐
│ GlobalStore  │─────────────►│ ChatHistoryRepo  │
│   (内存)      │              │   (SQLite)       │
└──────────────┘              └──────────────────┘
       ▲                            │
       └──────── load ──────────────┘
```

- `GlobalStore` 继续以 `chat_name` 为内存 key，运行时逻辑不变。
- 每条 `ChatMessage` 新增 `chatroom_id` 字段。
- `GlobalStore.save()` 只把脏聊天同步到 DB，不再写 JSON 分片。
- **启动时只从 DB 加载**，DB 是唯一权威源。旧 `data/chats/` JSON 分片已归档到 `backups/chats_archive_YYYYMMDD/`。

## 4. 关键流程

### 4.1 消息写入

1. `tick` 产生新消息 → `GlobalStore.merge_tick()`
2. `GlobalStore.save()` 被调用
3. 调用 `ChatHistoryRepo.bulk_sync_chat(chatroom_id, display_name, chat_type, messages)`
4. Repository 内部 upsert chatroom → upsert messages → upsert chat_members

### 4.2 启动加载

1. `_load()` 直接调用 `_load_from_db()`
2. `list_chatrooms()` 遍历 DB 中所有 chatroom
3. 对同名群按规则选择单一 chatroom_id：优先合成 ID，其次非合成 ID 中消息最多者
4. 每个聊天加载最近 `max_messages` 条到内存

### 4.3 同名群处理

- DB 中以 `chatroom_id` 区分。
- `ChatMessage.chatroom_id` 从抓取/导出层带入。
- 迁移脚本读取导出文件 `session.wxid` 作为 `chatroom_id`。

## 5. 迁移与归档

### 5.1 初始迁移
`scripts/db/migrate_exports_to_db.py`

功能：
- 读取 `data/exports/main/` 和 `data/exports/b/`（WeChat 导出 JSON）
- 提取 `chatroom_id`、`messages`、`senders`
- 批量 upsert 到 DB

### 5.2 JSON 分片归档
2026-07-04 已将旧 `data/chats/`（GlobalStore 分片 JSON）整体归档到 `backups/chats_archive_YYYYMMDD/`，bot 不再读取或写入该目录。

安全：
- 迁移前自动备份 `data/db/chat_history.db`（若存在）
- 使用 `INSERT OR IGNORE` 避免重复
- 完成后打印统计：chatrooms / messages / chat_members / skipped duplicates

## 6. 备份策略

- SQLite 启用 WAL 模式。
- 数据库文件：`data/db/chat_history.db`
- 提供 `scripts/db/backup_chat_db.py`：
  - 默认备份到 `backups/chat_db/chat_history.db.bak.<timestamp>`
  - 支持 `--retention N` 保留最近 N 个
- bot 启动时不再自动备份，改为显式调用脚本。

## 7. 验收标准

- [x] `data/db/chat_history.db` 创建成功，包含 3 张表。
- [x] bot 运行时新消息只写入 DB，不再写 JSON 分片。
- [x] 两个同名群按 `chatroom_id` 在 DB 中分开存储。
- [x] 旧导出文件可迁移到 DB。
- [x] `data/chats/` JSON 分片已归档，bot 启动从 DB 加载。
- [x] DB 备份可恢复。
