# 微信 Bot 全系统数据模型设计

## 1. 设计目标

当前系统数据散落在多种格式和目录中，导致：

- 同名群无法区分，wiki 串味
- 聊天记录丢失后无法恢复
- 别名/事实/纠正信息重复存储且没有来源追溯
- wiki 生成依赖易丢失的内存/JSON 状态
- 没有统一的备份点和恢复流程

本设计目标：

1. 用 **SQLite 作为核心持久化层**，统一存储聊天记录、身份、群成员、事实、别名、wiki 元数据。
2. Markdown 文件保留为 **wiki 渲染层**，内容由 DB 生成并附加版本控制。
3. 所有实体以 **微信原生 ID（wxid / chatroom_id）** 为主键，不再依赖显示名。
4. 提供 **来源追溯**：每条事实、每个别名都能追溯到原始消息或人工覆盖。
5. 建立 **自动备份和 point-in-time 恢复** 机制。

## 2. 实体关系总览

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│   Person    │◄─────►│    Alias    │       │    Fact     │
│  (统一身份)  │       │   (别名表)   │       │   (事实表)   │
└──────┬──────┘       └─────────────┘       └─────────────┘
       │
       │ 1:N            ┌─────────────┐       ┌─────────────┐
       ├───────────────►│ ChatMember  │◄─────►│  Chatroom   │
       │                │  (群成员关系) │       │  (群/私聊)   │
       │                └─────────────┘       └──────┬──────┘
       │                                               │
       │                                               │ 1:N
       │                                        ┌──────▼──────┐
       │                                        │   Message   │
       │                                        │  (聊天记录)  │
       │                                        └──────┬──────┘
       │                                               │
       │                                               │ 1:N
       │                                        ┌──────▼──────┐
       │                                        │  BotReply   │
       │                                        │  (Bot 回复) │
       │                                        └─────────────┘
       │
       │ 1:1            ┌─────────────┐
       └───────────────►│ WikiPage    │
                        │  (wiki 页)  │
                        └─────────────┘
```

## 3. 核心实体

### 3.1 `chatrooms` —— 聊天会话

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | 自增 |
| chatroom_id | TEXT UNIQUE NOT NULL | 微信 chatroom_id 或私聊对方 wxid |
| display_name | TEXT NOT NULL | 当前显示名，允许同名 |
| chat_type | TEXT NOT NULL | `group` / `single` / `official` |
| avatar_url | TEXT | 头像 URL |
| is_pinned | BOOLEAN DEFAULT 0 | 是否置顶 |
| is_muted | BOOLEAN DEFAULT 0 | 是否免打扰 |
| first_seen_at | REAL | 首次发现时间戳 |
| last_active_at | REAL | 最近活跃时间戳 |
| meta_json | TEXT | 扩展字段（JSON） |
| created_at | REAL | 记录创建时间 |
| updated_at | REAL | 记录更新时间 |

**设计要点**：
- `chatroom_id` 是全局唯一主键，解决同名群问题。
- 若微信改名，`display_name` 更新，但 `chatroom_id` 不变。

### 3.2 `persons` —— 统一身份

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | 自增 |
| canonical_name | TEXT NOT NULL | 系统认定的主名 |
| primary_wxid | TEXT UNIQUE | 主要 wxid |
| notes | TEXT | 人工备注 |
| meta_json | TEXT | 扩展字段 |
| created_at | REAL | 记录创建时间 |
| updated_at | REAL | 记录更新时间 |

**设计要点**：
- 一个人可能有多个 wxid（换号、小号），通过 `person_wxids` 关联表处理。
- `canonical_name` 由用户指定或算法推选，稳定不变。

### 3.3 `person_wxids` —— 人的微信 ID 关联

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | 自增 |
| person_id | INTEGER FK → persons.id | 统一身份 ID |
| wxid | TEXT UNIQUE NOT NULL | 微信 ID |
| display_name | TEXT | 该 wxid 的显示名 |
| first_seen_at | REAL | 首次发现时间 |
| source_chatroom_id | TEXT | 在哪个聊天中发现 |

### 3.4 `chat_members` —— 群成员关系

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | 自增 |
| chatroom_id | INTEGER FK → chatrooms.id | 群 ID |
| person_id | INTEGER FK → persons.id | 统一身份 ID（nullable，未识别时为空） |
| wxid | TEXT NOT NULL | 成员 wxid |
| group_nickname | TEXT | 群内昵称 |
| joined_at | REAL | 首次在群内发现时间 |
| left_at | REAL | 退群时间（nullable） |
| is_active | BOOLEAN DEFAULT 1 | 是否仍在群中 |
| UNIQUE(chatroom_id, wxid) | | |

**设计要点**：
- 同名群的区别就靠 `chat_members` 组成（一个有小夏，一个没有）。
- 退群不删记录，标记 `is_active=0`。

### 3.5 `messages` —— 聊天记录

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | 自增 |
| chatroom_id | INTEGER FK → chatrooms.id | 聊天 ID |
| local_id | INTEGER | 导出文件中的 localId |
| server_id | TEXT | 微信 server_msg_id |
| wxid | TEXT NOT NULL | 发送者 wxid |
| person_id | INTEGER FK → persons.id | 发送者统一身份（nullable） |
| is_self | BOOLEAN DEFAULT 0 | 是否林岚发送 |
| content | TEXT | 文本内容 |
| message_type | TEXT NOT NULL | `text` / `image` / `emoji` / `voice` / `video` / `location` / `system` / `revoke` |
| image_description | TEXT | 图片 OCR/描述 |
| is_at_me | BOOLEAN DEFAULT 0 | 是否 @ 我 |
| is_revoked | BOOLEAN DEFAULT 0 | 是否已撤回 |
| replied | BOOLEAN DEFAULT 0 | Bot 是否已回复 |
| reply_text | TEXT | Bot 回复内容 |
| reply_time | REAL | Bot 回复时间 |
| create_time | REAL NOT NULL | 消息原始时间戳 |
| raw_type | INTEGER | 微信原始类型码 |
| source_file | TEXT | 来源导出文件路径 |
| content_hash | TEXT NOT NULL | 内容哈希 |
| vector_indexed | BOOLEAN DEFAULT 0 | 是否已加入向量索引 |
| meta_json | TEXT | 扩展字段 |
| created_at | REAL | 入库时间 |

**唯一约束**：`UNIQUE(chatroom_id, wxid, create_time, content_hash)`

**索引**：
- `idx_messages_chatroom_create_time`
- `idx_messages_wxid`
- `idx_messages_person_id`
- `idx_messages_content_hash`
- `idx_messages_vector_indexed`

### 3.6 `bot_replies` —— Bot 回复记录

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | 自增 |
| message_id | INTEGER FK → messages.id | 回复的消息 |
| reply_text | TEXT | 回复内容 |
| llm_model | TEXT | 使用的模型 |
| tools_used | TEXT | JSON 数组，调用了哪些工具 |
| latency_ms | INTEGER | 响应耗时 |
| reply_time | REAL | 回复时间 |

### 3.7 `aliases` —— 别名表

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | 自增 |
| person_id | INTEGER FK → persons.id | 所属人 |
| alias | TEXT NOT NULL | 别名 |
| source_type | TEXT NOT NULL | `message` / `override` / `manual` / `inferred` |
| source_message_id | INTEGER FK → messages.id | 来源消息（nullable） |
| source_override_id | INTEGER | 来源覆盖记录（nullable） |
| is_valid | BOOLEAN DEFAULT 1 | 是否有效（被覆盖否定时置 0） |
| created_at | REAL | 创建时间 |

**唯一约束**：`UNIQUE(person_id, alias)`

### 3.8 `facts` —— 事实表

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | 自增 |
| person_id | INTEGER FK → persons.id | 所属人 |
| chatroom_id | INTEGER FK → chatrooms.id | 所属群（nullable） |
| relation | TEXT NOT NULL | 关系/属性名，如 `职业`、`城市`、`配偶` |
| value | TEXT NOT NULL | 关系值 |
| confidence | REAL DEFAULT 1.0 | 置信度 0-1 |
| source_type | TEXT NOT NULL | `message` / `override` / `manual` / `inferred` |
| source_message_id | INTEGER FK → messages.id | 来源消息 |
| source_override_id | INTEGER | 来源覆盖记录 |
| is_valid | BOOLEAN DEFAULT 1 | 是否有效 |
| created_at | REAL | 创建时间 |
| updated_at | REAL | 更新时间 |

### 3.9 `overrides` —— 人工覆盖/纠正

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | 自增 |
| target_type | TEXT NOT NULL | `person` / `chatroom` / `fact` / `alias` |
| target_id | INTEGER | 目标 ID |
| override_type | TEXT NOT NULL | `alias_correction` / `fact_lock` / `name_preference` / `group_rule` |
| content | TEXT NOT NULL | 覆盖内容（如 JSON 或纯文本） |
| reason | TEXT | 人工填写原因 |
| created_at | REAL | 创建时间 |

**设计要点**：
- 替代现有的 `data/memory/overrides/{aliases,corrections,facts}.json`。
- 保留 JSON 导出能力，但权威数据在 DB。

### 3.10 `wiki_pages` —— wiki 页面元数据

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | 自增 |
| page_type | TEXT NOT NULL | `user` / `group` / `topic` |
| target_id | INTEGER | 关联的 person_id 或 chatroom_id（nullable） |
| name | TEXT NOT NULL | wiki 文件名（不含扩展名） |
| file_path | TEXT NOT NULL | Markdown 文件路径 |
| content_hash | TEXT | 内容哈希 |
| generated_at | REAL | 生成时间 |
| generated_by | TEXT | 生成器版本/模型 |
| is_auto_generated | BOOLEAN DEFAULT 1 | 是否自动生成 |
| meta_json | TEXT | 扩展字段 |

### 3.11 `wiki_revisions` —— wiki 版本

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | 自增 |
| wiki_page_id | INTEGER FK → wiki_pages.id | wiki 页 ID |
| content | TEXT | 完整 Markdown 内容 |
| content_hash | TEXT | 哈希 |
| generated_at | REAL | 生成时间 |
| generated_by | TEXT | 生成器 |
| note | TEXT | 备注（如 "manual_backup" / "auto_generated"） |

**设计要点**：
- 每次生成 wiki 前自动保存旧版本到 `wiki_revisions`。
- 解决本次 wiki 被覆盖无法恢复的问题。

### 3.12 `media_files` —— 媒体文件索引

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | 自增 |
| message_id | INTEGER FK → messages.id | 关联消息（nullable） |
| file_type | TEXT | `screenshot` / `image` / `voice` / `video` |
| file_path | TEXT NOT NULL | 文件路径 |
| file_hash | TEXT | 文件哈希 |
| created_at | REAL | 创建时间 |

## 4. 数据流

### 4.1 消息入库

```
WeChat 抓取 / 导出文件
       │
       ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ ChatImporter │───▶│  Chatroom    │───▶│   Message    │
│ (解析导出)    │    │  (get or     │    │  (upsert)    │
└──────────────┘    │   create)    │    └──────────────┘
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ ChatMember   │
                    │ (get or      │
                    │  create)     │
                    └──────────────┘
```

### 4.2 身份识别

```
wxid / 别名 / 群内昵称
       │
       ▼
┌──────────────┐    未匹配 ──▶ 新建 Person
│ Identity     │
│ Resolver     │    匹配 ────▶ 关联到已有 Person
└──────────────┘
```

### 4.3 wiki 生成

```
Chatroom + Messages + Persons + Aliases + Facts + Overrides
                              │
                              ▼
                    ┌──────────────────┐
                    │ WikiGenerator    │
                    │ (LLM / 规则)      │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ 保存旧版本到      │
                    │ wiki_revisions   │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ 写新 wiki.md     │
                    └──────────────────┘
```

## 5. 存储分层

| 层级 | 存储 | 内容 | 备份策略 |
|---|---|---|---|
| L1 原始消息 | `data/chat_history.db` | chatrooms / messages / bot_replies | 每次启动 + 定时 `cp` 到 `backups/` |
| L2 身份与事实 | `data/chat_history.db` | persons / aliases / facts / overrides | 同上 |
| L3 wiki | `data/memory/wiki/**/*.md` | 渲染后的人类可读 wiki | 每次生成前保存到 `wiki_revisions` |
| L4 向量索引 | `data/memory/cache/vector_index_dense_messages.pkl` | 语义检索索引 | 从 L1 可重建 |
| L5 运行时状态 | `data/tick_log.db` / `data/cases.db` | bot 运行日志 | 现有策略 |

## 6. 关键规则

### 6.1 同名群区分

- 所有查询 wiki/历史时以 `chatroom_id` 为准。
- UI/命令中若用户只提供群名，则列出候选群及其关键成员（如"有小夏的示例交流群"），让用户确认。

### 6.2 别名/事实来源

- 所有别名和事实必须有 `source_type` 和来源 ID。
- `override` 类型优先级最高，可否定 `message` 来源的别名/事实。

### 6.3 wiki 生成安全

- 生成前必须：
  1. 备份旧版本到 `wiki_revisions`。
  2. 确认 `chatroom_id` 正确。
  3. 确认数据源时间范围。
- 生成后必须记录 `generated_by` 和 `generated_at`。

### 6.4 备份与恢复

- 数据库启用 WAL 模式。
- 每次 bot 启动时创建 `data/chat_history.db.bak.<timestamp>`，保留最近 7 个。
- 任何迁移/初始化脚本执行前强制创建备份。
- 提供 `scripts/restore_chat_db.py` 从 `.bak` 恢复。

## 7. 实施路线图

### Phase 1：数据库基建（MVP）

1. 创建 `src/db/` 包：SQLAlchemy 模型、连接池、迁移脚本。
2. 实现 `ChatHistoryRepository`：chatroom / message / chat_member 的 upsert/query。
3. 修改 `GlobalStore.save()` 双写 JSON + DB。
4. 新增 `scripts/db/migrate_exports_to_db.py`：批量导入 `data/exports/` 和 `data/chats/`。

### Phase 2：身份与 wiki

1. 实现 `IdentityResolver`：从 wxid/别名/昵称解析 `person_id`。
2. 把 `aliases` / `facts` / `overrides` 迁移到 DB。
3. 修改 wiki 生成器：从 DB 读取，按 `chatroom_id` 过滤，生成前保存版本。

### Phase 3：工具与监控

1. 新增 CLI：`backup_chat_db.py`、`restore_chat_db.py`、`audit_chat_db.py`。
2. admin 页面展示 chatroom 列表、消息数、最后活跃时间。
3. 回归测试：同名群不串、wiki 可回滚、DB 备份可恢复。

## 8. 验收标准

- [ ] `data/chat_history.db` 创建并包含全部 12 张表。
- [ ] 新消息同时写入 JSON 分片和 DB，DB 不丢数据。
- [ ] 两个同名群在 DB 中按 `chatroom_id` 区分，wiki 生成可分别指定。
- [ ] 别名/事实/覆盖全部可追溯来源。
- [ ] 每次 wiki 生成前自动备份旧版本。
- [ ] 数据库备份可一键恢复。
- [ ] 旧 `data/exports/` 数据可完整导入 DB。
