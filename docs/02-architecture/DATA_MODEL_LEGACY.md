# 聊天记录持久化数据库设计

## 1. 背景与问题

当前 bot 的聊天记录通过 `GlobalStore` 以 **JSON 分片文件** 形式保存在 `data/chats/` 目录：

- 每个聊天一个 JSON 文件
- 索引在 `data/chats/index.json`
- 当前运行期数据保留在内存，bot 退出或异常时落盘

本次事故暴露出该方案的致命缺陷：

1. **数据丢失后不可恢复**：`data/` 目录整体被 `.gitignore`，JSON 文件被覆盖或损坏后无法回滚。
2. **同名群无法区分**：两个群都叫"示例交流群"，但 JSON 只按群名分片，没有 chatroom ID 字段，导致 wiki 生成时把两个群的成员/话题混在一起。
3. **没有增量备份**：每次覆盖都是破坏性操作，没有 point-in-time 备份。
4. **查询困难**：想按时间、sender、chatroom ID 查历史只能遍历 JSON，无法支撑 wiki 重建和事实核查。

## 2. 目标

- 用 **SQLite + SQLAlchemy 2.0** 把聊天记录持久化到本地数据库。
- 数据库文件纳入日常备份机制（与代码/配置分离但可恢复）。
- 聊天记录以 **chatroom_id / wxid** 为主键区分，不再只靠群名。
- 保留现有 `GlobalStore` 接口，业务代码零侵入或最小侵入。
- 支持从数据库重建 wiki、检索历史、审计事实。
- 支持从旧 JSON 分片迁移历史数据。

## 3. 非目标

- 不做分布式数据库、不做主从同步。
- 不替换向量索引（`vector_index_dense_messages.pkl` 继续由 `history_search` 使用）。
- 不改 WeChat 消息抓取逻辑，只改持久化层。

## 4. 数据模型

数据库：`data/chat_history.db`

### 4.1 `chats` 表

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | 自增 |
| chatroom_id | TEXT UNIQUE | 微信 chatroom_id（如 `20934380170@chatroom`）或私聊 wxid |
| chat_name | TEXT | 当前显示名，允许同名 |
| chat_type | TEXT | `group` / `single` |
| avatar_url | TEXT | 可选 |
| first_seen_at | REAL | 首次发现时间戳 |
| last_seen_at | REAL | 最近更新时间戳 |
| created_at | REAL | 记录创建时间 |

### 4.2 `messages` 表

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | 自增 |
| chat_id | INTEGER FK → chats.id | 聊天归属 |
| local_id | INTEGER | 消息在导出文件中的 localId（可选） |
| server_id | TEXT | 微信 server_msg_id（可选） |
| sender_wxid | TEXT | 发送者 wxid |
| sender_display_name | TEXT | 发送者显示名 |
| is_self | BOOLEAN | 是否示例用户甲自己发送 |
| content | TEXT | 消息文本 |
| message_type | TEXT | `text` / `image` / `emoji` / `system` / ... |
| image_description | TEXT | 图片 OCR/描述 |
| is_at_me | BOOLEAN | 是否 @ 我 |
| replied | BOOLEAN | bot 是否已回复 |
| reply_text | TEXT | bot 回复内容 |
| reply_time | REAL | 回复时间 |
| create_time | REAL | 消息原始时间戳 |
| raw_type | INTEGER | 微信原始类型码 |
| source_file | TEXT | 来源导出文件路径 |
| content_hash | TEXT | 内容哈希，用于去重 |
| created_at | REAL | 入库时间 |

索引：

- `idx_messages_chat_id_create_time`
- `idx_messages_sender_wxid`
- `idx_messages_content_hash`（去重）

### 4.3 `chat_members` 表（可选，用于记录群成员变更）

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | 自增 |
| chat_id | INTEGER FK | 群 ID |
| wxid | TEXT | 成员 wxid |
| display_name | TEXT | 群内显示名 |
| joined_at | REAL | 首次发现时间 |
| UNIQUE(chat_id, wxid) | | |

## 5. 架构设计

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ WeChat 消息抓取  │────▶│  GlobalStore     │────▶│ chat_history.db │
│ (OCR / WeFlow)  │     │  (内存 + JSON)    │     │  (SQLite)       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                           │
                               ▼                           ▼
                        data/chats/*.json           重建 wiki / 审计
```

- `GlobalStore` 保留现有接口作为内存缓冲。
- 新增 `ChatHistoryRepository`（`src/memory/chat_history_repo.py`），负责把 `GlobalStore` 中的消息同步到 SQLite。
- 每次 `GlobalStore.save()` 时，把新增/变更的消息批量写入 `messages` 表。
- 启动时从数据库加载历史到 `GlobalStore`，恢复内存状态。

## 6. 关键流程

### 6.1 消息写入

1. `tick` 捕获到新消息 → `GlobalStore.merge_tick()`。
2. `GlobalStore.save()` 被调用时：
   - 先按现有逻辑写 JSON 分片（保持向后兼容）。
   - 再调用 `ChatHistoryRepository.upsert_messages(chat_name, messages)`。
3. `upsert_messages` 根据 `chatroom_id` 找到 `chats.id`，批量 INSERT OR IGNORE（按 content_hash 去重）。

### 6.2 启动加载

1. `GlobalStore._load()` 优先加载 JSON 分片（保持现有行为）。
2. 若 JSON 为空但数据库有数据，可选回退从数据库加载（migration 模式）。

### 6.3 同名群处理

- `chats.chat_name` 不唯一，`chatroom_id` 唯一。
- `GlobalStore` 内部仍以 `chat_name` 为 key，但 `ChatMessage` 新增 `chatroom_id` 字段。
- wiki 生成时优先用 `chatroom_id` 过滤消息，避免同名群串味。

## 7. 迁移策略

1. **零停机迁移**：数据库表创建后，旧 JSON 继续可用。
2. **批量导入旧数据**：提供 `scripts/migrate_chats_to_db.py`，读取 `data/chats/` 和 `data/exports/` 下的 JSON，按 `chatroom_id` 导入数据库。
3. **回退**：若数据库异常，仍可使用 JSON 分片启动。

## 8. 备份与恢复

- 数据库文件 `data/chat_history.db` 本身是一个独立文件，可定时 `cp` 到 `backups/`。
- 每次 bot 启动或定时任务创建 `.db.bak.<timestamp>`。
- 保留最近 N 个备份，自动清理旧的。

## 9. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| SQLite 单文件过大 | 查询变慢 | 后续可按月分表或迁移到 PostgreSQL；初期 100 万条约 1-2GB 可接受 |
| 双写（JSON + DB）不一致 | 数据冗余/冲突 | 以 DB 为最终一致源，JSON 作为降级缓存 |
| 迁移过程覆盖现有数据 | 再次丢失 | 迁移前强制备份 `.db` 和 `data/chats/` |
| SQLAlchemy 版本兼容 | 运行报错 | 使用已验证的 2.0 写法，避免 1.x/2.0 混用 |

## 10. 实施步骤

1. 新增 `src/memory/chat_history_repo.py`：模型定义、upsert、查询接口。
2. 新增 `src/memory/chat_history_migrate.py`：从 JSON/导出文件迁移历史。
3. 修改 `src/session/global_store.py`：在 `save()` 中同步写入 DB；在 `_load()` 中支持 DB 回退。
4. 修改 `src/models/base.py` 的 `ChatMessage`：新增 `chatroom_id` 字段。
5. 新增 `scripts/migrate_chats_to_db.py` 命令行工具。
6. 更新 `requirements.txt` 加入 `sqlalchemy>=2.0.0`。
7. 写测试覆盖 upsert、去重、同名群过滤。

## 11. 验收标准

- [ ] `data/chat_history.db` 创建成功且包含 `chats` / `messages` 表。
- [ ] bot 运行后新消息写入数据库，重启可恢复。
- [ ] 两个同名群的聊天记录在数据库中按 `chatroom_id` 区分。
- [ ] 从数据库可重建指定 chatroom_id 的群聊 wiki。
- [ ] 旧 JSON 数据迁移后数据库可查询。
