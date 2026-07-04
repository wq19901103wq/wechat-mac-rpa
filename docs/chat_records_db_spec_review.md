## 审查结论

- **总体判断：有条件通过**
- **最大风险**：双写 JSON+DB 的同步策略和唯一键设计不够具体，可能导致同名群去重错误或迁移时再次覆盖数据。

## 问题清单

| 维度 | 问题 | 影响 | 建议 |
|---|---|---|---|
| 架构设计 | 双写策略语焉不详。spec 说"以 DB 为最终一致源，JSON 作为降级缓存"，但没有说启动时以谁为准、冲突时如何裁决。 | 启动时 JSON 与 DB 内容可能不一致，bot 行为不可预测。 | 明确启动顺序：优先加载 DB；若 JSON 存在且 DB 为空，则迁移 JSON 到 DB；两者都不为空时以 DB 为准，JSON 仅作只读降级。 |
| 数据模型 | `messages` 表去重只依赖 `content_hash`，未说明哈希算法，也未说明同一人在短时间内发重复文本如何处理。 | 可能误判不同消息为重复，或漏掉真正的重复消息。 | 采用复合唯一键：`(chat_id, sender_wxid, create_time, content_hash)`，并说明 content_hash 用 xxhash/sha256。对微信 server_id/local_id 优先使用。 |
| 数据模型 | 缺少"消息编辑/撤回"字段和设计。 | 无法处理微信撤回消息或编辑消息，数据库会保留过期内容。 | 增加 `is_revoked` 和 `updated_at` 字段；同步逻辑中检测撤回系统消息并标记。 |
| 安全与风险 | 迁移脚本风险缓解只有一句话"迁移前强制备份"，没有具体步骤。 | 再次触发类似 wiki 覆盖事故。 | 迁移脚本必须：先创建 `data/chat_history.db.bak.<timestamp>` 和 `data/chats.bak.<timestamp>`；所有 INSERT 使用 `INSERT OR IGNORE`；迁移完成后只读校验条数。 |
| 可行性 | SQLAlchemy 2.0 已在环境，但 `requirements.txt` 没列。 | 新环境安装会缺依赖。 | 在 `requirements.txt` 加入 `sqlalchemy>=2.0.0,<3.0.0`。 |
| 可行性 | 未说明 SQLite 是否启用 WAL 模式。 | 高并发 tick 写入时可能锁库或损坏。 | 明确启用 WAL：`PRAGMA journal_mode=WAL;`，并说明检查点策略。 |
| 清晰完整 | spec 没解决"2026 年聊天记录已经丢失"这个当下问题。 | 用户以为做了数据库就能恢复 wiki，但实际上丢失的数据回不来。 | 在"背景"或"非目标"中明确：本方案防止未来丢失，不能恢复已丢失的历史；若需要恢复 wiki，必须重新导出/抓取目标群聊天记录。 |
| 清晰完整 | `chats.chatroom_id` 与 `GlobalStore` 以 `chat_name` 为 key 的映射关系没有说明清楚。 | 两个同名群在内存中仍可能冲突。 | 在 GlobalStore 内部把 key 改为 `chatroom_id`，`chat_name` 仅作显示；或者维护 `chat_name -> chatroom_id` 的二级索引。 |

## 必须补充的 3 件事

1. **确定启动/同步的权威数据源**：启动时先读 DB，JSON 仅作为历史迁移源和只读降级；写消息时先写内存，再双写 JSON 和 DB，失败时回滚并报警。
2. **明确消息去重键和撤回处理**：用 `(chat_id, sender_wxid, create_time, content_hash)` 复合键，并增加 `is_revoked` 字段处理撤回。
3. **迁移脚本必须带强制备份和只读校验**：任何迁移/初始化操作前自动创建 `.bak` 快照，迁移完成后打印"导入消息数 / 唯一 chat 数 / 重复跳过数"。

## 建议优化

- 把 `chat_members` 表从"可选"改为第一批必须实现，因为区分同名群的核心就是成员组成（有西西 vs 没西西）。
- 增加数据库备份 CLI：`python scripts/backup_chat_db.py --retention 7`。
- 在 `messages` 表增加 `vector_indexed` 布尔字段，方便后续把新消息同步进向量索引时去重。
- 测试用例里专门加一个"两个同名群数据不串"的回归测试。
