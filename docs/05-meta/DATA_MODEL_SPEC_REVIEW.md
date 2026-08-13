## 审查结论

- **总体判断：有条件通过**
- **最大风险**：范围过大，12 张表一次性推进容易半途而废；应优先落地 Phase 1 的消息持久化，再逐步迁移身份/wiki。

## 问题清单

| 维度 | 问题 | 影响 | 建议 |
|---|---|---|---|
| 可行性 | 12 张表 + 3 个 Phase 范围偏大，与当前"先把聊天记录存下来避免再丢"的紧迫需求不完全匹配。 | 实施周期长，可能卡在 Phase 1 很久，用户看不到效果。 | 把 Phase 1 拆成独立的 MVP：只建 `chatrooms` / `messages` / `chat_members` 三张表，先让消息不再丢。 |
| 架构设计 | `GlobalStore` 双写 JSON + DB 后，没有说明启动时如何裁决。 | 可能重复导入或丢消息。 | 明确：启动优先读 DB；若 DB 为空则迁移 JSON；两者都不为空时以 DB 为准，JSON 降级为只读。 |
| 数据模型 | `facts.relation` + `facts.value` 是键值对，太松散，难以做结构化查询。 | 后续查"所有在上海的人"、"所有医生"会很麻烦。 | 增加 `fact_type` 枚举（`location` / `job` / `education` / `relationship` / `custom`），`value` 保留原始文本，同时加 `normalized_value` 用于搜索。 |
| 数据模型 | `overrides.content` 是 TEXT，没有 schema。 | 不同 override_type 的格式不统一，解析容易出错。 | 每个 override_type 定义明确 JSON schema，并在代码层用 Pydantic/SQLAlchemy 校验。 |
| 数据模型 | `wiki_pages.content_hash` 和 `wiki_revisions.content_hash` 算法未指定。 | 哈希不一致导致版本去重失败。 | 统一使用 `hashlib.sha256(content.encode('utf-8')).hexdigest()`。 |
| 安全与风险 | 提到 WAL 和备份，但没有说明多表写入的事务边界。 | chatroom + message + chat_member 同时插入时部分失败会留下脏数据。 | 所有 upsert 用 SQLAlchemy session 包裹，失败 rollback；关键操作（迁移、wiki 生成）支持幂等重试。 |
| 安全与风险 | 没有字段级加密或敏感信息处理。 | 聊天记录包含手机号、地址、身份证号等敏感信息，明文存储在 `.db` 文件中有泄露风险。 | 增加 `messages.is_sensitive` 标记，对识别出的敏感内容（身份证/银行卡/手机号）可配置加密或脱敏；并在 AGENTS.md 安全规则中呼应。 |
| 清晰完整 | 缺少从现有 `data/memory/overrides/*.json` 迁移到 `overrides` 表的具体方案。 | 现有人工纠正会丢失。 | 补充迁移脚本逻辑：读 JSON → 按 target_type/override_type 映射 → 写入 DB，冲突时以 DB 为准或人工确认。 |
| 清晰完整 | 没有说明 `chat_members.person_id` 未识别时如何回填。 | 大量历史消息导入后 person_id 为空，影响身份查询。 | 增加离线 identity resolution 任务：定期根据 wxid/别名/群内昵称匹配并回填 `person_id`。 |

## 必须补充的 3 件事

1. **MVP 边界收窄**：Phase 1 只实现 `chatrooms` / `messages` / `chat_members` 三张表 + 双写 + 备份，先解决丢数据问题，其他表后续迭代。
2. **明确权威数据源和启动顺序**：写操作双写 JSON+DB，读操作以 DB 为唯一权威源；启动时 JSON 仅作一次性迁移源。
3. **补充现有 overrides 迁移方案**：不能把现有人工纠正丢下，必须提供从 `aliases.json` / `corrections.json` / `facts.json` 到 `overrides` / `aliases` / `facts` 表的迁移脚本。

## 建议优化

- `facts` 表增加 `fact_type` 和 `normalized_value`，提升查询能力。
- 数据库文件路径改为 `data/db/chat_history.db`，与 `data/` 下其他数据库（`tick_log.db` / `cases.db`）统一目录结构。
- 增加 `chatrooms.wxid` 字段，兼容私聊场景（当前 `chatroom_id` 对私聊来说就是对方 wxid，但语义上分开更清晰）。
- 为 `messages` 增加 `seq` 字段记录消息在群内的绝对序号，方便后续和 WeFlow 精确对齐。
- 在 `wiki_revisions` 中增加 `prev_revision_id` 形成链表，便于追溯完整修改历史。
