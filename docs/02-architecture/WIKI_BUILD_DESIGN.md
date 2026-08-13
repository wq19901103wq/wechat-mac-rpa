# Wiki 批量生成设计文档

> 本文档描述从 `data/chats/*.json` 批量生成用户/群聊/话题 wiki 的完整架构与策略。
> 最后更新：2026-05-10

---

## 1. 整体架构

```
data/chats/*.json  (wechat 导出数据)
        │
        ▼
┌─────────────────────────────┐
│  scripts/bulk_import_from_chats.py  │
│  (批量导入脚本)               │
└─────────────────────────────┘
        │
        ├──► 数据清洗 & 分类 ──► 私聊 / 群聊 / 系统消息过滤
        │
        ├──► 全局用户索引 ──► wxid → 跨聊天聚合
        │
        ├──► 别名解析 ──► aliases.json 映射
        │
        ├──► 上下文构建 ──► 私聊完整对话 / 群聊分层上下文
        │
        ▼
┌─────────────────────────────┐
│  src/memory/engine.py │
│  (MemoryEngine LLM Wiki 引擎)│
└─────────────────────────────┘
        │
        ├──► Prompt 组装 (规则 + current_wiki + conversation)
        │
        ├──► LLM 调用 (deepseek-v4-flash, 1M 上下文)
        │
        ├──► 解析返回的 wiki markdown
        │
        ├──► 别名提取 & 持久化
        │
        ▼
data/memory/wiki/
├── users/      用户 wiki (*.md)
├── groups/     群聊 wiki (*.md)
├── topics/     话题 wiki (*.md)
├── prompts/    每次生成使用的 prompt (*.md)
└── alias_suggestions/  别名建议 (*.json)
```

---

## 2. 数据清洗 & 聊天分类

### 2.1 输入格式
`data/chats/*.json` 由 weflow 导出，结构：
```json
{
  "chat_name": "某群显示名",
  "messages": [
    {
      "sender": "微信昵称",
      "sender_wxid": "wxid_xxx",
      "sender_type": "self" | "other",
      "text": "消息文本",
      "create_time": 1704067200,
      "account": "账号标识"
    }
  ]
}
```

### 2.2 系统消息过滤 (`is_system_message`)
排除以下特征的消息：
- 文本为空或仅空白
- 包含系统关键词（"邀请你加入群聊"、"拍了拍"、"撤回了一条消息"等）
- 微信内置功能消息（语音通话、视频号、位置等）

### 2.3 聊天分类 (`classify_chat`)
| 信号 | 权重 | 说明 |
|------|------|------|
| stem 含 `@chatroom` | 强 | 直接判定为群聊 |
| 消息中 sender 种类 > 2 | 强 | 多人参与 = 群聊 |
| 消息量 > 10 且名称含"群"/"群聊" | 中 | 辅助判断 |
| 其他 | 私聊 | 默认 |

历史问题：早期版本 `"" in text` 恒为 True，导致所有消息被误判为系统消息，已修复为 `if not text`。

---

## 3. 全局用户索引 (`build_wxid_index`)

### 3.1 核心逻辑
遍历所有聊天的所有消息，按 `sender_wxid` 聚合：

```python
{wxid: {
    "main_name": "消息量最多的昵称",
    "all_names": {"昵称A": 出现次数, "昵称B": 次数},
    "total_msgs": 总消息数,
    "chat_count": 涉及聊天数,
    "chats": {stem: [该用户在此聊天的消息列表]}
}}
```

### 3.2 过滤规则
- 排除 `sender_type == "self"`（bot 自己不建索引）
- 排除 `sender` 为空/"对方"/"[未知]"
- 排除系统消息
- **排除 `wxid.endswith("@chatroom")`**（群聊系统消息以群名作为 sender）

### 3.3 主名稳定性 (`resolve_main_name`)
三层优先级保证 wiki 文件名不漂移：
1. `aliases.json` 中已记录的主名 → 直接复用
2. `data/memory/wiki/users/*.md` 中已有文件匹配该用户的某个昵称 → 复用已有文件名
3. 消息量最多的昵称 → 首次生成时使用

### 3.4 冲突处理（同一主名多个 wxid）
如果两个不同 wxid 解析为同一主名，第二个起自动改名：
- `示例用户甲.md`（第一个 wxid）
- `示例用户甲_2.md`（第二个 wxid）
- `示例用户甲_3.md`（第三个 wxid）

原理：不同 wxid 极大概率是不同的人（微信/企业微信/不同账号），宁可拆分也不错误合并。

---

## 4. 上下文构建策略

这是最关键的部分，直接影响 wiki 质量。

### 4.1 私聊上下文
**策略：完整对话上下文**

私聊只有两个人，消息量可控，完整上下文对用户性格、互动模式、关系深度的理解不可替代。

实现：
```python
full_msgs = all_chats[stem].get("messages", [])
chat_msgs = [m for m in full_msgs if not is_system_message(m)]
# 包含 self（bot 自己）和 other（对方）的完整对话
```

在 conversation 中：
- `sender_type == "self"` → 显示为 "我"
- `sender_type == "other"` → 显示为对方的微信昵称

### 4.2 群聊上下文
**策略：分层上下文（精准捕获相关片段）**

群聊消息量巨大（单群可达 10 万条），不能全传。采用三层叠加：

#### 目标消息定义
1. **用户自己发的消息** (`sender_wxid == 目标 wxid`)
2. **提到该用户的消息** (`text` 中包含该用户的任一昵称/别名)

#### 上下文扩展
对每条目标消息，收集：
- **前后 k=10 条消息**（理解对话流）
- **时间 ±5 分钟内的所有消息**（捕获同一话题的连续讨论，可能超过 10 条）

#### 合并与去重
```python
context_indices = 前后k条索引 ∪ 时间窗口索引
chat_msgs = [full_msgs[i] for i in sorted(context_indices)]
# 去重 + 排除系统消息 + 按 max_msgs_per_chat 截断
```

### 4.3 Token 估算与动态截断

不再按固定条数（如 2 万条）截断，而是**按估算 token 数动态填充到接近 1M 上下文上限**：

```python
def estimate_tokens(text: str) -> int:
    cn = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other = len(text) - cn
    return int(cn * 1.5 + other * 0.5) + 4
```

- 中文 ≈ 1.5 tokens/字
- 英文/数字 ≈ 0.5 tokens/字
- +4 为格式开销

**截断策略**：从最近的消息往前累加，到 **90 万 tokens** 停止（留 10 万余量给 prompt 模板和 wiki 输出）。

**效果**：
- 短消息用户（条均 20 tokens）→ 可能传 **4-5 万条**
- 长消息用户（条均 200+ tokens）→ 可能只传 **4-5 千条**
- 自动适配，最大化利用 1M 窗口

### 4.4 超限用户自动分轮次更新

如果用户全部历史消息的 token 总数 **超过 90 万**，自动切成多批，**从旧到新逐批增量更新**：

```python
batches = split_by_tokens(all_msgs, max_tokens=900_000)
for batch_idx, batch in enumerate(batches):
    task["chat_name"] = f"... (批次 {batch_idx+1}/{len(batches)})"
    engine._do_update_user(task)  # 每轮自动读取上一轮生成的 wiki 作为 current_wiki
```

**为什么从旧到新？**
- 第一轮（最早历史）→ LLM 建立"基础画像"（性格、职业、长期偏好）
- 第二轮（中期）→ 补充经历、关系变化
- 最后一轮（最近）→ 更新"近期动态"，覆盖过期内容

**数据**：1490 活跃用户中仅 **2 人**需要分轮次（1 人需 2 轮，1 人需 3 轮），其他全部一轮搞定。

### 4.5 全局聚合 + 聊天分隔

一个用户可能出现在 N 个私聊和 M 个群聊中。策略：

1. **收集所有聊天的消息**，为每条消息标注来源 `chat_name`
2. **全局按时间排序**（旧 -> 新），混在一起
3. **按 token 估算截断**（90 万 tokens）或分轮次
4. **在 conversation 中自动插入分隔线**：当 `chat_name` 变化时，输出 `===== {chat_name} =====`

**效果示例**：

```
===== 私聊：严小严🎈 中山医院 =====
[2024-01-15 10:00]我：明天有空吗？
[2024-01-15 10:05]严小严：有，什么事？

===== 群聊：示例社区群 =====
[2024-01-15 14:00]示例用户甲：明天桌游有人来吗？
[2024-01-15 14:05]严小严：我去

===== 私聊：富比 =====
[2024-01-16 09:00]富比：在吗？
```

**优点**：
- 每用户只调 1-3 次 LLM（分轮次时），不会每个聊天都调
- 不同聊天在 conversation 中自然隔开，LLM 不会混淆语境
- 全局时间顺序保持，能反映用户一天内的活动轨迹

---

## 5. LLM 调用与 Prompt 设计

### 5.1 模型参数
| 参数 | 值 | 说明 |
|------|-----|------|
| model | deepseek-v4-flash | 1M 上下文，性价比高 |
| temperature | 0.3 | 低随机性，保证事实稳定 |
| max_tokens | 2000 | wiki 输出上限 |
| timeout | 6000s | 处理 2 万条消息可能需要数分钟 |

### 5.2 并发策略
- `ThreadPoolExecutor(max_workers=10)`
- 每个 worker 独立调用 LLM
- `engine.py` 中 `_merge_aliases` 加线程锁，防止并发写 aliases.json 冲突

### 5.3 Prompt 模板结构

#### 用户 wiki Prompt (`_UPDATE_PROMPT`)
```
请根据以下对话记录，更新用户 wiki。

【现有 wiki】
{current_wiki}

【新对话】
聊天：{chat_name}
时间：{current_time}

对话内容：
{conversation}

【更新规则】
1. 只修改/新增变化的部分，保留未变动的内容
2. 标注日期：日期必须严格来自对话记录开头的时间戳。禁止编造、推测、推断任何日期
3. 时间戳缺失：无法确定日期时不标注或用 [待验证] 标记
4. 过期处理：超过 7 天的"近期动态"移到"说过的话"或删除
5. 冲突处理：新信息覆盖旧信息
6. 不确定的信息用 [待验证] 标记
7. 多账号标注：如果对话来源包含不同账号标记，标注所属账号
8. 控制长度：个人 wiki 不超过 1500 字
9. 保持 Markdown 格式
10. 别名发现（严格）：只记录当前用户本人的其他称呼。严禁记录其他人的名字。
    格式：`- 别名：xxx` 或 `- 别名：xxx（来源：某群/某人称呼）`

【输出】直接输出更新后的完整 wiki markdown，不要加代码块标记。
```

#### 群聊 wiki Prompt (`_UPDATE_GROUP_PROMPT`)
类似结构，重点记录群成员画像、热点话题、群内文化、规则禁忌。

### 5.4 Prompt 持久化
每次生成 wiki 后，实际使用的 prompt（截断中间过长部分）保存到：
```
data/memory/wiki/prompts/users/{user_name}.md
data/memory/wiki/prompts/groups/{group_name}.md
```
方便排查"为什么这个 wiki 生成了未来日期/错误别名"等问题。

---

## 6. 别名系统

### 6.1 三层别名来源

| 层级 | 来源 | 写入时机 |
|------|------|---------|
| L1 人工标注 | `data/memory/overrides/aliases.json` | 人工维护 |
| L2 自动发现（sender 字段） | `update_aliases_json()` 从聊天记录中统计出现次数≥3 的昵称 | 批量导入前 |
| L3 LLM 推断 | 从生成的 wiki `## 别名` 段落解析 | 每次 LLM 生成后 |

### 6.2 别名提取过滤 (`_extract_aliases_from_user_wiki`)
从 wiki 的 `## 别名` 段落解析后，代码层再做严格过滤：
- 排除长度 > 30 的（不是别名）
- 排除包含中文动词/标点的（是句子而非别名）
- 排除已作为其他用户主名的（防止把他人名字误当别名）

### 6.3 别名建议持久化
LLM 发现的别名通过 `_save_alias_suggestion` 保存为 JSON：
```json
{
  "main_name": "示例用户甲",
  "aliases": ["示例用户辛", "示例用户丙", "han"],
  "source": "user",
  "generated_at": "2026-05-10 17:50:13"
}
```
保存路径：`data/memory/wiki/alias_suggestions/{users,groups}/{name}.json`

---

## 7. 文件输出结构

```
data/memory/wiki/
├── users/                          # 用户 wiki
│   ├── 示例用户甲.md
│   ├── 严小严🎈 中山医院.md
│   └── ...
├── groups/                         # 群聊 wiki
│   ├── 📮美港股价值投资群.md
│   └── ...
├── topics/                         # 话题 wiki（关键词聚类）
│   ├── 股票投资.md
│   └── ...
├── prompts/                        # 每次生成的 prompt 记录
│   ├── users/示例用户甲.md
│   └── groups/📮美港股价值投资群.md
└── alias_suggestions/              # 别名建议（可人工审核后导入 aliases.json）
    ├── users/示例用户甲.json
    └── groups/...
```

---

## 8. 参数清单

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--min-user-msgs` | 20 | 生成用户 wiki 的消息量阈值 |
| `--min-group-msgs` | 30 | 生成群聊 wiki 的消息量阈值 |
| `--max-msgs-per-chat` | 20000 | 单个聊天最多取多少条 |
| `--max-total-msgs` | 20000 | 单个 wiki 更新传入 LLM 的最大消息数 |
| `--workers` | 10 | 并发 LLM 调用数 |
| 群聊上下文 k | 10 | 目标消息前后各取 10 条 |
| 群聊时间窗口 | 5min | 目标消息 ±5 分钟内的所有消息 |

---

## 9. 已知问题与历史修复

| 问题 | 原因 | 修复 |
|------|------|------|
| 私聊误放群聊 | `"" in text` 恒为 True | 改为 `if not text` |
| 未来日期 | prompt 未约束日期来源 | 添加"日期必须来自时间戳，禁止编造" |
| 别名混入他人名字 | prompt 表述模糊 + 无过滤 | prompt 明确"只记录本人" + 代码层过滤 |
| 群聊名称进 users | `@chatroom` 消息未排除 | `build_wxid_index` 排除 `@chatroom` wxid |
| 缺少上下文 | 只传了用户自己的消息 | 私聊完整对话 + 群聊分层上下文 |
