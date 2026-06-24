# Memory Module Spec

> 重构于 2026-06-24，对标 Karpathy LLMWiki 架构（gist: karpathy/442a6bf555914893e9891c11519de94f）。
> 核心转变：wiki 是 **LLM 编译的摘要中间层**，不是对话流水账存储。中等规模下不依赖 RAG，
> 靠 LLM 维护的索引+摘要检索。详见 `docs/03-guides/MEMORY_SYSTEM.md`。

## 1. 模块职责

管理 Bot 的长期记忆，遵循 LLMWiki 三层模型：
- **raw 层（不可变）**：原始对话消息，LLM 只读不改，是事实来源。
- **wiki 层（LLM 编译产物）**：结构化 markdown——实体页、群页、摘要、交叉引用。LLM 写，人读。
- **schema 层**：本 spec + prompt 模板，指导 LLM 如何 ingest / query / lint。

提供四个操作：Ingest（编译更新）、Query（检索回答）、Lint（健康检查）、Alias（别名管理）。

## 2. 功能需求 (FR)

### Ingest 编译更新
- **FR-1**: `update_user_wiki()` / `update_group_wiki()`：把更新任务加入队列，后台异步执行。
  LLM 读取新对话 → 提取要点 → **整合进现有 wiki**（更新实体页、修订摘要、标记冲突），而非 append 原文。
- **FR-2**: 近期动态记**事件摘要**（每人每天最多 1-2 条，只记发生了什么，不记原文对话）。
  超过 7 天的近期动态必须删除或并入"说过的话"。身份事实（姓名/职业/关系/偏好/MBTI）不删。

### Query 检索
- **FR-3**: `get_user_memory(user_name)`：读取用户 wiki（含别名合并 + 外挂 facts），返回压缩摘要。
- **FR-4**: `get_group_memory(group_name)`：读取群 wiki（含外挂 corrections）。
- **FR-5**: `search_keyword(keyword)`：BM25 搜索所有 wiki。命中本人返回完整 wiki，命中别人返回片段。
- **FR-6**: `search_related_mentions(text)`：扫描文本中提到的人名，加载这些人自己的 wiki。

### Lint 健康检查 ⭐ 新增
- **FR-7**: `lint_memory()`：定期扫描记忆库，产出问题清单：
  - 矛盾别名（同一别名指向多个主名）
  - 膨胀 wiki（超长度上限的文件）
  - 孤立页面（无入链的 user/group wiki）
  - 重复群（归一化后同名）
  - 过时近期动态（超 7 天未清理）
- **FR-8**: Lint 产出可写回：自动截断膨胀 wiki、合并重复群、标记矛盾别名供人工审核。

### Alias 别名管理
- **FR-9**: 别名自动发现：从 LLM 生成的 wiki 提取别名，经校验后入库。
- **FR-10**: 别名**拆分**：`"老王、王总"` 必须拆成 `["老王", "王总"]`，禁止整串入库。
- **FR-11**: 别名**校验**：拒绝角色词（Bot/对话中/匿名/群主/记录者）、房号（4-1-703/6幢5号501）、
  描述句（"被群友称为..."）、含标点、过长（>15字）、微信ID（wxid_/@chatroom）。
- **FR-12**: 别名**幂等去重**：merge 前拆分去重，禁止存重叠长串。

### 归一化与外挂
- **FR-13**: `normalize_chat_name(name)`：群名归一化（去 emoji、折叠空格）用于路径计算，
  避免重复群（3D打印/D打印、築岛空格差异）。
- **FR-14**: 广告群黑名单：归一化后命中（如"百果园"+斤价模式）的群不生成 wiki。
- **FR-15**: 外挂配置加载：`aliases.json`、`facts.json`、`corrections.json`。

## 3. 非功能需求 (NFR)

- **NFR-1**: Wiki 更新异步执行，不阻塞主 tick。Worker 每 5 秒检查队列，批量处理（每批最多 3 条，或积压超 5 分钟的全部处理）。
- **NFR-2**: **代码级长度护栏** `enforce_wiki_limits(wiki, max_chars=4000)`：落盘前强制截断。
  超长时按 `## ` 切 section，优先从"近期动态""说过的话"底部（最老的）砍，身份/关系 section 保留。
  ⚠️ 不再依赖 LLM 自律，代码兜底。
- **NFR-3**: 个人/群 wiki 不超过 4000 字。LLM prompt 约束 + NFR-2 代码兜底双保险。
- **NFR-4**: LLM 生成 wiki 带 3 次重试：429 配额（指数退避）、400 输入超长（截断 conversation）。
- **NFR-5**: `tick_log` 保留 7 天，超期删除 + 定期 VACUUM。无效 tick（skip_reason 非空）不写全量 `session_input_messages_json`，只记摘要。
- **NFR-6**: 别名/群名归一化在所有路径计算处统一调用，禁止裸用 chat_name 当文件名。

## 4. 接口契约

### 输入
```python
MemoryEngine(llm_client=None)

# Ingest
update_user_wiki(user_name, chat_name, messages, bot_replies)
update_group_wiki(group_name, chat_name, messages, bot_replies)

# Query
get_user_memory(user_name: str, max_chars: int = 2000) -> str
get_group_memory(group_name: str, max_chars: int = 2000) -> str
search_keyword(keyword: str, max_chars: int = 6000) -> str
search_related_mentions(text: str, exclude_user=None, max_files=5) -> List[str]

# Lint (新增)
lint_memory() -> dict   # {conflicts, bloated, orphans, duplicates, stale}

# 归一化 (新增)
normalize_chat_name(name: str) -> str
```

### 输出
- Wiki 摘要：Markdown，可能含 `（…记忆已截断）`
- 搜索结果：带标签片段，如 `【某某的记忆】...`
- Lint 报告：结构化 dict

## 5. 核心规则与约束

### 规则 1: 增量更新——编译而非 append（修订）
**旧规则"严禁删除任何内容"已废止**——它与长度约束矛盾，导致膨胀。新规则：
- **身份事实**（姓名/职业/关系/偏好/MBTI）：增量保留，不主动删除。新信息覆盖旧信息，冲突标 `[待验证]`。
- **近期动态**：滚动窗口，7 天前的必须删除或并入"说过的话"。只记事件摘要，不记原文对话。
- 所有事实标注来源 `（来源：某群/日期）`。

### 规则 2: 别名校验三步走（修订）
入库前必经：① 拆分（顿号/斜杠/空格）→ ② 校验（黑名单/房号/句子/标点/长度）→ ③ 去重。
别名不能是其他人的主名。详见 FR-10/11/12。

### 规则 3: 区分陈述和疑问
以"吗""呢""?"结尾是疑问，严禁当事实提取。

### 规则 4: Facts 放在 Wiki 前面
`get_user_memory` 返回时，外挂 facts 放在 LLM wiki 前，确保截断不丢人工标注。

### 规则 5: BM25 搜索本人优先
命中本人返回完整 wiki；命中别人只返回片段。

### 规则 6: 群名归一化（新增）
所有 wiki 路径、aliases 查找、search 召回统一用 `normalize_chat_name`，避免重复群。

## 6. 错误处理

| 情况 | 处理 |
|------|------|
| LLM 生成失败（3 次重试后） | 记录 error，不更新 wiki |
| Wiki 文件读写失败 | 记录 warning，返回空字符串 |
| 别名解析冲突 | 优先匹配主名，冲突记 debug |
| Lint 发现膨胀 wiki | 自动截断 + 记录 |
| Lint 发现重复群 | 标记供人工合并，不自动删 |

## 7. 依赖关系
- 依赖 LLM 客户端（wiki 更新 / lint 时）
- 被 `src.bot.WeChatBot` 和 `src.reply.generator.ReplyGenerator` 调用
- `scripts/sanitize_aliases.py` 一次性迁移脚本（已应用）
- `scripts/lint_memory.py` 定期 lint 脚本（待实现）

## 8. 重构进度

- ✅ P0-A: 别名拆分+校验+清洗（FR-9~12），1217→1039 别名，"王总"召回修复
- ✅ P0-B: wiki 膨胀（FR-1/2, NFR-2/3）— prompt 修订（编译摘要非流水账）+ `enforce_wiki_limits` 代码护栏
- ✅ P1 Lint 操作（FR-7/8）— `lint_memory()` + `scripts/lint_memory.py`，首次报告清零
- ✅ P1 别名冲突裁决 — `scripts/resolve_alias_conflicts.py`，21 冲突→0，删 62 脏/幽灵主名
- ✅ P1 wiki 截断 — 阈值统一为 4000 字（与 NFR-2/3 一致），152 个超限文件截断（备份在 `.lint-bak`）
- ✅ P1 cases.db 清理（NFR-5）— `scripts/cleanup_cases_db.py`，2.86GB→1.5GB（删 7 天前 + VACUUM）；bot 写入改为只存最近 50 条上下文（原存全量 2078 条/927KB）
- ✅ P1 归一化+广告拦截（FR-13/14）— `normalize_chat_name` + 斤价模式不入库
- ✅ P2: humor RAG 移除 — `MessageVectorIndex`（TF-IDF）索引已废弃、静默失效，删除 generator 注入逻辑与 vector_index.py 的 MessageVectorIndex 类；历史原文检索统一由 `search_history`（BGE dense + keyword 两路融合，见 history_search.py）承担
- ✅ 召回质量 — 人名查询时 user wiki ×1.3 boost（避免 group wiki BM25 挤占），benchmark 5 个 known_issue 修 4 个；剩 multi_wangqiaosheng_biaoge 是 wiki 内容缺失（王海 wiki 未提王乔生）非排序问题
- ✅ LLM rerank — search_keyword 的 BM25 top10 候选调 llm_client 按语义重排（temperature=0），降级完善（无 llm_client/异常→回退 BM25）。只修排序不修召回。
- ✅ benchmark 三层指标 — 召回（P/R/F1）+ 排序（MRR@5/Hit@5）+ 召回阶段（召回池Hit@30，rerank 前）。wiki 68 case（MRR@5 77%/Hit@5 91%/召回池 100%），history 21 case（MRR@5 100%/Hit@5 100%/召回池 100%）。
- ⏳ 残留: 6 组重复群（emoji-only 名）、8 广告群 wiki 待删
- ⏳ 待办: LLM rerank 候选 top10→top30（捞回 pool=Y11-30 的 primary）；pool=N 真召回问题需 query 改写/同义词扩展（非 rerank）
