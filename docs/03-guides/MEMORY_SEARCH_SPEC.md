# Memory Search Spec — `search_keyword` 召回与排序

## 1. 召回阶段（Retrieval）

遍历 `wiki_dir/{users,groups}/*.md`，把所有非空文档加载到内存：

```
文档内容 = facts_text + "\n\n" + wiki_text
```

- **users**: `facts`（外挂事实）+ `wiki`（LLM 生成的 markdown）
- **groups**: `corrections`（纠正列表）+ `wiki`

召回**不做过滤**，所有文档进入下一步打分。

## 2. 关键词扩展

输入 `keyword` 按空格分词，去掉长度 < 2 的词：

```
raw_keywords = [kw.strip() for kw in keyword.split() if len(kw.strip()) >= 2]
```

每个 `raw_keyword` 通过 `_expand_search_keywords()` 扩展：

```
keywords = [主名] + [所有别名]
```

例：输入 `"示例用户甲 同事"` → `raw_keywords = ["示例用户甲", "同事"]` → `keywords = ["示例用户甲", "同事"]`（假设示例用户甲没有别名需要扩展）

`resolved_keyword = _resolve_alias(raw_keywords[0])`，用于后续 primary 判定。

## 3. 排序阶段（BM25 Ranking）

对每个文档计算 BM25 分数：

```
score = Σ idf(q) * f(q) * (k1 + 1) / (f(q) + k1 * (1 - b + b * dl / avgdl))
```

- `f(q)`: 关键词 q 在文档中的出现次数
- `idf(q) = log((N - n(q) + 0.5) / (n(q) + 0.5))`
- `N`: 总文档数
- `n(q)`: 包含关键词 q 的文档数
- `dl`: 文档长度
- `avgdl`: 平均文档长度
- `k1 = 1.5`, `b = 0.75`

**score = 0 的文档直接丢弃。**

## 4. Primary 判定

Primary 文档 = "本人文档"，返回时给**完整 wiki**。

```python
name_match     = (_resolve_alias(文档名) == _resolve_alias(resolved_keyword))
substring_match = (len(resolved_keyword) >= 2 and resolved_keyword in 文档名)
is_primary     = ((name_match or substring_match) and not 群文档)
```

| 条件 | 说明 |
|------|------|
| `name_match` | 文档名经别名解析后 == 搜索词经别名解析后 |
| `substring_match` | 搜索词是文档名的**子串**（`搜索词 in 文档名`）|
| `not 群文档` | 群 wiki 永远不会是 primary |

**陷阱**: `substring_match` 用的是 `resolved_keyword in name`，不是 `name in resolved_keyword`。所以搜 "示例用户甲" 时：
- ✅ "示例用户甲.md" → primary（name_match）
- ✅ "示例用户甲_2.md" → primary（substring_match: "示例用户甲" in "示例用户甲_2"）
- ✅ "示例用户甲、示例用户午、示例用户甲.md" → primary（substring_match: "示例用户甲" in "示例用户甲、示例用户午、示例用户甲"）
- ❌ "王.md" → **不是 primary**（"示例用户甲" not in "王"）

## 5. 排序规则

```python
scored.sort(key=lambda x: (not x.is_primary, -x.score))
```

1. **Primary 文档强制排在前面**（无论 score 高低）
2. 同优先级内按 BM25 score **降序**

**后果**: 如果有多个 primary 文档，它们会占据 Top N 的多个位置，可能把高分的 non-primary 文档挤出。

## 6. Top-N 截断

只取 `scored[:5]`，即**最多 5 个文档**。

- Primary 文档 → 返回**完整 wiki**（不截断）
- Non-primary 文档 → 调用 `_extract_all_snippets()` 提取片段

## 7. Snippet 提取（`_extract_all_snippets`）

对每个 non-primary 文档，按 `keywords` 顺序搜索，提取包含关键词的上下文片段：

```
片段范围 = [命中位置 - 80, 命中位置 + 150]
去重：±50 字范围内视为重叠，只保留第一个
上限：max_snippets（默认 2）
```

**问题**: 按 keyword 顺序提取，如果第一个 keyword 就在文档中出现了 2 次，直接 break，后续 keyword 的片段不会被提取。

例：文档里 "同事" 出现 30 次，"示例用户甲" 出现 1 次。如果 "同事" 在 keywords 列表中排在前面，可能提取的 2 个片段都围绕 "同事"，完全不包含 "示例用户甲"。

## 8. 结果截断（`max_chars`）

默认 `max_chars = 6000`：

1. 先保留所有 primary 文档的完整 wiki
2. 再按顺序追加 non-primary 的 snippet
3. 超出长度时截断，追加 "（…更多结果省略）"

## 9. 已知问题清单

| 问题 | 现象 | 根因 |
|------|------|------|
| Primary 过多挤占 Top 5 | 搜 "示例用户甲 同事" 只返回示例用户甲自己的 wiki，示例用户乙/Alex 被挤出 | 4 个文件名含 "示例用户甲" 的文档都被判为 primary，占了 Top 4 |
| Snippet 关键词偏置 | 提取的片段可能只包含部分关键词 | `_extract_all_snippets` 按 keyword 顺序提取，达到上限后 break |
| Top 5 太小 | 高分相关文档被截断 | 硬编码 5，无动态调整 |

## 10. 修复方向（待讨论）

1. **限制 primary 数量**: 只把 exact name_match 的文档判为 primary，substring_match 不再享受置顶待遇
2. **增大 Top N**: 从 5 改为 10 或按 `max_chars` 动态决定
3. **改进 snippet 提取**: 提取时保证每个 keyword 至少覆盖一次，或按相关性（离其他关键词近）排序
