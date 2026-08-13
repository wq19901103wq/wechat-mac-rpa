# Memory Search 排查手册

> 当 `search_memory` / `search_keyword` 的召回结果不符合预期时，按本文档系统化诊断。
>
> **原则**：先看日志 → 再跑数据 → 最后才下结论。严禁猜测。

---

## 诊断流程图

```
用户反馈 search_memory 结果不对
    │
    ▼
┌─────────────────────────────┐
│ 1. 查日志 [Search]          │ ← grep "\[Search\]" data/logs/bot_*.log
│    确认 keyword / 召回数    │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ 2. 对照症状速查表           │ ← 本文档 §2
│    定位问题类别             │
└─────────────────────────────┘
    │
    ├──→ 匹配到已知症状 → 按 §3 深度验证 → 确认根因 → 修复/记录
    │
    └──→ 未匹配 → 进入 §4 全面数据提取 → 人工分析 → 更新速查表
```

---

## 1. 查日志：第一步永远是看日志

### 1.1 快速定位相关日志

```bash
grep -h "\[Search\]" data/logs/bot_*.log | tail -30
```

**关键日志字段及含义：**

| 日志 | 含义 | 正常值示例 |
|------|------|-----------|
| `keyword='...' raw=[...] expanded=[...] resolved='...'` | 搜索词及扩展后的关键词 | `keyword='示例用户甲 同事' raw=['示例用户甲','同事'] expanded=['示例用户甲','关系别名甲',...,'同事']` |
| `retrieved N docs` | 召回的文档总数 | 应接近 wiki 文件总数 |
| `N=1473 avgdl=2249.1` | 总文档数、平均长度 | — |
| `doc_has_q={...}` | 每个关键词命中的文档数 | 不应全为 0 |
| `idf={...}` | 每个关键词的 IDF 值 | 不应为 0（否则该关键词不贡献分数） |
| `primary doc: {name} score=...` | 被判定为 primary 的文档 | 应有且仅有 1 个（精确别名匹配） |
| `non-primary top: {name} score=...` | 高分 non-primary | 应包含预期人物 |
| `selected top N docs` | 最终进入 Top N 的文档数 | 应为 10 |
| `select: {name} score=... primary=...` | 每个被选中的文档 | 检查预期人物是否在列 |
| `snippets for {name}: N extracted` | 片段提取数量 | 0 表示该文档无有效片段 |
| `raw results length=...` | 原始结果长度 | 对比 max_chars 判断是否触发截断 |

### 1.2 一键输出 search 诊断摘要

```python
import re, glob

paths = sorted(glob.glob("data/logs/bot_*.log"))[-3:]  # 最近 3 个日志
for p in paths:
    lines = open(p).readlines()
    for i, line in enumerate(lines):
        if "[Search] keyword" in line:
            print(f"\n=== {p.split('/')[-1]} ===")
            # 打印该 keyword 相关的全部 Search 日志
            for j in range(i, min(i+20, len(lines))):
                if "[Search]" in lines[j]:
                    print(lines[j].strip())
```

---

## 2. 症状速查表

### 症状 A：返回"未在本地记忆中找到关于'xxx'的信息"

**识别**：日志中 `scored count=0` 或没有 `select:` 日志。

**可能根因**：
1. **IDF 为负**：所有文档都包含关键词，IDF < 0，被 `idf[q] > 0` 过滤 → score 全为 0
2. **关键词扩展错误**：`_expand_search_keywords` 返回空列表或错误别名
3. **wiki 目录为空**：`retrieved 0 docs`
4. **resolved_keyword 解析错误**：别名映射导致搜索词被解析成不存在的人名

**快速验证**：
```python
# 检查 doc_has_q 和 idf
# 日志中应有 idf={...}，如果全为 0 → IDF 为负问题
# 如果 retrieved 0 → wiki 目录为空
```

**修复**：
- IDF 为负 → 改用 `max(log(...), 0.01)` 兜底
- 关键词扩展错误 → 检查 `aliases.json`

---

### 症状 B：只返回本人 wiki，没有返回相关人物

**识别**：日志中只有 1 个 `select:`（本人的），预期相关人物没有出现在 `non-primary top:` 或 `select:` 中。

**可能根因**：
1. **primary 过多挤占 Top N**：多个文件名包含搜索词的文档都被判为 primary，占了 Top N 位置
2. **Top N 太小**：只有 5 个位置，primary 占了 4 个，只剩 1 个给 non-primary
3. **相关人物的 wiki 里不包含搜索关键词**：比如搜"示例用户甲 同事"，但示例用户乙的 wiki 里写的是"一起工作"而不是"同事"
4. **相关人物的 score 太低**：文档太长、关键词出现频率低，BM25 分数被挤出 Top N

**快速验证**：
```bash
# 1. 检查日志中 primary 数量
grep "\[Search\] primary doc:" data/logs/bot_*.log | tail -10

# 2. 检查相关人物的 wiki 是否包含关键词
grep -l "示例用户甲\|同事" data/memory/wiki/users/*.md | head -10

# 3. 检查相关人物的 score 排名
grep "\[Search\] non-primary top:" data/logs/bot_*.log | tail -10
```

**修复**：
- primary 过多 → 取消 `substring_match` 的 primary 资格，只允许 `name_match`
- Top N 太小 → 从 5 改到 10
- 关键词不匹配 → 需要改进 wiki 生成策略，确保关系词统一

---

### 症状 C：返回了不相关的人物

**识别**：结果中包含预期之外的人物，且该人物与查询无关。

**可能根因**：
1. **关键词扩展过宽**：`_expand_search_keywords` 把别名也加进搜索词，导致不相关文档被召回
2. **snippet 提取偏差**：`_extract_all_snippets` 按关键词顺序提取，前 2 个片段可能不包含最相关的上下文
3. **BM25 分数接近**：文档长度短、关键词频率高，导致噪声文档 score 高于真实相关文档

**快速验证**：
```python
# 检查 expanded keywords 是否过多
# 日志中 expanded=[...] 不应包含明显不相关的别名

# 检查噪声文档的内容
grep -n "示例用户甲\|同事" data/memory/wiki/users/噪声文档.md
```

**修复**：
- 别名扩展过宽 → 限制 `_expand_search_keywords` 只扩展搜索词本身的别名，不扩展其他用户的别名
- snippet 偏差 → 改进 snippet 提取逻辑，按多关键词共现密度排序

---

### 症状 D：结果被截断，关键信息丢失

**识别**：结果以"（…更多结果省略）"或"（…内容截断）"结尾，且用户需要的具体信息不在结果中。

**可能根因**：
1. **max_chars 太小**：默认 6000，但 primary 文档很长，占用了大部分空间
2. **primary 文档过多**：多个 primary 各占一段完整 wiki，快速耗尽 max_chars
3. **snippet 过长**：每个 snippet 取 [idx-80, idx+150]，平均 230 字符，2 个就是 460 字符

**快速验证**：
```bash
# 日志中应有 raw results length=... max_chars=...
# 对比两者判断是否触发截断
```

**修复**：
- max_chars 太小 → 增大到 8000 或 10000
- primary 过多 → 限制为 1 个 primary
- snippet 过长 → 缩短 snippet 窗口或动态调整

---

### 症状 E：跨人物关系查询失败（如"我同事有哪些"）

**识别**：用户问关系型问题，但只返回了本人的 wiki，没有返回关系另一方的信息。

**可能根因**：
1. **关系信息分散**：A 的 wiki 里写了"与 B 为同事"，但 B 的 wiki 里没有写"与 A 为同事"
2. **搜索词设计问题**：搜"示例用户甲 同事"只能召回包含这两个词的文档，无法召回只写"与示例用户甲共事"的文档
3. **跨文档关联缺失**：BM25 是单文档匹配，不支持跨文档 join

**快速验证**：
```bash
# 检查哪些人的 wiki 里写了关系
grep -l "示例用户甲.*同事\|同事.*示例用户甲" data/memory/wiki/users/*.md

# 检查本人的 wiki 里有没有同事列表
grep -n "同事" data/memory/wiki/users/示例用户甲.md
```

**修复**：
- 关系分散 → 在 wiki 生成 prompt 中明确要求"在本人 wiki 中汇总所有已知关系"
- 搜索词设计 → 考虑支持关系类型单独搜索（如只搜"同事"）
- 跨文档关联 → 长期方案：建立关系图谱索引

---

## 3. 深度验证方法

### 3.1 验证 keywords 扩展

```python
from src.memory.engine import MemoryEngine
engine = MemoryEngine()

keyword = "示例用户甲 同事"
raw = [kw.strip() for kw in keyword.split() if len(kw.strip()) >= 2]
expanded = []
for kw in raw:
    expanded.extend(engine._expand_search_keywords(kw))
print(f"raw: {raw}")
print(f"expanded ({len(set(expanded))} unique): {list(dict.fromkeys(expanded))[:20]}")
# 如果 expanded 包含大量无关别名，说明扩展过宽
```

### 3.2 验证 BM25 打分

```python
import math

# 在 search_keyword 中插入以下代码（或看日志）
# 重点关注：
# - doc_has_q: 关键词命中的文档数
# - idf: 是否接近 0
# - 目标文档的 score 是否在 Top 10 内
```

### 3.3 验证 primary 判定

```python
resolved = engine._resolve_alias("示例用户甲")
for path in sorted(engine.wiki_dir.glob("users/*.md")):
    name = path.stem
    name_match = engine._resolve_alias(name) == resolved
    substring = len(resolved) >= 2 and resolved in name
    print(f"{name}: name_match={name_match}, substring={substring}")
# name_match=True 的才应是 primary
# substring=True 但 name_match=False 的不应再是 primary
```

### 3.4 验证 snippet 提取

```python
content = open("data/memory/wiki/users/示例用户乙-远舟.md").read()
keywords = ["示例用户甲", "同事"]
snippets = engine._extract_all_snippets(content, keywords, max_snippets=2)
for s in snippets:
    print(f"--- snippet ({len(s)} chars) ---")
    print(s[:300])
# 检查：片段是否包含关系描述？是否被截断？
```

---

## 4. 未知异常：全面数据提取

如果速查表未匹配，运行以下脚本提取 search 的全部中间状态：

```python
from src.memory.engine import MemoryEngine
from pathlib import Path
import math

engine = MemoryEngine()
keyword = "示例用户甲 同事"

# 复现 search_keyword 的完整流程并打印每一步
# （可直接在 engine.py 的 search_keyword 中插入 print，或看日志）
```

---

## 5. 根因确认后

1. **如果是配置/参数问题**（如 Top N 太小、max_chars 太小）→ 修改 `engine.py` → 添加单元测试 → 更新 spec
2. **如果是代码 bug**（如 primary 判定过宽、IDF 为负）→ 修复代码 → 添加单元测试 → 更新 spec
3. **如果是 wiki 内容问题**（如关系信息分散、关键词不一致）→ 更新 wiki 生成 prompt → 批量重生成
4. **如果是未知问题** → 保存问题 case → 添加回归测试 → 更新本文档速查表
