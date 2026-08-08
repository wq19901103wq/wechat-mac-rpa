# 排查文档索引

> **使用规则**：遇到问题 → 先查本文档索引 → 找到对应排查文档 → 按文档步骤执行 → 排查结束后更新文档。
> 
> **禁止**：不看文档直接猜测、凭经验推断、凭记忆排查。

---

## 按模块索引

| 模块 | 问题类型 | 排查文档 | 上次更新 |
|------|----------|----------|----------|
| **Tick 全流程** | 某个 tick 异常（action 不符合预期、消息丢失、昵称误判等） | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | 2026-04-20 |
| **Tick 快速扫描** | 扫描最近 100 个 tick 的整体健康状况 | [TICK_INVESTIGATION_GUIDE.md](TICK_INVESTIGATION_GUIDE.md) | 2026-05-05 |
| **Memory Search** | search_memory 召回不符合预期、只返回本人、关系查询失败 | [MEMORY_SEARCH_TROUBLESHOOTING.md](MEMORY_SEARCH_TROUBLESHOOTING.md) | 2026-05-12 |
| **Runtime 异常** | 启动失败、崩溃、死锁 | [RUNTIME_INVESTIGATION.md](RUNTIME_INVESTIGATION.md) | 2026-05-03 |
| **已知解决方案** | 已验证有效的修复方案汇总 | [SOLUTIONS.md](SOLUTIONS.md) | 2026-05-03 |
| **修复协议** | 强制执行的修复流程，防止修 A 坏 B | [FIX_PROTOCOL.md](FIX_PROTOCOL.md) | 2026-05-16 |
| **踩坑记录** | 历史踩坑及教训 | [LESSONS_LEARNED.md](LESSONS_LEARNED.md) | 2026-05-16 |

---

## 按症状快速定位

| 症状关键词 | 对应文档 | 章节 |
|-----------|----------|------|
| chat_name 为空 | TROUBLESHOOTING.md | 症状 A |
| messages 数量少 | TROUBLESHOOTING.md | 症状 B |
| 昵称误判 sender 不对 | TROUBLESHOOTING.md | 症状 C |
| action='none' 但应有 switch | TROUBLESHOOTING.md | 症状 D |
| action='none' 但应有 send | TROUBLESHOOTING.md | 症状 E |
| OCR 为空 | TROUBLESHOOTING.md | 症状 F |
| screenshot 路径异常 | TROUBLESHOOTING.md | 症状 G |
| 微信已打开但截图失败、隐藏或禁止捕获 | TROUBLESHOOTING.md | 症状 H |
| **search_memory 返回空** | **MEMORY_SEARCH_TROUBLESHOOTING.md** | **症状 A** |
| **只返回本人 wiki** | **MEMORY_SEARCH_TROUBLESHOOTING.md** | **症状 B** |
| **返回不相关人物** | **MEMORY_SEARCH_TROUBLESHOOTING.md** | **症状 C** |
| **结果被截断** | **MEMORY_SEARCH_TROUBLESHOOTING.md** | **症状 D** |
| **关系查询失败** | **MEMORY_SEARCH_TROUBLESHOOTING.md** | **症状 E** |

---

## 排查流程（强制执行）

```
用户报告问题
    │
    ▼
1. 打开本文档 INDEX.md
    │
    ▼
2. 根据症状关键词，找到对应排查文档
    │
    ▼
3. 打开排查文档，按流程执行（看日志 → 跑验证脚本 → 确认根因）
    │
    ▼
4. 如果排查文档未覆盖 → 进入该文档的"未知异常"章节 → 全面提取数据
    │
    ▼
5. 根因确认后 → 修复 → 添加测试 → 更新排查文档（速查表、验证脚本）
    │
    ▼
6. 更新 INDEX.md 的"上次更新"时间戳
```

---

## 文档更新记录

| 日期 | 更新人 | 更新内容 |
|------|--------|----------|
| 2026-05-12 | AI | 新增 MEMORY_SEARCH_TROUBLESHOOTING.md，更新索引 |
