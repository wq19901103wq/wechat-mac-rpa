# 项目行为守则

> 本文档是守则总纲。具体规则按主题和职责拆分为子守则，AI Agent 和人类开发者应根据当前分工按需读取对应子守则，避免上下文腐化。
> 
> **没有固定期限，只有两条原则：发现问题立即修，修不了立即说。**

---

## 子守则索引（按分工）

| 你的分工 | 读取子守则 | 覆盖内容 |
|--------|-----------|---------|
| **测试/QA / 修复 bug / 跑回归** | [standards/STANDARDS_TESTING_BUGS.md](standards/STANDARDS_TESTING_BUGS.md) | 测试错误零容忍、P0-P2 优先级、CI 红线、错误资产化、待修复清单 |
| **架构设计 / 写文档 / 更新接口** | [standards/STANDARDS_DOCUMENTATION.md](standards/STANDARDS_DOCUMENTATION.md) | 文档同步政策、强制进度标签（🚧未实现 → 🚀已上线） |
| **写代码 / 提交 / 分支管理 / 合并** | [standards/STANDARDS_CODING.md](standards/STANDARDS_CODING.md) | 一个提交一件事、禁止直推 main、分支存活期 ≤7 天 |

---

## 快速决策表

**当你要...** → **先读...**

- 跑测试 / 发现测试红了 / 修 bug / 确认 bug 收尾 → [standards/STANDARDS_TESTING_BUGS.md](standards/STANDARDS_TESTING_BUGS.md)
- 改 `ARCHITECTURE.md` / 更新 API / 写新文档 / 标注进度 → [standards/STANDARDS_DOCUMENTATION.md](standards/STANDARDS_DOCUMENTATION.md)
- 提交代码 / 开分支 / 准备合并 / 代码审查 → [standards/STANDARDS_CODING.md](standards/STANDARDS_CODING.md)

---

## 跨守则通用禁令

无论执行哪类任务，以下行为一律禁止：

- ❌ 看到测试失败却置之不理
- ❌ 删除失败的测试用例来"修复"测试
- ❌ 修改测试标准（如把相似度从90%降到50%）
- ❌ 架构已经改了，`ARCHITECTURE.md` 还是旧版
- ❌ 目标架构文档不标注"目标架构/尚未实现"
- ❌ 用"模块化架构 V2 完成"等表述掩盖目标架构与当前代码的不一致
- ❌ 禁止直接推 main
- ❌ 确认 bug 后不修复也不上报

---

## 文档维护禁令

以下行为会导致文档腐败，一律禁止：

- ❌ **删除代码文件前不做全仓库引用扫描**：删除 `.py` / `.sh` 前，必须 `grep -rn "filename" docs/ README.md` 确认无文档引用
- ❌ **参数改了，历史文档不更新**：修改 `LayoutProfile` 或任何硬编码阈值后，必须同步搜索 docs/ 中所有硬编码的该数值
- ❌ **废弃方案只加横幅不清理正文**：废弃文档的正文必须替换为 redirect 或删除操作步骤，不能只加顶部警告
- ❌ **文档迁移不检查内部链接**：移动/复制文档到新目录时，必须检查所有 `.md` 相对路径是否随目录层级变化
- ❌ **状态标注各自为政**：修改 `CODE_OF_CONDUCT.md` 分类表中的文档状态时，必须同时检查该文档正文的第一段状态声明，确保两者一致

---

## 文档变更检查清单

执行以下任一操作后，必须逐项确认：

| 操作 | 检查项 | 命令/方法 |
|------|--------|----------|
| 删除代码文件 | 无文档引用该文件 | `grep -rn "filename" docs/ README.md` |
| 修改 `LayoutProfile` 参数 | 所有文档中的该数值已同步 | `grep -rn "old_value" docs/` |
| 废弃某方案/文档 | 正文已替换为 redirect | 肉眼检查 |
| 移动文档到新目录 | 内部 `.md` 链接已更新 | `grep -rn "\.md" docs/ \| grep -v "docs/"` |
| 改架构实现状态 | 分类表与文档正文一致 | 对比 `CODE_OF_CONDUCT.md` 和该文档第一段 |

---

## 文档分类总表

维护项目文档的当前分类，修改文档前请先确认本文档的分类：

| 文档 | 分类 | 状态 |
|------|------|------|
| `README.md` | 当前实现 | 🚀 已上线 |
| `AI_QUICKSTART.md` | 当前实现 | 🚀 已上线 |
| `PROJECT_STATUS.md` | 当前实现 | 🚀 已上线 |
| `SOLUTIONS.md` | 当前实现 | 🚀 已上线 |
| `LESSONS_LEARNED.md` | 当前实现 | 🚀 已上线 |
| `LOGGING_DESIGN.md` | 当前实现 | 🚀 已上线 |
| `TICK_INVESTIGATION_GUIDE.md` | 当前实现 | 🚀 已上线 |
| `RUNTIME_INVESTIGATION.md` | 当前实现 | 🚀 已上线 |
| `ARCHITECTURE.md` | 目标重构架构 | ⚠️ 部分实现（代码与文档存在差异） |
| `API_SURFACE.md` | 目标重构架构 | ⚠️ 部分实现（代码与文档存在差异） |
| `MODULE_INDEX.md` | 目标重构架构 | ⚠️ 部分实现（代码与文档存在差异） |
| `AUTO_BOT_GUIDE.md` | 废弃方案 | ❌ 已废弃 |
| `GET_DB_KEY.md` | 废弃方案 | ❌ 已废弃 |
| `KEY_EXTRACTION_GUIDE.md` | 废弃方案 | ❌ 已废弃 |
| `MANUAL_KEY_EXTRACTION.md` | 废弃方案 | ❌ 已废弃 |
| `PROJECT_MAP.md` | 历史索引 | ❌ 已废弃 |
| `MAC_RPA_SUMMARY.md` | 调研背景（含废弃方案） | ⚠️ 仅供参考 |

---

## 守则更新记录

| 日期 | 版本 | 更新内容 |
|------|------|----------|
| 2026-04-13 | v1.0 | 初始版本，确立测试错误零容忍政策 |
| 2026-04-15 | v1.1 | 新增文档同步政策、代码提交规范、测试与质量门槛、错误资产化政策；清理已修复的 error_20260413_001 |
| 2026-04-15 | v1.2 | 同步更新 `ARCHITECTURE.md`、`API_SURFACE.md`、`LESSONS_LEARNED.md`：修正去重机制（时间窗口优先）、移除 `ChatSession.should_reply`（统一由 `ReplyPolicy` 负责）、强化 `seen_messages` key 隔离（chat_name+sender+hash）。同步更新 `AI_QUICKSTART.md`、`PROJECT_STATUS.md`、`LOGGING_DESIGN.md`、`V2_FEATURES.md`（已归档，见 `archive/`）、`SOLUTIONS.md`；为 `AUTO_BOT_GUIDE.md` 和 `MAC_RPA_SUMMARY.md` 添加"数据库解密方案已废弃"警告。 |
| 2026-04-15 | v1.3 | 明确区分**当前实现文档**与**目标重构文档**：`ARCHITECTURE.md`/`API_SURFACE.md`/`MODULE_INDEX.md` 加"目标重构架构"提示；修正 `README.md`、`PROJECT_STATUS.md`、`AI_QUICKSTART.md`、`SOLUTIONS.md` 以匹配实际代码结构；移除"模块化架构 V2 完成"虚假声明；补充 V4 到所有版本对比表；为 `GET_DB_KEY.md`、`KEY_EXTRACTION_GUIDE.md`、`MANUAL_KEY_EXTRACTION.md` 添加废弃警告。 |
| 2026-04-15 | v1.4 | **守则模块化拆分**：将 `CODE_OF_CONDUCT.md` 拆分为 `standards/STANDARDS_TESTING.md`、`STANDARDS_DOCUMENTATION.md`、`STANDARDS_CODING.md`、`STANDARDS_BUGS.md`，总纲保留索引和更新记录；后续合并 `STANDARDS_TESTING.md` + `STANDARDS_BUGS.md` 为 `STANDARDS_TESTING_BUGS.md`，按分工读取。 |
| 2026-04-15 | v1.5 | **文档审查体系升级**：创建 `scripts/doc_review.py` 覆盖 lint 盲区；修复 `ARCHITECTURE.md`/`API_SURFACE.md` 中 `HistoryRecord` 缺失、`source_elements` 类型标注、`BotLogger` 返回类型、`on_message` 类型一致性、`TIMESTAMP_PATTERNS` 常量定义等问题；新增 `wechat-rpa-doc-review` Skill 建立迭代闭环审查流程。 |
| 2026-04-15 | v1.6 | **确立测试驱动开发流程（TDD）**：在 `STANDARDS_TESTING_BUGS.md` 新增"测试驱动开发流程"章节，强制要求开发前写测试、开发后全量回归；修复 `ARCHITECTURE.md`/`API_SURFACE.md`/`LOGGING_DESIGN.md` 间 5 处字段/接口/注释不一致。 |
| 2026-04-19 | v1.7 | **文档一致性审查**：修复 V3/V4 幽灵引用、过时参数数值、架构状态自相矛盾、废弃方案正文未清理等问题；新增"文档维护禁令"和"参数变更同步规范"。 |

---

**所有贡献者必须遵守此守则。**

**签名**：_______________________
