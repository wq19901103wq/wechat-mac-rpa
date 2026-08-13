# 回归测试指南

> 针对模块化重构后的 RPA 系统的分层回归测试框架。

---

## 快速开始

```bash
# 运行完整回归测试套件
python3 tests/regression_suite.py

# 只跑某一层
python3 tests/regression_suite.py --layer session
python3 tests/regression_suite.py --layer layout
python3 tests/regression_suite.py --layer errors
python3 tests/regression_suite.py --layer docs
```

---

## 测试分层

### L1: Session / Deduplication

**目标**: 验证 `ChatSession` 的去重、回声检测、冷却期逻辑。

**覆盖场景**:
- 首次消息 → 应被识别为 `new_messages`
- 重复消息 → 应被 `filter_new()` 过滤
- 回声消息 → 自己刚发的消息在 10 秒窗口内出现 → 应被过滤
- 上下文去重 → 上一条也匹配历史时 → 应被过滤（滚动场景）

**状态**: 需要 `src/session/chat_session.py` 按目标架构实现后才会真正执行。

---

### L2: Layout / Message Extraction

**目标**: 验证 `LayoutParser` 和 `MessageExtractor` 对 fixtures 的解析能力。

**覆盖场景**:
- `small_scene.png` — 简单聊天窗口
- `medium_scene.png` — 中等复杂度
- `large_scene.png` — 复杂聊天窗口

**当前限制**: 需要 OCR 引擎接入后才能做完整的端到端断言。目前先验证模块可实例化。

---

### L3: 历史错误案例回归

**目标**: 确保已修复的错误不再复发。

**数据来源**: `tests/fixtures/errors/error_YYYYMMDD_NNN.json`

**使用方式**:
1. 发现新错误 → 用 `auto_add_error_case.py` 记录
2. 修复后 → 将 `status` 改为 `fixed`
3. 回归测试自动验证 `fixed` 案例是否仍通过

---

### L4: 文档一致性

**目标**: 确保目标架构文档（`ARCHITECTURE.md`、`API_SURFACE.md`）保持自洽。

**自动检查**:
- `scripts/doc_lint.py`
- `scripts/doc_review.py`

---

## Fixture 场景扩展清单

当前 fixtures 主要覆盖 OCR 识别场景。建议逐步增加以下专用场景：

| 场景名称 | 描述 | 测试重点 |
|---------|------|---------|
| `scroll_repeat` | 聊天滚动后同一批消息出现在新位置 | 窗口指纹去重 |
| `echo_immediate` | 自己刚发消息，对方立即回复 | 回声检测 + 新消息识别 |
| `multi_chat_list` | 左侧聊天列表有多个未读 | `ChatListItem` 提取 |
| `timestamp_dense` | 消息中间穿插多个时间戳 | `TIMESTAMP_PATTERNS` 过滤 |
| `draft_input` | 输入框有草稿内容 | 输入框 vs 消息区分 |
| `group_at_me` | 群聊中 @ 我的消息 | `is_at_me` 识别 |
| `self_other_mixed` | 自己和对方消息交替 | `SenderType` 判定 + 气泡检测 |

**添加方式**: 截图后保存到 `tests/fixtures/`，并编写同名 `.json` 描述预期结果。

---

## 与 `run_tests.sh` 的关系

- `run_tests.sh`: ⚠️ 已失效（依赖已删除的 V4 代码）
- `regression_suite.py`: 针对**模块化目标架构**的分层回归，随着 `src/` 目录的实现逐步生效

**建议**:
- 修改 `docs/02-architecture/ARCHITECTURE.md` / `docs/02-architecture/API_SURFACE.md` → 跑 `regression_suite.py --layer docs`
- 修改 `src/session/` → 跑 `regression_suite.py --layer session`
- 修改 `src/layout/` 或 `src/message/` → 跑 `regression_suite.py --layer layout`
- 发现/修复错误 → 跑 `regression_suite.py --layer errors`

---

## 最佳实践

1. **每次改 Session 逻辑前**，先确认 `regression_suite.py --layer session` 当前状态
2. **每次改布局/解析逻辑后**，检查 fixtures 是否全部通过
3. **新增 fixture 场景时**，同步更新本指南的 Fixture 清单
4. **错误修复后**，立即将对应 error case 的 `status` 改为 `fixed`
