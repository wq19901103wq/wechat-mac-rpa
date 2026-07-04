# 前端开发测试流程

> 目标：新功能开发完成后，必须通过浏览器端自动化验证才能提交，杜绝"写完了才发现跑不通"。

---

## 分层测试策略

| 层级 | 工具 | 覆盖范围 | 编写时机 |
|------|------|----------|----------|
| 前端 E2E | Playwright | 页面加载、DOM 渲染、用户交互、表单提交、视觉反馈 | **与新功能同步编写** |
| 后端 API | pytest + TestClient | 路由参数、DB 读写、状态码、返回格式 | **与新功能同步编写** |

**原则**：Playwright 只测前端，后端用 TestClient，不混用。

---

## 开发流程（Step by Step）

```
Step 1: 需求确认
    ↓
Step 2: 写测试（先写 or 并行写）
    ├─ tests/e2e/test_admin_<feature>.py   ← Playwright
    └─ tests/test_api_<feature>.py          ← TestClient
    ↓
Step 3: 开发功能代码
    ↓
Step 4: 本地跑全部测试
    pytest tests/e2e/ tests/test_api_*.py
    ↓
Step 5: 全部通过 → 提交
    有失败 → 修 → 回到 Step 4
```

---

## Playwright 测试必须覆盖什么

每个新页面/新交互至少验证：

1. **页面可加载** — `page.goto()` 不 500，关键文案存在
2. **交互元素存在** — 按钮、下拉框、输入框渲染正确
3. **用户操作链路** — 点击/选择/输入 → 提交 → 反馈
4. **状态持久化** — 刷新页面后数据是否保留
5. **视觉反馈** — 成功/失败的样式变化（如边框变绿、文案变化）

---

## 老功能 Regression 防护

- 已有测试脚本（如 `test_admin_code_audit.py`）每次提交前必须跑通
- 新功能不得破坏老功能的 DOM 结构（如 id/class 变更需同步改测试）

---

## 快速运行命令

```bash
# 后端 API 测试
pytest tests/test_api_*.py -v

# 前端 E2E 测试
pytest tests/e2e/ -v --headed   # 有头模式看执行过程
pytest tests/e2e/ -v            # 无头模式 CI 用
```

---

## 历史教训

| 时间 | 问题 | 根因 | 后果 |
|------|------|------|------|
| 2026-05-29 | admin 保存按钮无响应 | checkbox 改 select 后 JS 选择器未同步 | 连续 3 次修复 |
| 2026-05-29 | GitHub URL 错误 | 硬编码组织名写错 | 7 条发现链接失效 |
| 2026-05-29 | 保存变量冲突 | `status` 同时指向 DOM 和字符串 | 请求体传了 DOM 对象 |

**共同点**：均未在浏览器端验证就提交。**本流程旨在杜绝此类问题。**
