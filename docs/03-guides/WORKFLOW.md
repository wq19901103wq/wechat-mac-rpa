# 开发者工作流手册

本文档固化日常开发、测试、实验的标准流程。所有操作必须按流程执行，禁止跳过验证环节。

---

## 1. 目录整洁纪律

**运行时生成的文件严禁放在根目录。**

### 根目录白名单

以下文件/目录允许存在于根目录，其他一律清理：

| 类型 | 允许的文件/目录 | 说明 |
|------|----------------|------|
| 入口 | `run_bot.py` | 生产环境入口 |
| 配置 | `.env`、`.env.example`、`.gitignore` | 环境配置 |
| 文档 | `README.md`、`AGENTS.md` | 项目说明 |
| 代码 | `src/`、`scripts/`、`skills/` | 源代码 |
| 测试 | `tests_integration/` | 集成测试与回归测试 |
| 文档 | `docs/` | 文档体系 |
| 数据 | `data/` | 运行时数据（已被 gitignore） |
| 归档 | `archive/` | 旧文档归档 |
| 模型 | `models/` | 本地模型（如 embedding） |
| 第三方 | `third_party/` | 外部依赖 |

### 运行时文件归属

| 文件类型 | 生成来源 | 正确位置 |
|---------|---------|---------|
| Benchmark 报告（HTML） | `scripts/generate_benchmark_dashboard.py` | `data/reports/` |
| 趋势报告（HTML） | `scripts/monitor_benchmark.py` | `data/reports/` |
| 运行日志 | `bot_logger` | `data/logs/` |
| Tick 调试 JSON | `debug_logger` | `data/debug/` |
| 截图 | `window_capture` | `data/screenshots/` |
| 后台运行时输出 | `admin.py` | `*.out`（已被 gitignore） |

### 清理检查清单

每次提交前检查：
- [ ] 根目录无 `.html` 文件
- [ ] 根目录无 `.out` 文件
- [ ] 根目录无旧版脚本（如 `generate_benchmark_report.py`）
- [ ] 根目录无空目录（如 `app/`）

---

## 2. Bug 修复工作流

```
发现 Badcase
    │
    ▼
1. 复现确认
    - 查 tick_log（chat_name + tick_id）
    - 查 debug JSON（data/debug/）
    - 查截图（data/screenshots/）
    - 确认是感知层 / 推理层 / 行动层问题
    │
    ▼
2. Benchmark 复现
    - 将 badcase 写成 benchmark case
    - 跑对应 benchmark 验证必现
    │
    ▼
3. 根因定位
    - 用 MODULE_INDEX.md 定位文件
    - 分析是通用问题还是特例
    │
    ▼
4. 通用规则修复
    - 严禁 case-by-case 修 prompt
    - 必须从根因出发写通用规则
    │
    ▼
5. 小样本验证（1-2 个 case）
    - 确认修复有效
    - 确认无回归
    │
    ▼
6. 全量 benchmark 回归
    - python3 -m pytest src/tests/test_xxx_benchmark.py -v
    - 全部通过才能继续
    │
    ▼
7. 提交
    - git commit（附 benchmark 结果截图）
    - 同步更新相关文档
    │
    ▼
8. 如有新教训 → 更新 AGENTS.md 历史教训
```

---

## 3. 测试排查工作流

线上异常时的排查路径：

```
收到异常反馈
    │
    ├── 路径 A：Bot 未回复（漏回复）
    │   ├── 查 tick_log：should_reply=0 → 看 skip_reason
    │   ├── 查截图：感知层是否识别到消息
    │   ├── 查 debug JSON：LayoutParser 输出是否正常
    │   └── 定位：感知层 / Policy / Generator
    │
    ├── 路径 B：Bot 回复质量差
    │   ├── 查 tick_log：看 judge_score、judge_reason
    │   ├── 查 case_db：cases 表看完整对话和 prompt
    │   ├── 查工具调用：tool_calls_json 是否正确
    │   └── 定位：Prompt / 工具调用 / 记忆召回
    │
    ├── 路径 C：Bot 重复回复
    │   ├── 查 tick_log：new_messages_count 是否合理
    │   ├── 查 debug JSON：LCS 对齐结果
    │   └── 定位：global_store.merge_tick
    │
    └── 路径 D：Bot 发错聊天
        ├── 查 tick_log：chat_name 是否匹配
        ├── 查截图：当前聊天名是否识别正确
        └── 定位：chat_list_clicker / LayoutParser
```

---

## 4. AB 实验工作流

```
提出假设（如：新 prompt 能降低幻觉率）
    │
    ▼
1. 实验设计
    - 确定变量（prompt / 模型 / 路由策略）
    - 确定指标（badcase_rate / avg_score / 特定维度）
    - 确定样本量（建议 ≥ 10 条 tick，Judge 调用慢，样本量过大实验时间过长）
    │
    ▼
2. 样本选择
    - **不过滤空回复 tick**：空回复本身就是 badcase，保留以暴露配置问题
    - 排除明显系统错误的 tick（如 raw_response 为 `[空回复，attempt=3]` 且非模型正常输出）
    - 等距采样或随机采样，确保覆盖不同聊天场景
    │
    ▼
3. 小样本验证（强制步骤，禁止跳过）
    - python3 scripts/run_experiment.py --exp <name> --n-samples 1
    - 检查 Judge 是否成功：查看输出是否正常评分（非"空返回"/"JSON 解析失败"）
    - 结果异常（如 0 分、空返回、或明显不合理评分）→ 先排查 Judge 稳定性，**严禁直接全量**
    - 失败原因排查：检查 prompt 长度、API 端点、模型选择
    │
    ▼
4. 全量运行
    - 基线采集：python3 scripts/run_experiment.py --exp <name> --n-samples <N>
      - 对照组 = CONTROL（当前生产配置），重新生成回复
    - 实验组运行：同上脚本同时完成实验组
    - Judge 失败时自动重试 3 次（judge_worker.py 内置）
    │
    ▼
5. Judge 评估与无效排除
    - 两组结果统一过 JudgeWorker
    - **排除无效评分**：Judge reason 为"空返回"或"JSON 解析失败"的 tick 不计入统计
    - 对比 badcase_rate、avg_score、各维度评分（仅有效样本）
    - 输出"有效 X/Y 个"，确保样本量足够
    │
    ▼
6. 结果入库
    - 实验结果自动写入 experiments 表
    - 维度差异写入 dimension_diffs_json
    - summary 中标注有效样本数（如"badcase +1 均分 +2.3 (有效 8/10)"）
    │
    ▼
7. Dashboard 查看
    - 打开 admin.py 的实验管理页
    - 可视化对比两组结果
    - 检查 diff 表头是否正确（"线上配置(CONTROL)" vs "实验组"）
    │
    ▼
8. 决策
    ├── 显著改善 → 合并上生产 → 跑全量 benchmark 回归 → 提交
    └── 无显著差异 / 恶化 → 废弃实验 → 记录结论
```

### 实验注意事项

- **Judge 稳定性**：deepseek-v4-pro 处理长 prompt 时偶发空返回，已通过去掉 `response_format` + 重试 3 次缓解，但仍可能失败
- **对照组设计**：CONTROL（生产配置）重新生成，不是直接使用历史回复，引入少量随机性
- **无效样本处理**：实验脚本已自动排除 Judge 失败的 tick，若有效样本 < 50% 则实验结果不可信
- **后台运行**：实验耗时较长（10 tick ≈ 15-20 分钟），建议用 nohup 后台运行

---

## 5. 文档书写规范

### 何时必须写文档

| 场景 | 必须写的文档 | 位置 |
|------|------------|------|
| 新增/删除模块 | MODULE_INDEX.md 更新 | `docs/02-architecture/` |
| 修改公共接口 | API_SURFACE.md 更新 | `docs/02-architecture/` |
| 修改 L1-L5 依赖关系 | ARCHITECTURE.md 更新 | `docs/02-architecture/` |
| 新增 benchmark | PROJECT_STATUS.md 更新 | `docs/03-guides/` |
| 修 bug 有新教训 | AGENTS.md 历史教训 | 根目录 |
| 做 AB 实验 | 实验记录（假设 / 配置 / 结果 / 结论） | `data/experiments/` |
| 前端页面开发 | Playwright 自测截图 | `/tmp/` 留证 |
| 目录结构调整 | 本文件（WORKFLOW.md）更新 | `docs/03-guides/` |

### 文档同步义务

修改代码后，**必须在同一 commit 或紧随其后的 commit 中更新对应文档**。禁止"先改代码，文档以后补"。

---

## 6. 前端修改后强制验证

**任何修改前端相关代码（admin.py、HTML 模板、CSS、JS、路径配置）后，必须全量验证所有 admin 页面，禁止只看代码就声称完成。**

### 验证范围

必须检查 admin.py 的全部 8 个页面：
- `/` Dashboard
- `/ticks` Tick 查看
- `/gt` GT 标注
- `/review` 人工审核
- `/screenshots` 截图 OCR
- `/benchmark/judge` Judge 质量
- `/benchmark/reply` 回复质量
- `/experiments` 实验管理

### 验证清单

1. **所有页面 HTTP 200**，不是空白页/500 错误
2. **Admin 侧边栏是否保留**（不能把独立 HTML 直接返回，必须嵌入 `_page()` 框架）
3. **页面不出现原始 JSON/代码**（检查 `.inner_text()` 不包含 `[{`、`is_badcase` 等字段名）
4. 所有链接、按钮是否可点击且有响应
5. 数据是否正确展示（不是空白/报错），关键指标数字 > 0
6. 分页、筛选、跳转等交互是否生效
7. 图片/截图是否能正常加载
8. **表格、卡片、指标区布局是否整齐**（`page.locator('.card').count()` > 0）

### Playwright 验证脚本

```python
import asyncio, time
from playwright.async_api import async_playwright

async def verify_admin():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        for path in ["/", "/ticks", "/gt", "/review", "/screenshots", "/benchmark/judge", "/benchmark/reply", "/experiments"]:
            await page.goto(f"http://localhost:8766{path}")
            assert page.url.endswith(path), f"跳转失败: {path}"
            text = await page.inner_text("body")
            assert "[{" not in text and "is_badcase" not in text, f"暴露原始数据: {path}"
            count = await page.locator(".card").count()
            assert count > 0 or path in ["/gt", "/review"], f"布局异常: {path}"
            await page.screenshot(path=f"/tmp/admin_{path.replace('/', '_')}_{int(time.time())}.png", full_page=True)
        await browser.close()
        print("✅ 全部 8 个页面验证通过")

asyncio.run(verify_admin())
```

### 纪律

- 发现问题立即修复，修复后重新**全量验证**（不能只验证改过的页面）
- **禁止只看代码不实际浏览就声称"完成"**
- **禁止返回独立 HTML 导致侧边栏消失** — 所有页面必须通过 `_page()` 嵌入 admin 框架
- **每次前端修改后必须用 Playwright 截图留证**，保存到 `/tmp/` 下

---

## 7. 提交规范

### Commit Message 格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

| Type | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | 修复 bug |
| `docs` | 文档更新 |
| `refactor` | 重构（不改变行为） |
| `test` | 测试相关 |
| `chore` | 构建/工具/目录整理 |

### 提交前检查清单

- [ ] 代码已自测通过
- [ ] Benchmark 回归通过（如涉及 prompt/模型/感知层变更）
- [ ] 文档已同步更新
- [ ] 根目录无运行时文件
- [ ] `.gitignore` 已更新（如新增运行时文件类型）
