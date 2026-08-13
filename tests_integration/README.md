# 微信 OCR 测试套件

## 测试框架

当前测试基于 `pytest`，覆盖 `src/` 模块化架构和 `tests_integration/` 外部测试套件。

## 运行测试

### 运行所有内部测试
```bash
python3 -m pytest src/tests/ -v
```

### 运行 OCR 质量 Benchmark（推荐）
```bash
# 使用私有缓存（不调用 API）
python3 scripts/run_private_benchmarks.py

# 显式刷新 OCR API 缓存
python3 scripts/run_private_benchmarks.py --refresh ocr
```

### 生成可视化报告
```bash
python3 scripts/generate_ocr_benchmark_report.py
# 输出: data/reports/ocr_benchmark_report.html
```

### 运行集成测试
```bash
python3 tests_integration/test_integration.py
```

## 测试用例结构

每个测试用例包含两个文件：
- `{name}.png` - 微信截图
- `{name}.json` - 预期 OCR 结果（Ground Truth）

真实 Fixture 存放在 Git 忽略的 `data/private_benchmarks/ocr/fixtures/`，不会进入公开仓库。

## 当前测试用例

| 测试套件 | 位置 | 数量 | 说明 |
|---------|------|------|------|
| 内部单元测试 | `src/tests/` | 148+ | 模块化架构各层单元测试 |
| OCR 质量 Benchmark | `src/benchmarks/ocr_quality.py` | 33 | 私有真实截图与缓存评估 |
| 真实场景回归 | `tests_integration/test_real_scene_extraction.py` | - | 基于真实截图的回归验证 |

## 当前私有缓存快照

| 指标 | 数值 |
|------|------|
| **代表性场景严格通过** | **27/29** |
| **已知回归挑战恢复** | **0/4** |
| Chat Name 准确率 | 93.9% |
| Message Count 准确率 | 93.9% |
| Sender 平均准确率 | 89.1% |
| Text 平均准确率 | 90.9% |

严格整 case 通过要求聊天名、消息数、全部 sender 均正确，且 text 正确率至少 80%；它不等于文字 OCR 准确率。代表性场景与已知回归挑战不合并展示。

## 添加回归测试

发现新的识别错误时：
1. 保存错误截图到 `data/private_benchmarks/ocr/fixtures/`
2. 编写同名 `.json` 描述预期结果
3. 运行 `python3 scripts/run_private_benchmarks.py --refresh ocr` 生成缓存
4. 重新生成报告验证

## 测试标准

- 聊天名称准确率 >= 90%
- 发送者类型识别率 >= 85%
- 消息数量准确率 >= 90%
- 消息内容准确率 >= 80%

## 文件位置

```
src/benchmarks/ocr_quality.py                 # 指标与运行逻辑
scripts/run_private_benchmarks.py             # 私有统一入口
data/private_benchmarks/ocr/fixtures/         # Git 忽略的真实截图与 GT
data/private_benchmarks/reports/              # Git 忽略的自动报告
```

---

**历史说明**: 旧版 `test_ocr_v4.py`、`add_test_case.py` 及 `core/auto_bot_vision_ocr_v4.py` 已删除，由 `src/` 模块化架构 + pytest 完全替代。
