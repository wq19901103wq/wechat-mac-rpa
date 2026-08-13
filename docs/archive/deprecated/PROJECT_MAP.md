# 微信 Mac RPA 项目地图

> 项目: wechat-mac-rpa  
> 核心功能: 基于 Vision OCR 的微信自动化  
> 更新日期: 2026-04-12

---

## 🗺️ 项目结构

```
wechat-mac-rpa/
├── core/                          # 核心 OCR 机器人
│   ├── auto_bot_vision_ocr_v4.py  ⭐ 主程序 - V4 完整版
│   ├── auto_bot_vision_ocr_v3.py  # V3 颜色气泡版
│   ├── auto_bot_vision_ocr_v2.py  # V2 增强版
│   ├── auto_bot_accessibility.py  # Accessibility API 版
│   └── debug_ocr_v4.py            # 调试工具
│
├── tests/                         # 测试框架
│   ├── test_ocr_v4.py             # 单元测试框架
│   ├── run_tests.sh               # 测试运行脚本
│   ├── add_test_case.py           # 添加测试用例工具
│   ├── capture_and_add_test.sh    # 截图+添加工具
│   ├── README.md                  # 测试文档
│   └── fixtures/                  # 测试用例目录
│       ├── current.png/json       # 群聊基础场景
│       ├── private_w1han.png/json # 私聊长文本+@消息
│       ├── large_scene.png/json   # 大窗口群聊
│       ├── medium_scene.png/json  # 中等窗口
│       └── small_scene.png/json   # 登录弹窗
│
├── utils/                         # 工具模块
│   ├── llm_client.py              # Kimi LLM 客户端
│   └── accessibility.py           # Accessibility 工具
│
├── data/                          # 数据存储
│   ├── screenshots/               # OCR 截图存档
│   └── logs/                      # 历史消息日志
│
├── config/                        # 配置
│   └── config.yaml                # 配置文件
│
├── examples/                      # 示例代码
│   ├── simple_mac_bot.py
│   └── kimi_llm_bot.py
│
├── run_simple.py                  # 简化版启动
├── run_auto_accessibility.sh      # Accessibility 版启动
├── capture_and_add_test.sh        # 截图测试工具
├── PROJECT_MAP.md                 # 本文件
└── README.md                      # 项目说明
```

---

## 🎯 核心模块说明

### src/bot/wechat_bot.py (主程序)
**功能**: L1-L5 模块化架构编排

**核心类**:
- `WeChatBot` - 机器人主类，协调感知→决策→执行
- `VisionPipeline` - 视觉感知管道（Capture → OCR → Layout → Extract）
- `WeChatLoginHandler` - 登录恢复（检测登录按钮并自动点击）

**关键方法**:
- `tick()` - 主循环：感知 → 生成回复 → 发送
- `perceive()` - `VisionPipeline.perceive()` 封装
- `generate_reply()` - 调用 LLM 生成回复
- `send_message()` - AppleScript 发送消息

**识别逻辑**:
1. `WindowCapture` 截取微信窗口（自动处理登录弹窗）
2. `VisionOCREngine` OCR 识别所有文本
3. `LayoutParser` 按 UI 区域分组（标题/列表/消息区/输入区）
4. `MessageExtractor` 提取结构化消息（SELF/OTHER）
5. `ChatSession` 去重与回声过滤
6. LLM 生成回复 → `MessageSender` 发送

---

## 🧪 测试框架

### 测试标准 (严格)
- ✅ 消息数量: 必须完全一致
- ✅ 内容相似度: > 90%
- ✅ 发送者 ID: 必须正确识别

### 当前测试用例

| 用例 | 场景 | 测试内容 | 所属架构 |
|------|------|---------|----------|
| real_login_recovered_scene | 群聊 | 登录恢复后消息提取准确性 | 新架构 |
| large_scene | 群聊 | LayoutParser 可实例化 | 新架构 |
| medium_scene | 群聊 | LayoutParser 可实例化 | 新架构 |
| small_scene | 弹窗 | LayoutParser 可实例化 | 新架构 |

**重点变化**:
- 旧 V4 fixture (`current`, `private_w1han`) 已归档到 `tests/fixtures/legacy/`
- 新增 `test_real_scene_extraction.py` 对真实截图做端到端准确性断言

### 添加测试用例

```bash
# 方式1: 自动截图并添加
./capture_and_add_test.sh test_name

# 方式2: 从现有图片添加
python3 tests/add_test_case.py /path/to/image.png --name test_name

# 方式3: 识别错误时自动添加（见下方机制）
```

---

## 🔧 自动添加错误案例机制

当 OCR 识别出现错误时，自动保存为测试用例以便后续修复。

### 使用方式

在代码中捕获识别错误时调用:

```python
from tests.auto_add_error_case import auto_add_error_case

# 当检测到识别错误时
if recognition_error:
    auto_add_error_case(
        image_path="/tmp/wechat_ocr.png",
        expected_result={...},  # 正确的预期结果
        error_type="WRONG_SENDER",  # 错误类型
        error_details="发送者识别错误"
    )
```

### 错误案例存储

错误案例保存在 `tests/fixtures/errors/` 目录:
```
tests/fixtures/errors/
├── error_20260412_001.png/json   # 错误截图+预期结果
├── error_20260412_002.png/json
└── ...
```

### 修复流程

1. 错误自动保存到 `errors/` 目录
2. 运行测试时会包含错误案例
3. 修复代码后，错误案例通过测试
4. 手动移动到 `fixtures/` 成为正式测试用例

---

## 📊 数据存储

### 截图存档
```
data/screenshots/
└── wechat_YYYYMMDD_HHMMSS_XXX.png
```

### 历史消息
```
data/logs/
├── message_history.json   # 结构化历史（所有消息）
└── chat_history.txt       # 文本日志（便于查看）
```

---

## 🚀 快速命令

```bash
# 运行新架构集成测试
python3 tests/test_integration.py

# 运行真实场景回归测试
python3 tests/test_real_scene_extraction.py

# 运行完整回归套件
python3 tests/regression_suite.py

# 运行 pytest（新架构测试）
pytest tests/test_real_scene_extraction.py tests/test_integration.py tests/test_wechat_not_ready.py -q
```

---

## 📝 关键配置

### 布局常量 (新架构)
配置位于 `src/layout/profile.py`
```python
PROFILE_WECHAT_MAC_1760X1280 = LayoutProfile(
    window_width=1760,
    window_height=1280,
    left_boundary=480,           # 左右区域分界线
    chat_list_x_max=360,
    title_y_max=50,              # 顶部标题栏高度
    title_x_max_ratio=0.95,      # 标题栏覆盖范围
    input_y_min=1160,            # 底部输入区起始
    self_green=(176, 240, 167),  # 自己消息气泡颜色
    self_green_tolerance=35,     # 颜色容差
    nickname_x_min_ratio=0.30,   # 昵称区域左边界
    nickname_x_max_ratio=0.55,   # 昵称区域右边界
    message_cluster_threshold=80,
)
```

### 测试标准
```python
SIMILARITY_THRESHOLD = 0.9   # 内容相似度阈值 90%
ALLOW_COUNT_MISMATCH = False # 不允许消息数量误差
```

---

## 🔄 更新记录

| 日期 | 更新内容 |
|------|---------|
| 2026-04-11 | 项目初始化，OCR V1-V3 |
| 2026-04-12 | V4 完整版发布，昵称识别 |
| 2026-04-12 | 测试框架建立，5个测试用例 |
| 2026-04-12 | 自动添加错误案例机制 |
| 2026-04-16 | 新架构 L1-L5 模块化重构完成，旧 V2/V3/V4 删除 |
| 2026-04-17 | 新增真实场景准确性回归测试，修复时间戳/噪声误识别 |

---

## 🎯 TODO

- [x] 增加多发言者群聊测试用例
- [ ] 支持图片内容识别（多模态）
- [ ] 优化昵称识别准确率
- [ ] 支持多显示器场景
- [ ] 建立错误案例自动收集机制

---

## 📋 项目守则

### 测试错误必须立即修复

**核心原则**：所有测试错误必须在 24 小时内修复，不得积累。

**流程**：
1. 发现识别错误 → 保存真实截图到 `tests/fixtures/`
2. 运行测试确认错误 → `python3 tests/test_real_scene_extraction.py`
3. 分析错误原因 → 查看 `PerceptionResult` 各层输出
4. 修复代码 → 调整 `LayoutParser` / `MessageExtractor` / `LayoutProfile`
5. 验证修复 → 真实场景测试通过
6. 提升为正式用例 → 更新 `test_real_scene_extraction.py` 断言

**禁止事项**：
- ❌ 忽视测试失败
- ❌ 删除失败的测试用例
- ❌ 修改测试标准降低要求

**当前状态**：
- ✅ 新架构真实场景回归测试已建立
- ✅ 时间戳/噪声误识别已修复
- ⚠️ 旧 V4 错误案例已归档至 `tests/fixtures/legacy/errors/`

---
