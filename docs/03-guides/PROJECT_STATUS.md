# 微信 Mac RPA 项目进度

## 更新时间
2026-05-15

> 本文主体是 2026-05 的历史状态记录，不应作为当前 benchmark 数值来源。当前口径以 README 的“Benchmark 快照”和私有自动报告为准。

## 当前状态
- ✅ 项目架构：双感知管道（SmartPerceptionPipeline 主力 + VisionPipeline 备用）
- ✅ 微信运行（版本 4.1.8）
- ✅ OCR 识别正常
- ✅ LLM 连接正常（OpenClaw/Kimi 用于回复生成，qwen3.6-flash 用于感知 API 兜底）
- ✅ 消息发送正常
- ✅ 登录恢复：支持自动点击登录按钮并恢复主窗口
- ✅ 模块化实现（`src/`）全部完成
- ✅ 真实场景回归测试已建立
- ✅ 智能感知管道已上线（`SmartPerceptionPipeline`：本地预判 + API 兜底，92.6% tick 无需调用 API）
- ✅ Memory 引擎集成完成
- ✅ Tool calling / Skill 匹配机制完成
- ✅ **结构化 Prompt + SessionMemory**（跨 tick 工具缓存，避免重复搜索）
- ✅ **browse_url 工具**（用户分享链接时自动提取正文）
- ✅ **web_search 结果带链接**（支持 browse_url 二次打开）
- ✅ **Hermes 深度分析路径**（skill 匹配时走 Hermes，支持 300 字/5 条回复）
- ✅ **评估体系全面重建**（LLM-as-a-Judge + Rubric，5 个 benchmark 覆盖核心能力）
- ⏳ Tool Decision 对抗性 case 条件反射查 wiki（5/5 失败）
- ⏳ OCR 私聊场景系统性消息错位（15/18 失败）
- ⏳ Memory Search alias_wangzong("王总")召回失败
- ⏳ 昵称识别准确率仍需优化
- ⏳ 多显示器场景支持
- ⏳ GitHub 推送（网络超时，待手动 push）

## 最近修复

### error_20260413_001 - 聊天名称识别错误 ✅ 已修复
- **问题**: 聊天名称 "示例用户甲" 被错误识别为 "®v QS."
- **原因**: 标题栏识别范围 `title_y_max` 太宽泛，包含窗口控制按钮区域；同时 `_is_garbage()` 过滤不足
- **修复**: 
  - 收紧 Y 范围: `title_y_max=95`（覆盖 y=90 的标题，排除 y≥100 的消息区）
  - 添加 X 范围过滤: `title_x_max_ratio=0.95`
  - 增强 `_is_garbage()` 过滤特殊字符和短噪声
- **验证**: 回归测试 `test_regression_title_y_max_extracts_chat_name` 通过

### 2026-05-04 批量更新 ✅ 已上线
- **结构化 Prompt 重构**
  - System prompt 精简为核心人设 + 工具 + 规则
  - User prompt 改为 `[会话]` / `[对方信息]` / `[历史消息]` / `[未读消息]` 结构化格式
  - 新增 `[已缓存数据]` 段落，注入 SessionMemory 工具缓存
- **SessionMemory 跨 tick 缓存**
  - `src/reply/session_memory.py`
  - web_search 5min / stock_query 1min / get_weather 30min / search_memory 10min
- **新增 browse_url 工具**
  - 支持提取网页正文（含微信公众号文章特殊处理）
  - 正文截断到 3000 字
- **web_search 结果带链接**
  - 提取 360 搜索结果的 URL（解码跳转链接）
  - LLM 可用 browse_url 二次打开感兴趣的结果
- **Hermes 调优**
  - 字数 50 → 300 字
  - 回复条数 0-3 → 0-5 条
- **风格调整**
  - 不用"您"，用"你" casual
  - 口头禅：羡慕你们这些有钱人 / 被你装到了 / 等我有钱了...
- **Bug 修复**
  - debug JSON 中 Hermes 字段残留问题（generate() 开头未重置）
  - 标题栏 OCR 失败时盲目切换导致误点单聊框
  - 聊天列表点击位置偏左，改为正中心 + 更长等待时间

### 2026-05-15 评估体系重建 ✅ 已完成
- **P2 评估体系重构**
  - 从关键词匹配升级为 LLM-as-a-Judge + Rubric 评估
  - 使用 deepseek-v4-pro 做 Judge
  - 18 个自定义 Rubric 覆盖 24 个 case
- **Prompt 优化**
  - 规则 8: 承认错误优先于调侃（correction case 通过）
  - 规则 12: 禁止编造具体事实（self_msg_hallucination/unknown_info 通过）
- **删除审计 case**: 移除 7 个 audit_* 历史审计 case，只测当前系统实时表现
- **当时的全量真实 API 重跑**: 曾记录 Reply Quality 24/24、Tool Decision 混合集 22/27；这两项均为历史口径，不作为当前状态
- **新增 OCR Quality Benchmark**: 33 个 case（10 现有 + 23 legacy），覆盖 sender/text/chat_name/chat_list
- **Judge 缓存重建**: Reply Quality 24/24 通过，Tool Decision 对抗性 case Judge 评估已缓存

---

## 评估体系（2026-05-15 重建）

### Benchmark 全景

| Benchmark | Cases | 通过 | 准确率 | 核心指标 |
|-----------|-------|------|--------|----------|
| **Reply Quality** | 24 | **22/24** | — | 2026-05-23 私有历史快照；更早报告为 24/24，待统一版本重跑 |
| **Reply Quality v2** | — | — | — | 回复质量多维度评估（test_reply_quality_benchmark_v2） |
| **Reply Stability** | — | — | — | 回复稳定性一致性（test_reply_stability_benchmark） |
| **Tool Decision** | 27 | 常规 **22/22**；对抗 **0/5** | — | 2026-05-23 私有历史快照，不合并为单一准确率 |
| **Memory Search** | 29 | **28/29** | — | 2026-05-23 私有历史快照 |
| **Chat List Unread** | 23 | — | — | 留存报告冲突，待重跑 |
| **OCR Quality** | 33 | 代表性 **27/29**；回归挑战 **0/4** | — | 私有缓存快照；严格整 case 口径 |
| **Judge Quality（旧版）** | 23 | **15/23** | — | 2026-05-23 历史快照，旧数据集不可完整复现 |
| **Judge Quality v2** | 15 | 缓存 **0/15** | — | 人工 GT 已迁移到私有目录，需当前 Judge 重新评分 |

### 评估模式
- **LLM-as-a-Judge**: Reply Quality + Tool Decision 对抗性 case 使用 deepseek-v4-pro 做 Rubric 评估
- **结构化 Rubric**: 18 个自定义 rubric 覆盖 24 个 case（basic 12 + correction 6 + tool 4 + group 2）
- **缓存策略**: Judge 结果按内容哈希缓存，`judge_{hash}.json`

### 当时记录的问题（2026-05，非当前状态）

#### 1. Tool Decision - 对抗性 case 条件反射（🔴 P0）
- **症状**: 5/5 对抗性 case 因"看到人名/关键词就查 wiki"失败
- **失败 case**: `adversarial_link`, `adversarial_websearch`, `adversarial_stock`, `adversarial_statement`, `adversarial_vague`
- **根因**: Bot 对 search_memory 的条件反射调用，缺乏对查询意图的判断
- **Judge 诊断**: "未过度调用 search_memory"维度全部失败

#### 2. OCR Quality - 私聊场景系统性错位（🔴 P0）
- **症状**: private_chat + legacy_private 共 18 个 case，15 个失败
- **表现**: API 返回消息数 > 预期数（6→7 或 5→7），sender/text 整体偏移
- **根因**: qwen3.6-flash 在私聊截图中把输入框/时间戳/UI 元素误判为消息
- **影响**: 消息序列错位 → sender 污染 → 消息被过滤为 0 条 → Bot 漏消息
- **典型**: `error_20260414_006~016` (11 个 case 全部 6→7)

#### 3. Memory Search - alias_wangzong 召回失败（🟡 P1）
- **症状**: 查询"王总"召回失败（P=0% R=0% F1=0%）
- **可能原因**: wiki 中没有"王总"别名映射

---

## 技术方案

### 当前方案：Vision OCR（推荐）
```
消息接收: Vision OCR 识别微信界面截图
消息发送: AppleScript + System Events  
大模型: Kimi
```

**优点**:
- 无需关闭 SIP
- 无需获取 db_key
- 不依赖微信数据库
- 更安全稳定

---

## 机器人版本

### 新架构模块化版（当前唯一版本）
```bash
cd ~/wechat-mac-rpa
python3 run_bot.py
```
- L1-L5 模块化架构（`src/`）
- 双感知管道：SmartPerceptionPipeline（主力，本地预判 + qwen3.6-flash API 兜底） + VisionPipeline（纯本地 OCR 备用回退）
- 环境变量 `USE_MULTIMODAL_OCR=false` 可切换回纯本地模式
- 支持自动登录恢复（`WeChatLoginHandler`）
- 真实场景回归测试覆盖

### Accessibility API 版（已删除）
```bash
# 此版本已删除，功能已合并到模块化架构
```
- 需要辅助功能权限
- 更精确的界面控制

### 历史版本（已删除）
- `core/auto_bot_vision_ocr_v2.py` - 已删除
- `core/auto_bot_vision_ocr_v3.py` - 已删除
- `core/auto_bot_vision_ocr_v4.py` - 已删除（由新架构完全替代）

---

## 关键文件

### 可直接运行版本
| 文件 | 说明 |
|------|------|
| `src/bot/wechat_bot.py` | ⭐ 模块化架构机器人（当前唯一版本） |
| `run_bot.py` | 一键启动脚本（双管道自动选择） |
| `scripts/generate_ocr_benchmark_report.py` | OCR benchmark 报告生成 |

### 模块化架构（按 `ARCHITECTURE.md` 拆分）
| 文件 | 说明 |
|------|------|
| `src/perception/smart_pipeline.py` | ⭐ L3.5 智能感知管道（主力：本地预判 + qwen3.6-flash API 兜底） |
| `src/perception/vision_pipeline.py` | L3.5 纯本地 OCR 管道（备用回退） |
| `src/session/global_store.py` | L4 会话与去重 |
| `src/reply/policy.py` | L4 回复决策 |
| `src/reply/generator.py` | L4 回复生成（支持双模型：OpenClaw/Kimi + Hermes） |
| `src/action/message_sender.py` | L4 消息发送 |
| `src/bot/wechat_bot.py` | ⭐ L5 主循环编排 |
| `src/logging/bot_logger.py` | 运行时日志 |
| `src/memory/engine.py` | Memory 引擎 |
| `src/tools/` | Tool Registry & Built-in Tools |

---

## 历史截图存档
```
~/wechat-mac-rpa/data/screenshots/   # 当前项目统一路径
/tmp/wechat_screenshots/             # V2 历史兼容路径
├── wechat_20250411_204538_123.png
├── wechat_20250411_204541_456.png
└── ...
```

---

## 注意事项

1. **微信窗口需要可见** - OCR 需要截图
2. **授予辅助功能权限** - 系统设置 → 隐私与安全 → 辅助功能
3. **避免高频发送** - 建议间隔 3-5 秒

---

## 废弃方案

### 数据库解密方案（不再使用）
```
原方案: 解密微信 SQLite 数据库读取消息
状态: 已废弃
原因: 需要关闭 SIP + 获取 db_key，过于复杂
```

---

**状态：✅ OCR 方案运行正常，无需 db_key**
