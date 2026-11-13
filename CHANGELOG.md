# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- pytest 运行不再污染生产日志：测试会话经 `conftest.py` 将 `WECHAT_BOT_LOGS_DIR` 指向临时目录，`BotLogger` 默认目录优先读该环境变量，未设置时行为不变
- DH 审查拦截路径引用不存在的 `unreplied` 变量（NameError 落外层异常，拦截被误记为生成失败且无限误拦计数失效）：改用本方法已有的 `messages`
- wechat_reader 截图成功路径把 `scale_factor` 传给未定义的 `rect`（NameError 会导致重启后感知链路整体失败）：改用 `window_rect`（wechat_reader 已提交 `72d4e9f` 并推送，pin 已更新）
- 发送在“粘贴校验通过”与“回车”之间二次确认前台仍是微信，焦点被切走时放弃本次发送并重试，不再把消息发到别的应用
- `pending_self_messages` 的 tick/admin 双线程 append 加锁保护
- ReAct 重试时重置消息流（不带入上一轮的 assistant/tool 残留），重试睡眠不超过剩余 deadline，已执行工具后重试会告警（有副作用工具可能重复执行）
- DH review 操作使用独立短超时（300s，可配置常量 `REVIEW_TIMEOUT_SECONDS`），不再与 draft 共用最长 600s 超时
- `generate()` 持锁串行化，并发流水线不再互相覆盖模型客户端/超时与 `last_*` 观测状态
- 同一消息被审查连续拦截 3 次升级为 error 日志（无限误拦的人工出口）
- DH worker 协议流层面的外来错误（id=None）不再销毁整个 runtime，继续等待本请求响应
- 多显示器场景取窗口所在屏幕的缩放因子；Layout 失败回退的聊天列表坐标改用窗口真实尺寸（wechat_reader 已提交 `1ac17f0` 并推送，pin 已更新）
- 共享仓 wechat-reader 已推送 GitHub（`b716b58`）并更新 requirements pin；此前提交：感知 API 失败不丢消息（`ba0a1c6`）、layout 时间戳正则与数字开头昵称（`b716b58`）
- DH 审查 quote 比对做空白归一化：仅空白差异不再判无效依据；真实不符维持 fail-closed（整条拒绝）
- `session_memory` 同名会话类型不一致时保留原类型并告警，不再静默改写
- DH worker 输入超限时消费到换行符重新对齐协议流；draft 失败取消 ContextPrefetch 并清空引用，不再占住单线程池
- `layout_parser` 两处逻辑错误（wechat_reader 已提交）：聊天列表时间戳判定改为 `re.match`；昵称清洗不再剥数字开头
- `on_message` 回调异常不再中断整 tick 回复流程
- persona few-shot 排序 tie-breaking 统一 `str(id)`，混合类型 id 不再导致整个检索 TypeError；语义索引加载失败不再粘性缓存（瞬时错误恢复后自动重试）
- `sent_ledger` 发送证据语义修正：`tick_log` 预写行 `send_success` 改为 NULL（待终结），0/1 均为已终结且 `replies_sent_json` 只含实际送达文本；未终结行按 id 复认，游标越过后 UPDATE 进来的已送达回复不再丢失，永远不会把没发出去的回复记入 DH 历史
- `_apply_latest_request_evidence_guard` 接入 `_generate` 主链路（位于 LLM rewrite 之后作确定性兜底），与 `latest-request-first` 同一开关控制，默认关闭
- persona few-shot 加载改为逐行跳过坏 JSON/空行，一行坏数据不再静默清空整个示例库（同时修展示库 render 路径）
- 感知层 API 失败不再静默永久丢消息：失败 tick 不推进 diff 基线、不覆盖 `_last_perception`，下个 tick 自动重试；API 响应结构（messages/chat_list/chat_name）增加校验（wechat_reader 已提交 `ba0a1c6` 并推送，pin 已更新至 `72d4e9f`）
- 登录恢复增加 60 秒冷却与“每 tick 最多恢复一次”守卫，掉线时不再每个 tick 重复点击登录按钮，恢复后仍不就绪也不会单 tick 内无限往复
- 发送文本时若用户剪贴板是图片/文件：先备份图片（PNG），发送后还原，不再用空文本覆盖用户剪贴板
- `generator` 未读去重与 30 分钟历史窗口适配 OCR 消息的 `observed_at` 字段（DB 回读路径 `create_time` 为 None 时不再静默退化）；无时间戳消息不再被当作“当前时间”

### Changed
- 登录恢复的“主界面最小窗口尺寸”默认值改为引用感知层共享常量（`SmartPerceptionPipeline.MIN_WINDOW_*`），消除两处硬编码漂移
- 生成后多个独立大模型审查（latest-request / current-stance / answer-authority）在 >=2 个同时激活时合并为一次组合审查调用（`_run_combined_reply_review`），复用同一份 current_replies 并按原固定顺序应用结论；<=1 个激活或不支持组合的形态（stance 的 freshness_boundary 聚焦上下文模式）自动降级为原顺序调用，行为不变

### Added
- 组合审查路径的回归测试（单次调用合并、rewrite 固定顺序、无效子结论兜底、freshness 降级、单激活降级、_generate 端到端接线）
- 上述修复的回归测试（登录恢复冷却/单 tick 守卫、剪贴板图片备份还原）
- Multi-language support (in progress)
- Windows support exploration (in progress)

## [0.1.0] - 2024-01-01

### Added
- Core visual RPA engine with multimodal AI vision perception
- Memory system with vector-indexed chat history (BGE embeddings)
- Digital Twin feature for mimicking user's chatting style
- Smart Home Skill (Tuya device integration)
- 3D Printer Management Skill
- Multi-agent support (OpenClaw, Hermes, and other agent frameworks)
- WeChat auto-reply with LLM-powered response generation
- Screenshot-based UI element detection
- AppleScript and Accessibility API for UI interaction
- Configuration via YAML
- Homebrew installation guide
- English README (README_EN.md)
- Contributing guidelines (CONTRIBUTING.md)
- Issue templates (Bug Report, Feature Request, Skill Request)
- Code of Conduct
- Security Policy
- GitHub Pages landing page

### Technical Details
- **No protocol reverse engineering**: Uses visual perception instead
- **No database access**: Does not read WeChat's SQLite databases
- **macOS native**: Built with AppleScript and Accessibility API
- **Extensible Skill system**: Plugin architecture for custom automations
