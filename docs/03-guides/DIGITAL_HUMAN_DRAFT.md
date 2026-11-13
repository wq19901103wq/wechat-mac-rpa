# digital-human 接入

## 源码依赖

RPA 直接调用配置指定的 DH checkout，不复制其生成代码。默认目录为相邻的 `../digital-human`，通过常驻子进程隔离两个项目的 `src` 包和环境变量。更新 DH 源码后重启 Bot，才能加载更新。

需要使用支持 `src.generator.draft.DraftRuntime` 和 `observed` 截图输入协议的 DH 版本，并在该 checkout 中准备好 production 实例材料和依赖。公开仓库的任意版本不一定具备这些入口；本仓库不包含 DH 私有材料或未发布源码。子进程优先使用 DH 自己的 `.venv/bin/python`，没有时使用宿主解释器。

## 本机配置

1. 安装 RPA 和 DH 各自的依赖。
2. 复制 `.env.example` 为 `.env`，填写截图识别等服务配置。
3. 复制 `config/digital_human.json.example` 为 `config/digital_human.json`，填写 DH 实例名称与路径。

配置字段：

| 字段 | 用途 |
|---|---|
| `repository` | DH 源码 checkout |
| `instance` | 已准备 production 材料的实例 |
| `instances_root` | 私有实例材料目录，默认位于 DH 下的 `instances` |
| `env_file` | DH 自己的环境文件，默认位于 DH 下的 `.env` |
| `settings_file` | 可选的 DH 设置文件 |
| `timeout_seconds` | 每次初始化或调用的有限正数超时 |
| `review_client` | 回复审核使用的 DH 客户端，`private` 或 `group` |
| `direct_context` | 可选布尔值，默认 `false`；启用后不单独生成上下文摘要，初审与复审直接读取原始上下文 |
| `sent_replies_db` | 可选的实际发送账本，只读加载并排除已知机器人回复作为本人示例 |
| `sent_replies_history_db` | 可选的历史库；启动时以观察时间、原文、说话人和角色精确关联缺少聊天 ID 的旧发送记录，歧义或冲突时跳过 |
| `source_chat_ids` | 可选的 OCR 聊天 ID 到原始聊天 ID 的精确映射；不按显示名匹配同名群 |

配置中的相对路径以 RPA 根目录为基准。本机配置、环境文件、聊天与实例材料不进入 Git。

发送账本在 worker 启动时初始化，每次生成前按串行发送账本增量更新，重读最新回合以捕获发送结果回写；历史关联索引只在启动时构建。修改配置或源码后仍需重启 Bot。启动日志中的 `sent_replies` 仅记录摘要与计数，可用来核对实际加载和示例排除数量。

生成器启用 `context_reply_source=confirmed-sends-v1` 后，可靠匹配的旧机器人消息在内部上下文中标为 `[机器人代发]`，用于理解对话经过，不用于学习本人风格、偏好或近况，也不主动延续其中自创的梗。未标记的消息不等于已证明由本人手动发送。

账本仅排除已进入 DH 选例历史、且能可靠匹配到实际发送记录的机器人回复，不会导入新聊天。历史与账本的时间范围不重叠，或对应聊天不在选例历史中时，匹配和排除数可以为 0；账本成功加载不等于已有示例被排除。

## 草稿与自动回复

只生成草稿，不发送消息：

```bash
.venv/bin/python scripts/digital_human_draft.py --screenshot
.venv/bin/python scripts/digital_human_draft.py --help
```

也可通过 `--input-file` 提供结构化输入，通过 `--stream` 逐行输入并复用运行时。没有本机配置时，可显式传入 `--repository`、`--instance` 等参数。`--save-input` 保存的内容可能包含私人聊天，只应保存到被 Git 忽略的本地目录。

自动回复入口为 `run_bot.py`。群聊默认关闭；启用需配置 `WECHAT_GROUP_REPLY_ENABLED=1` 并重启。实际发送仍受现有静默模式与白名单限制。

截图输入使用首次观察时间，该时间不代表微信发送时间。当前上下文会排除时间不明和超出时间窗口的历史消息。默认在生成后由大模型做上下文分析、首审和放行复核；启用 `direct_context` 后，上下文理解随生成完成，不再单独等待摘要，独立初审与放行复核直接核对原始上下文和生成材料。两种模式下拦截或审核失败时均不发送，生成失败时不回退旧生成器。

单次草稿 CLI 仅用于检查 DH 输出，不经过 Bot 的发送前审核流程；其输出不能视为审核放行结果。

通用网页工具由 DH 的 `src/tools/web.py` 提供，复用原 RPA 的搜索解析、正文提取和页内缓存检索；RPA 通过 `third_party/digital-human` 的 `digital_human` 命名空间引用，不再由 worker 向 DH 反向注入 RPA 工具。该目录当前尚未提交固定为 gitlink，草稿运行仍按本机配置引用相邻 DH checkout。搜索抓取 360（`so.com`）结果页，不使用带 Key 的搜索 API；浏览抓取静态正文，不执行 JavaScript。DH 显式启用网页工具后自行装配 backend，HTTP 传输沿用公网地址、重定向、8 秒和 512 KiB 限制，Wiki 与网页工具仍共用每轮两次查询额度。工具实现摘要进入运行绑定和生成缓存；旧策略不会因此自动启用联网，历史评测仍禁止实时联网。线上是否加载应以当前启动记录和运行绑定为准。

## 共用微信历史

历史读取和输入转换依赖独立的 `personal-context` 包。使用历史模式的环境需要安装共用源码：

```sh
.venv/bin/python -m pip install --no-deps -e /path/to/personal-context
.venv/bin/python scripts/wechat_history.py chats --limit 20
.venv/bin/python scripts/wechat_history.py dh-input --chat 'CHAT_ID' --limit 50
.venv/bin/python scripts/digital_human_draft.py --history-chat 'CHAT_ID' --history-limit 50
```

`--history-db` 指定 RPA 原始 SQLite，默认取 `WECHAT_HISTORY_DB` 或本仓库默认数据库；
`--history-account`、`--history-source` 提供账号和来源绑定。适配器只读同一份原始库，
不复制消息。DH 输入使用 `observed` 格式，RPA 行号不作为 DH 冻结消息 ID。
重新建立虚拟环境时需重新安装共用包，截图和原有结构化输入模式不依赖该包。

## 发布前检查

```bash
.venv/bin/python scripts/check_privacy.py --staged
```

该命令检查完整 Git index 的实际内容，不依赖工作区文件是否已经改回。它可发现常见密钥、账号标识和私人路径，仍需人工审阅提交内容，避免提交真实聊天原文与身份信息。
