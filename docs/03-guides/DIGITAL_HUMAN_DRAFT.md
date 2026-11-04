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

配置中的相对路径以 RPA 根目录为基准。本机配置、环境文件、聊天与实例材料不进入 Git。

## 草稿与自动回复

只生成草稿，不发送消息：

```bash
.venv/bin/python scripts/digital_human_draft.py --screenshot
.venv/bin/python scripts/digital_human_draft.py --help
```

也可通过 `--input-file` 提供结构化输入，通过 `--stream` 逐行输入并复用运行时。没有本机配置时，可显式传入 `--repository`、`--instance` 等参数。`--save-input` 保存的内容可能包含私人聊天，只应保存到被 Git 忽略的本地目录。

自动回复入口为 `run_bot.py`。群聊默认关闭；启用需配置 `WECHAT_GROUP_REPLY_ENABLED=1` 并重启。实际发送仍受现有静默模式与白名单限制。

截图输入使用首次观察时间，该时间不代表微信发送时间。当前上下文会排除时间不明和超出时间窗口的历史消息。生成后由大模型做上下文分析、首审和放行复核；拦截或审核失败时不发送，生成失败时不回退旧生成器。

单次草稿 CLI 仅用于检查 DH 输出，不经过 Bot 的发送前审核流程；其输出不能视为审核放行结果。

## 发布前检查

```bash
.venv/bin/python scripts/check_privacy.py --staged
```

该命令检查完整 Git index 的实际内容，不依赖工作区文件是否已经改回。它可发现常见密钥、账号标识和私人路径，仍需人工审阅提交内容，避免提交真实聊天原文与身份信息。
