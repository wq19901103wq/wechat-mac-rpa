# 共用微信原始消息组件

截图/OCR、消息模型、会话点击、去重、原始 SQLite 保存与读取由独立的 [wechat-reader 仓库](https://github.com/wq19901103wq/wechat-reader) 唯一维护，包名 `wechat-reader`，保留原 MIT 许可。RPA 的旧导入入口兼容；回复选择、回复标记、发送、DH 和 Wiki 记忆仍留在 RPA。个人助理独立维护在 `personal-context`；RPA、DH 保留原仓库和历史，不使用整个 RPA submodule。

## 安装与升级

组件尚未发布到 PyPI，依赖通过公开 GitHub HTTPS 地址匿名安装，无需 GitHub token：

```sh
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` 的 `wechat-reader[capture]` 与个人助理 `pyproject.toml` 固定到相同完整提交号；升级时同步更新两库。查询消费者无需 `[capture]`。联合开发或离线安装可先将独立 `wechat-reader` 库检出到固定提交，再在 RPA 环境安装 `.venv/bin/python -m pip install -e '/path/to/wechat-reader[capture]'`；离线安装其余依赖时应使用已准备好的本地 wheel，不从 PyPI 查找同名包。仓库位置变化不影响已安装的 wheel，无需因此重启采集服务。

RPA 原有默认数据库和截图路径保持不变，不需要迁移或清理历史库。共享模型保留旧回复字段和数据库列用于兼容；共享采集不作回复决策。`GlobalStore.merge_tick` 仍返回未回复列表，共享 `RawMessageStore.merge_tick` 返回本轮新消息。

## 独立采集

仅采集而不运行回复 Bot 时，可用：

```sh
.venv/bin/wechat-reader --db ./data/db/chat_history.db collect --once --env-file .env
.venv/bin/wechat-reader --db ./data/db/chat_history.db collect --env-file .env
```

`--once` 只采当前会话；持续模式会打开可见未读会话，改变已读状态。可用 `--current-only` 关闭切换，或重复 `--chat-name` 限定采集范围。首次标题／消息识别需要有效的 `DASHSCOPE_API_KEY`；本地 OCR 不能替代首次标题识别。`--env-file` 只读取采集所需的 DASHSCOPE 配置，已有环境变量优先。不加载回复或记忆模块。桌面、微信登录、屏幕录制及辅助功能权限需保持可用，完整条件及 LaunchAgent 安装见读取库 README。

独立采集循环与新版 RPA tick 共用同一用户的 UI 文件锁，避免不同数据库的采集者争抢窗口。一次只启一个采集所有者；旧的运行中 Bot 不会自动加载新锁，切换时先停止旧采集，再启动更新后的入口。只读消费者可同时读取同一原始库。仓库整理不启动回复 Bot，也不改变已有服务与聊天库。

## GitHub Actions 安装公开依赖

CI 与 Quality 的 coverage job 直接通过 `requirements-dev.txt` 安装固定提交的公开读取库，无需额外的仓库 secret。外部 fork 的 pull request 使用相同的匿名安装方式。不要把运行时 API Key、登录态或其他凭据写入 requirements、Git URL、代码或日志。

数据库、截图、密钥、登录态与日志继续保存在本地被 Git 忽略的位置，不随包分发。
