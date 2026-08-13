# CLAUDE.md

> wechat-mac-rpa · 每次会话全量加载 · < 150 行

## Karpathy 铁律

1. **先澄清再实现**：不确定需求先问。禁止脑补用户意图。
2. **越简单越好**：3 行能解决不写 30 行。不改无关代码。
3. **手术式修改**：只动目标代码。禁止顺手重构、格式化、加注释。
4. **验证再报告**：改完必须跑测试/截图确认。禁止"应该没问题"。

## 修改流程

1. Read 目标文件，确认当前状态
2. 口头说明改什么、为什么、预期效果
3. 等用户明确同意（"好""改""执行"）
4. 最小化修改，执行
5. 验证 → 报告结果（失败就说失败）

## 禁止事项

- 未经同意执行修改、脚本、重启服务
- "顺便"改无关代码、删除现有功能
- 编造不确定的信息（说"不确定"）
- 用猜代替验证（先复现最小 case 对比）

## 项目速查

```
src/bot/wechat_bot.py          # L5 主循环
src/reply/generator.py         # L4 回复生成
src/badcase/judge_worker.py    # Judge 评分
src/perception/smart_pipeline.py # L3.5 感知
src/memory/engine.py           # L4 记忆
scripts/admin.py               # 管理后台 :8766
data/persona.md                # Bot 私人人设（Git 忽略）
```

## 规则分层

| 文件 | 加载时机 | 内容 |
|------|---------|------|
| `CLAUDE.md` | 每次会话 | 核心铁律 + 速查 |
| `.claude/rules/frontend.md` | 改 admin.py 时 | Playwright 验证 |
| `.claude/rules/debugging.md` | 改 src/ 时 | 调试流程 + 常见坑 |
