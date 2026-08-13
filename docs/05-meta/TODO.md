# 待办 / 实验记录

## 2026-07-06: 回复生成架构切换（两步推理 → ReAct + Self-Refine）

### 变更说明
Phase B 两步推理（`_deep_analysis` / `_should_use_two_step` / `_plan_analysis` / `_gather_analysis_data`）
已被 **ReAct + Self-Refine** 架构替代。详细设计见 `docs/02-architecture/specs/REACT_SELF_REFINE_DESIGN.md`。

### 新架构
```
收到消息 → persona + 上下文 → ReAct 工具循环 → Self-Refine → 输出
                                      ↑              ↑
                                 think 工具    Feedback + Iterate
```

### 关键变更
- 删除 Hermes fallback 路径（`complex_llm_client`、`is_hermes` 分支）
- 删除两步推理原型（`_deep_analysis` 系列方法 + 测试脚本）
- 删除 `session_memory` 的 `bot_replies` 死代码
- `max_tokens` 统一提升到 10000
- 新增 `think` 工具（self-contained，不调外部服务）
- 新增 `feedback.md` / `iterate.md` prompt 文件
- 新增 Self-Refine 可观测字段（`self_refine_applied`、`feedback_decision` 等）
- `persona.md` 允许内心分析

### 待完成
- [ ] 真实群聊场景验证 5~10 条（投产观察）
- [ ] 评估延迟与成本影响
