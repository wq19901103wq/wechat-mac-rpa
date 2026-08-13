# Spec Review: ReAct + Self-Refine 回复生成重构（最终版）

## 审查结论

- **总体判断**：通过
- **最大风险**：无重大风险，改动范围已明确为现有 ReAct 循环的增量改造

---

## 问题清单

| 维度 | 问题 | 状态 | 说明 |
|---|---|---|---|
| 架构与设计 | ReAct 循环与现有工具循环边界不清 | ✅ 已解决 | 明确现有循环就是 ReAct，本次为增量改造（加 think 工具 + Self-Refine） |
| 架构与设计 | `reasoning_content` 回传责任不明 | ✅ 已解决 | 明确 `generator.py` 保留，`qwen_client.py` 透传；当前代码已部分实现 |
| 工具/API/接口 | `_self_refine` 返回类型不统一 | ✅ 已解决 | 已改为清晰的多返回值结构 |
| 安全性与风险 | 开关组合语义不明 | ✅ 已解决 | 四态组合表 + Self-Refine=1 自动开启 ReAct |
| 可行性 | 测试清理范围不清 | ✅ 已解决 | 列出受影响测试文件 |
| 清晰与完整 | “复杂场景”没量化 | ✅ 已解决 | 给出 skill 匹配 / 消息特征 / 上下文特征判定标准 |
| 清晰与完整 | persona.md 决策不明确 | ✅ 已解决 | 明确推荐修改方案 |
| 工具/API/接口 | qwen_client.py 接口变更未说明 | ✅ 已解决 | 新增行为约束章节 |
| 清晰与完整 | 缺少实现顺序 | ✅ 已解决 | Phase 1~4 已明确 |
| 清晰与完整 | 验收指标不够量化 | ✅ 已解决 | P95/P99 + think 触发率 |
| 可观测性 | admin 能否看到完整多轮轨迹 | ✅ 已解决 | 新增可观测性章节，明确 tick_log 新增字段和 generator 新增 debug 字段 |

---

## 关键结论

1. **当前 `generator.py` 已经是 ReAct 循环**，本次改动是增量改造而非重写。
2. **可观测性已覆盖**：tick_log 会增加 `self_refine_applied`、`feedback_decision`、`iterate_count`、`react_round_count`、`think_tool_called` 等字段，admin 可以看到完整轨迹。
3. **改动范围可控**：主要是加 think 工具、加 Self-Refine、删 Hermes/两步代码/死代码。

---

## 建议的下一步

规格已通过，建议调用 `writing-plans` 技能创建实现计划。
