# Persona Few-shot 生产部署

`data/few_shot/` 含真实聊天衍生数据，被 `.gitignore` 排除，不会随 Git 部署。

## 生成与审核

1. 在生产机器本地运行 `scripts/build_persona_few_shots.py`。
2. 人工检查 `data/few_shot/persona_examples.md`，删除仍包含语义隐私、事实或指令注入的样本。
3. 将 `data/few_shot/report.json` 的 `review_status` 从 `pending` 改为 `approved`。
4. 设置 `ENABLE_PERSONA_FEW_SHOTS=1` 后重启 Bot。

## 对象/场景/话题分层生成

需要生成更大的本地样本库时，使用分层策略：

```bash
python scripts/build_persona_few_shots.py \
  --output data/few_shot_v4 \
  --limit 1500 \
  --stratified \
  --min-per-chat 4 \
  --max-per-chat 8 \
  --humor-ratio 0.4 \
  --sincere-ratio 0.1 \
  --temporal-holdout \
  --holdout-ratio 0.2 \
  --holdout-limit 240

python scripts/build_persona_few_shot_index.py \
  --examples data/few_shot_v4/persona_examples.jsonl \
  --output data/few_shot_v4/persona_embeddings.npz

python scripts/evaluate_persona_few_shots.py \
  --examples data/few_shot_v4/persona_examples.jsonl \
  --holdout data/few_shot_v4/holdout_cases.jsonl \
  --output data/few_shot_v4
```

`report.json` 提供全局分布；`object_report.json` 按脱敏 `chat_id` 展示每个聊天对象的场景、话题、幽默比例和示例 ID。时间留出的较新真实对话写入 `holdout_cases.jsonl`，召回结果和汇总指标写入 `holdout_retrieval_cases.jsonl`、`holdout_evaluation.json`。报告默认保持 `pending`，人工复核后才能接入生产。

不要把 JSONL、Markdown 或 report 文件提交到公开仓库。换机器部署时应通过加密私有制品复制，或在目标机器重新生成并审核。

生产日志只记录召回样本 ID，不应持久化 few-shot 正文。`PERSONA_FEW_SHOT_ALLOW_UNREVIEWED=1` 仅用于本地测试。
