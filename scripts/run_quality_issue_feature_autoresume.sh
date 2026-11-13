#!/bin/bash
# 2026-11-11: 质量特征组（Q01-Q11）Judge 候选的自动续跑。
# 背景：codex 账户限额 2026-11-13 21:05 恢复；此前特征提取预检全部被限额打断。
# 行为：睡到 11/13 21:20 本地时间，循环续跑 extract-features（断点检查点），
#       全部 1360 条 valid 后依次 train 与 compare-stable-judge-classifier。
#       不采用、不改生产配置，只产出对比结果留档。
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
RUN="$ROOT/scripts/run_judge_feature_classifier_experiment.py"
PRIV="$ROOT/data/reports/private/judge_feature_classifier_v1"
CKPT="$PRIV/quality_issue_feature_checkpoint.json"
RESULT="$PRIV/quality_issue_learning_curve_result.json"
CMP="$PRIV/stable_judge_vs_feature_classifier_quality_issue_1200.json"
LOG="$PRIV/quality_issue_autoresume.log"
STATUS="$PRIV/quality_issue_autoresume_status.json"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# 睡到 2026-11-13 21:20（本地，缓冲于声明的 21:05 恢复点之后）
TARGET=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-13 21:20:00" "+%s" 2>/dev/null || echo 0)
NOW=$(date "+%s")
if [ "$TARGET" -gt "$NOW" ]; then
  log "sleeping $(( (TARGET - NOW) / 60 )) minutes until 2026-11-13 21:20 local"
  sleep $(( TARGET - NOW ))
fi

extract_done() {
  "$PY" - "$CKPT" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
from collections import Counter
c = Counter(e.get("status") for e in d.get("entries", {}).values())
print("valid" if c.get("valid", 0) == 1360 and c.get("error", 0) == 0 else f"pending:{dict(c)}")
EOF
}

ATTEMPT=0
while [ "$ATTEMPT" -lt 30 ]; do
  ATTEMPT=$((ATTEMPT + 1))
  log "extract-features attempt $ATTEMPT start"
  "$PY" "$RUN" extract-features --retry-failed --workers 8 \
      --checkpoint "$CKPT" >> "$LOG" 2>&1
  STATE=$(extract_done)
  log "extract-features attempt $ATTEMPT state: $STATE"
  if [ "$STATE" = "valid" ]; then
    break
  fi
  sleep 1800
done

if [ "$STATE" != "valid" ]; then
  log "ABORT: extraction incomplete after $ATTEMPT attempts: $STATE"
  printf '{"status":"extraction_incomplete","attempts":%d,"state":"%s"}\n' "$ATTEMPT" "$STATE" > "$STATUS"
  exit 1
fi

log "train start"
"$PY" "$RUN" train \
    --features "$CKPT" \
    --output "$RESULT" >> "$LOG" 2>&1 || { log "ABORT: train failed"; printf '{"status":"train_failed"}\n' > "$STATUS"; exit 1; }

log "compare start"
"$PY" "$RUN" compare-stable-judge-classifier \
    --learning-curve "$RESULT" \
    --output "$CMP" >> "$LOG" 2>&1 || { log "ABORT: compare failed"; printf '{"status":"compare_failed"}\n' > "$STATUS"; exit 1; }

log "DONE: comparison at $CMP"
printf '{"status":"done","comparison":"%s"}\n' "$CMP" > "$STATUS"
