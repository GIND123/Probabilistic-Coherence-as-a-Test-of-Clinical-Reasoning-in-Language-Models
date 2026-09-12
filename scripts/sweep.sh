#!/usr/bin/env bash
# Full CoDx sweep. One model at a time (single GPU), resumable at task
# granularity: rerunning skips any task whose parquet already exists.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export HF_TOKEN=$(grep '^hf=' .env | cut -d= -f2)

PY=.venv/bin/python
BATTERY=build/battery/codx_battery_v1.json

CORE="a1_posterior a4_posterior a2_posterior a3_posterior"
METHOD="elr_prior elr_weights_numeric elr_weights_logodds elr_weights_ordinal"
ABL="a1_answer cot_posterior a1_narrative"

MODELS="${MODELS:-qwen3-4b-nothink qwen3-4b-think qwen3-8b-nothink qwen3-8b-think \
medgemma-1.5-4b med42-8b gpt-oss-20b medgemma-27b qwen3-32b-nothink qwen3-32b-think \
r1-distill-32b}"
TASKS="${TASKS:-$CORE $METHOD $ABL}"

for m in $MODELS; do
  echo "=============================================================="
  echo "MODEL $m   $(date -Is)"
  echo "=============================================================="
  $PY -m coherence.run.runner --model "$m" --battery "$BATTERY" \
      --tasks $TASKS --max-num-seqs "${MAX_NUM_SEQS:-256}" \
      2>&1 | grep -vE "^\(|it/s\]|Adding requests|Processed prompts"
  echo "exit=$? for $m at $(date -Is)"
done
