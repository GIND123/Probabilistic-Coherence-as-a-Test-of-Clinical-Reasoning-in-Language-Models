#!/usr/bin/env bash
# #6 token-logprob baseline across the full sweep. Resumable per chunk.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export HF_TOKEN=$(grep '^hf=' .env | cut -d= -f2)
PY=.venv/bin/python
MODELS="${MODELS:-qwen3-4b-nothink qwen3-4b-think qwen3-8b-nothink qwen3-8b-think \
medgemma-1.5-4b med42-8b gpt-oss-20b medgemma-27b qwen3-32b-nothink qwen3-32b-think \
r1-distill-32b}"
for m in $MODELS; do
  echo "=============================================================="
  echo "LOGPROB $m   $(date -Is)"
  $PY -m coherence.run.logprob_baseline --model "$m" \
      --max-num-seqs "${MAX_NUM_SEQS:-192}" \
      2>&1 | grep -vE "^\(|it/s\]|Adding requests|Processed prompts"
  echo "exit=$? for $m at $(date -Is)"
done
echo "### LOGPROB SWEEP DONE $(date -Is)"
