#!/usr/bin/env bash
# Full pipeline: finish the model sweep, sync, train the ELR LoRA, sync again.
# Every stage is resumable; rerunning skips completed work.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export HF_TOKEN=$(grep '^hf=' .env | cut -d= -f2)
PY=.venv/bin/python

# Wait for any sweep already running to finish before touching the GPU.
while pgrep -f "coherence.run.runner" >/dev/null; do sleep 30; done

echo "### STAGE 1  model sweep  $(date -Is)"
bash scripts/sweep.sh

# Second pass. Tasks are resumable, so this only re-attempts whatever the
# first pass failed to finish; completed tasks are skipped in seconds.
echo "### STAGE 1-retry  $(date -Is)"
bash scripts/sweep.sh

echo "### STAGE 1b  gpt-oss-120b, A1 core only (CPU-offloaded, does not fit VRAM)"
MODELS="gpt-oss-120b" TASKS="a1_posterior" MAX_NUM_SEQS=32 bash scripts/sweep.sh \
  || echo "120b arm failed (non-fatal)"

echo "### STAGE 1c  MIMIC external-validity arm (local only, never uploaded)"
for m in qwen3-8b-nothink qwen3-32b-nothink medgemma-27b; do
  $PY -m coherence.run.mimic_runner --model "$m" \
    2>&1 | grep -vE "^\(|it/s\]|Adding requests|Processed prompts" || true
done

echo "### STAGE 2  sync results to the Hub  $(date -Is)"
$PY -m coherence.hub.hf_sync || echo "sync failed (non-fatal)"

echo "### STAGE 3  ELR LoRA training  $(date -Is)"
$PY -m coherence.train.build_lr_dataset
$PY -m coherence.train.train_lr_lora --base Qwen/Qwen3-8B \
    --out build/models/elr-lora-qwen3-8b --epochs 2

echo "### STAGE 4  analysis  $(date -Is)"
$PY -m coherence.analysis.run_analysis

echo "### STAGE 5  final sync  $(date -Is)"
$PY -m coherence.hub.hf_sync || echo "sync failed (non-fatal)"
echo "### DONE  $(date -Is)"
