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
