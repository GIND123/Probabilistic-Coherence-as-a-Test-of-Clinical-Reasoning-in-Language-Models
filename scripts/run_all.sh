#!/usr/bin/env bash
# Full pipeline: finish the model sweep, sync, train the ELR LoRA, sync again.
# Every stage is resumable; rerunning skips completed work.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export HF_TOKEN=$(grep '^hf=' .env | cut -d= -f2)
PY=.venv/bin/python

# Hub archival is ON by default. Set CODX_HF_SYNC=0 to suppress it.
# Every sync stage below goes through this wrapper so there is exactly one
# switch, rather than three call sites that can drift apart.
hf_sync() {
  if [ "${CODX_HF_SYNC:-1}" != "1" ]; then
    echo "  [hf] skipped (CODX_HF_SYNC=0)"
    return 0
  fi
  $PY -m coherence.hub.hf_sync "$@" || echo "  [hf] sync failed (non-fatal)"
}

# Wait for ANY of our GPU jobs still running before touching the GPU.
# Training and the sweep both want the whole card; overlapping them OOMs the
# vLLM engine at load time and silently kills the sweep.
while pgrep -f "coherence.run.runner|coherence.train.train_lr_lora|coherence.run.mimic_runner" >/dev/null; do
  echo "  waiting for a running GPU job to finish  $(date -Is)"; sleep 60
done

echo "### STAGE 1  model sweep  $(date -Is)"
bash scripts/sweep.sh

# Second pass. Tasks are resumable, so this only re-attempts whatever the
# first pass failed to finish; completed tasks are skipped in seconds.
echo "### STAGE 1-retry  $(date -Is)"
bash scripts/sweep.sh

# STAGE 1b  gpt-oss-120b -- DISABLED.
# 63 GB of MXFP4 weights on a 48 GB card means cpu_offload_gb=28, so every
# forward pass streams 28 GB over PCIe. Measured: 16 h of generation produced
# 0 of 20 chunks, i.e. <0.035 items/s against 3.2 for gpt-oss-20b. The A1 core
# alone would take ~13 days, and the arm blocks analysis and sync while it
# runs. Set CODX_RUN_120B=1 to attempt it anyway.
if [ "${CODX_RUN_120B:-0}" = "1" ]; then
  echo "### STAGE 1b  gpt-oss-120b, A1 core only (CPU-offloaded)"
  MODELS="gpt-oss-120b" TASKS="a1_posterior" MAX_NUM_SEQS=32 bash scripts/sweep.sh \
    || echo "120b arm failed (non-fatal)"
else
  echo "### STAGE 1b  gpt-oss-120b SKIPPED (infeasible on 48 GB; see registry notes)"
fi

echo "### STAGE 1c  MIMIC external-validity arm (local only, never uploaded)"
for m in qwen3-8b-nothink qwen3-32b-nothink medgemma-27b; do
  $PY -m coherence.run.mimic_runner --model "$m" \
    2>&1 | grep -vE "^\(|it/s\]|Adding requests|Processed prompts" || true
done

echo "### STAGE 2  sync results to the Hub  $(date -Is)"
hf_sync

echo "### STAGE 3  ELR LoRA training  $(date -Is)"
# Resumable: a completed run leaves final_metrics.json with the selected
# checkpoint. Do not burn 90 GPU-minutes retraining an adapter we already have.
if [ -f build/models/elr-lora-qwen3-8b/final_metrics.json ]; then
  echo "  adapter already trained -- skipping"
  cat build/models/elr-lora-qwen3-8b/final_metrics.json
else
  $PY -m coherence.train.build_lr_dataset
  $PY -m coherence.train.train_lr_lora --base Qwen/Qwen3-8B \
      --out build/models/elr-lora-qwen3-8b --epochs 2
fi
hf_sync --model-only

echo "### STAGE 3b  ELR-Fusion with the trained adapter (ablation 17)  $(date -Is)"
MODELS="qwen3-8b-elr-lora" \
  TASKS="elr_prior elr_weights_numeric elr_weights_logodds elr_weights_ordinal" \
  bash scripts/sweep.sh || echo "elr-lora arm failed (non-fatal)"

echo "### STAGE 4  analysis  $(date -Is)"
$PY -m coherence.analysis.run_analysis

echo "### STAGE 5  final sync  $(date -Is)"
hf_sync

# STAGE 6  Pin and verify the release. Cheap (two Hub listings, no download),
# and it is the only step that checks the artifacts a reader is told to fetch
# actually exist. A non-zero exit here means the paper cites something the
# release does not contain.
echo "### STAGE 6  pin + verify the release  $(date -Is)"
$PY -m coherence.hub.manifest || echo "  [hf] manifest failed (non-fatal)"
$PY -m coherence.hub.hf_pull verify || echo "  [hf] RELEASE INCOMPLETE -- see above"

echo "### DONE  $(date -Is)"
