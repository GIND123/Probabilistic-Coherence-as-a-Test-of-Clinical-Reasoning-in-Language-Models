#!/usr/bin/env bash
# Compact progress report for the running sweep.
cd "$(dirname "$0")/.."
TASKS="a1_posterior a4_posterior a2_posterior a3_posterior elr_prior elr_weights_numeric elr_weights_logodds elr_weights_ordinal a1_answer cot_posterior a1_narrative"
N_TASKS=$(echo $TASKS | wc -w)
printf "%-20s %6s  %s\n" MODEL DONE "in-progress"
for d in results/raw/*/; do
  m=$(basename "$d")
  done_n=$(ls "$d"/*.parquet 2>/dev/null | wc -l)
  ip=$(ls -d "$d"/*.chunks 2>/dev/null | head -1)
  if [ -n "$ip" ]; then
    t=$(basename "$ip" .chunks); c=$(ls "$ip"/*.parquet 2>/dev/null | wc -l)
    printf "%-20s %3d/%-2d  %s (%d chunks)\n" "$m" "$done_n" "$N_TASKS" "$t" "$c"
  else
    printf "%-20s %3d/%-2d  -\n" "$m" "$done_n" "$N_TASKS"
  fi
done
echo
echo "stage: $(grep '^###' logs/run_all.log 2>/dev/null | tail -1)"
echo "gpu:   $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
echo "alive: $(pgrep -fc 'coherence.run.runner|coherence.train' 2>/dev/null || echo 0) runner(s)"
