#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
PYTHON=/venv/main/bin/python
SOURCE=experiments/syllogimous_neural_computer/targeted_recurrent_context_parallel
OUT=experiments/syllogimous_neural_computer/targeted_adaptive_context_replication
mkdir -p "$OUT"
pids=()
for seed in 11 23 37; do
  OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 "$PYTHON" \
    -m experiments.syllogimous_neural_computer.evaluate_adaptive_context \
    --controller experiments/syllogimous_neural_computer/vast_continual_sharp.pt \
    --selector "$SOURCE/seed_${seed}.pt" --report "$OUT/seed_${seed}.json" \
    --calibration-streams 128 --eval-streams 128 --contexts 8 --delay 0 \
    --seed "$seed" --device cuda >"$OUT/seed_${seed}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
