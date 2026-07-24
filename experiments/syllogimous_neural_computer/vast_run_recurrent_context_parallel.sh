#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
PYTHON=/venv/main/bin/python
OUT=experiments/syllogimous_neural_computer/targeted_recurrent_context_parallel
mkdir -p "$OUT"
pids=()
for seed in 11 23 37; do
  OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 "$PYTHON" \
    -m experiments.syllogimous_neural_computer.train_context_selector \
    --controller experiments/syllogimous_neural_computer/vast_continual_sharp.pt \
    --checkpoint "$OUT/seed_${seed}.pt" --report "$OUT/seed_${seed}.json" \
    --train-streams 64 --eval-streams 64 --contexts 8 --delay 0 \
    --context-cost 0.001 --target-temperature 1.0 \
    --seed "$seed" --device cuda >"$OUT/seed_${seed}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
