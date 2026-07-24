#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
PYTHON=/venv/main/bin/python
OUT=experiments/syllogimous_neural_computer/targeted_dense_stop_weights
mkdir -p "$OUT"
for weight in 0.5 1.0 2.0; do
  tag=${weight/./p}
  "$PYTHON" -m experiments.syllogimous_neural_computer.train_consolidation \
    --controller experiments/syllogimous_neural_computer/vast_continual_sharp.pt \
    --checkpoint "$OUT/weight_${tag}.pt" --report "$OUT/weight_${tag}.json" \
    --train-streams 64 --eval-streams 128 --contexts 8 --delay 0 --attempts 5 \
    --tournament-candidates 4 --rehearsal-groups 1 --autonomous-stop \
    --autonomous-storage-value 0.01 --stop-loss-weight "$weight" \
    --seed 11 --device cuda
done
