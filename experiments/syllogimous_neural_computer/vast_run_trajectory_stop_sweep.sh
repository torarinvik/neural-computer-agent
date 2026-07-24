#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
PYTHON=/venv/main/bin/python
OUT=experiments/syllogimous_neural_computer/targeted_trajectory_stop_sweep
mkdir -p "$OUT"
for weight in 1 2 4 8; do
  "$PYTHON" -m experiments.syllogimous_neural_computer.train_consolidation \
    --controller experiments/syllogimous_neural_computer/vast_continual_sharp.pt \
    --checkpoint "$OUT/weight_${weight}.pt" \
    --report "$OUT/weight_${weight}.json" \
    --train-streams 128 --eval-streams 256 --contexts 8 --delay 0 --attempts 5 \
    --tournament-candidates 4 --rehearsal-groups 1 --autonomous-stop \
    --trajectory-stop --autonomous-storage-value 0.01 \
    --stop-loss-weight "$weight" --seed 11 --device cuda
  "$PYTHON" -m experiments.syllogimous_neural_computer.train_consolidation \
    --controller experiments/syllogimous_neural_computer/vast_continual_sharp.pt \
    --initial-policy "$OUT/weight_${weight}.pt" \
    --checkpoint "$OUT/forced_weight_${weight}.pt" \
    --report "$OUT/forced_weight_${weight}.json" \
    --train-streams 0 --eval-streams 256 --contexts 8 --delay 0 --attempts 5 \
    --rehearsal-groups 1 --seed 11 --device cuda
done
