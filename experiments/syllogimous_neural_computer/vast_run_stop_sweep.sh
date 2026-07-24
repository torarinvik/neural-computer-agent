#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
PYTHON=/venv/main/bin/python
OUT=experiments/syllogimous_neural_computer/targeted_stop_sweep
mkdir -p "$OUT"
for value in 0.003 0.01 0.03; do
  tag=${value/./p}
  "$PYTHON" -m experiments.syllogimous_neural_computer.train_consolidation \
    --controller experiments/syllogimous_neural_computer/vast_continual_sharp.pt \
    --checkpoint "$OUT/value_${tag}.pt" --report "$OUT/value_${tag}.json" \
    --train-streams 64 --eval-streams 128 --contexts 8 --delay 0 --attempts 5 \
    --tournament-candidates 4 --rehearsal-groups 1 --autonomous-stop \
    --autonomous-storage-value "$value" --seed 11 --device cuda
done
