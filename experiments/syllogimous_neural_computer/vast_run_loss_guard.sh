#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
PYTHON=/venv/main/bin/python
BASE=experiments/syllogimous_neural_computer/targeted_consolidation
OUT=experiments/syllogimous_neural_computer/targeted_loss_guard
mkdir -p "$OUT"
for seed in 11 23 37; do
  "$PYTHON" -m experiments.syllogimous_neural_computer.train_consolidation \
    --controller experiments/syllogimous_neural_computer/vast_continual_sharp.pt \
    --initial-policy "$BASE/trained_seed_${seed}.pt" \
    --checkpoint "$OUT/strict_seed_${seed}.pt" \
    --report "$OUT/strict_seed_${seed}.json" \
    --train-streams 0 --eval-streams 128 --contexts 8 --delay 0 --attempts 4 \
    --loss-tolerance 0.0 --seed "$seed" --device cuda
done
