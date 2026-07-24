#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
PYTHON=/venv/main/bin/python
OUT=experiments/syllogimous_neural_computer/targeted_generalizing_policy
mkdir -p "$OUT"
for seed in 11 23 37; do
  "$PYTHON" -m experiments.syllogimous_neural_computer.train_consolidation \
    --controller experiments/syllogimous_neural_computer/vast_continual_sharp.pt \
    --checkpoint "$OUT/seed_${seed}.pt" --report "$OUT/seed_${seed}.json" \
    --train-streams 128 --eval-streams 128 --contexts 8 --delay 0 --attempts 4 \
    --rehearsal-groups 1 --generalization-reward-weight 1.0 \
    --seed "$seed" --device cuda
done
