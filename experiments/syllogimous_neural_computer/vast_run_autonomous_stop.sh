#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
PYTHON=/venv/main/bin/python
OUT=experiments/syllogimous_neural_computer/targeted_autonomous_stop
mkdir -p "$OUT"
for seed in 11 23 37; do
  "$PYTHON" -m experiments.syllogimous_neural_computer.train_consolidation \
    --controller experiments/syllogimous_neural_computer/vast_continual_sharp.pt \
    --checkpoint "$OUT/autonomous_seed_${seed}.pt" \
    --report "$OUT/autonomous_seed_${seed}.json" \
    --train-streams 64 --eval-streams 128 --contexts 8 --delay 0 --attempts 5 \
    --tournament-candidates 4 --rehearsal-groups 1 --autonomous-stop \
    --autonomous-storage-value 0.01 --stop-loss-weight 4 \
    --seed "$seed" --device cuda
  "$PYTHON" -m experiments.syllogimous_neural_computer.train_consolidation \
    --controller experiments/syllogimous_neural_computer/vast_continual_sharp.pt \
    --initial-policy "$OUT/autonomous_seed_${seed}.pt" \
    --checkpoint "$OUT/forced_seed_${seed}.pt" \
    --report "$OUT/forced_seed_${seed}.json" \
    --train-streams 0 --eval-streams 128 --contexts 8 --delay 0 --attempts 5 \
    --rehearsal-groups 1 --seed "$seed" --device cuda
done
