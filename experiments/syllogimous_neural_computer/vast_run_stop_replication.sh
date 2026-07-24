#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
PYTHON=/venv/main/bin/python
SWEEP=experiments/syllogimous_neural_computer/targeted_dense_stop_weights
OUT=experiments/syllogimous_neural_computer/targeted_stop_replication
mkdir -p "$OUT"
cp "$SWEEP/weight_0p5.pt" "$OUT/autonomous_seed_11.pt"
cp "$SWEEP/weight_0p5.json" "$OUT/autonomous_seed_11.json"
for seed in 23 37; do
  "$PYTHON" -m experiments.syllogimous_neural_computer.train_consolidation \
    --controller experiments/syllogimous_neural_computer/vast_continual_sharp.pt \
    --checkpoint "$OUT/autonomous_seed_${seed}.pt" \
    --report "$OUT/autonomous_seed_${seed}.json" \
    --train-streams 64 --eval-streams 128 --contexts 8 --delay 0 --attempts 5 \
    --tournament-candidates 4 --rehearsal-groups 1 --autonomous-stop \
    --autonomous-storage-value 0.01 --stop-loss-weight 0.5 \
    --seed "$seed" --device cuda
done
for seed in 11 23 37; do
  "$PYTHON" -m experiments.syllogimous_neural_computer.train_consolidation \
    --controller experiments/syllogimous_neural_computer/vast_continual_sharp.pt \
    --initial-policy "$OUT/autonomous_seed_${seed}.pt" \
    --checkpoint "$OUT/forced_seed_${seed}.pt" \
    --report "$OUT/forced_seed_${seed}.json" \
    --train-streams 0 --eval-streams 128 --contexts 8 --delay 0 --attempts 5 \
    --rehearsal-groups 1 --seed "$seed" --device cuda
done
