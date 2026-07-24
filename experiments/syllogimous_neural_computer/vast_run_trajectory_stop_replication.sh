#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
PYTHON=/venv/main/bin/python
OUT=experiments/syllogimous_neural_computer/targeted_trajectory_stop_replication
mkdir -p "$OUT"
cp experiments/syllogimous_neural_computer/targeted_trajectory_stop_sweep/weight_1.pt \
   "$OUT/autonomous_seed_11.pt"
cp experiments/syllogimous_neural_computer/targeted_trajectory_thresholds/threshold_0p60.json \
   "$OUT/autonomous_seed_11.json"
cp experiments/syllogimous_neural_computer/targeted_trajectory_stop_sweep/forced_weight_1.json \
   "$OUT/forced_seed_11.json"
for seed in 23 37; do
  "$PYTHON" -m experiments.syllogimous_neural_computer.train_consolidation \
    --controller experiments/syllogimous_neural_computer/vast_continual_sharp.pt \
    --checkpoint "$OUT/autonomous_seed_${seed}.pt" \
    --report "$OUT/autonomous_seed_${seed}.json" \
    --train-streams 128 --eval-streams 256 --contexts 8 --delay 0 --attempts 5 \
    --tournament-candidates 4 --rehearsal-groups 1 --autonomous-stop \
    --trajectory-stop --autonomous-storage-value 0.01 --stop-loss-weight 1 \
    --stop-threshold 0.60 --seed "$seed" --device cuda
  "$PYTHON" -m experiments.syllogimous_neural_computer.train_consolidation \
    --controller experiments/syllogimous_neural_computer/vast_continual_sharp.pt \
    --initial-policy "$OUT/autonomous_seed_${seed}.pt" \
    --checkpoint "$OUT/forced_seed_${seed}.pt" \
    --report "$OUT/forced_seed_${seed}.json" \
    --train-streams 0 --eval-streams 256 --contexts 8 --delay 0 --attempts 5 \
    --rehearsal-groups 1 --seed "$seed" --device cuda
done
