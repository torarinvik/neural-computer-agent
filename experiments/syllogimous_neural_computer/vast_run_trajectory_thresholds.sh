#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
PYTHON=/venv/main/bin/python
POLICY=experiments/syllogimous_neural_computer/targeted_trajectory_stop_sweep/weight_1.pt
OUT=experiments/syllogimous_neural_computer/targeted_trajectory_thresholds
mkdir -p "$OUT"
for threshold in 0.55 0.60 0.65 0.70 0.75; do
  name=${threshold/./p}
  "$PYTHON" -m experiments.syllogimous_neural_computer.train_consolidation \
    --controller experiments/syllogimous_neural_computer/vast_continual_sharp.pt \
    --initial-policy "$POLICY" --checkpoint "$OUT/threshold_${name}.pt" \
    --report "$OUT/threshold_${name}.json" \
    --train-streams 0 --eval-streams 256 --contexts 8 --delay 0 --attempts 5 \
    --rehearsal-groups 1 --autonomous-stop --trajectory-stop \
    --stop-threshold "$threshold" --seed 11 --device cuda
done
