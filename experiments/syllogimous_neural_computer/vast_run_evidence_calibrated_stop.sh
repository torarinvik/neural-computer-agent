#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
PYTHON=/venv/main/bin/python
SOURCE=experiments/syllogimous_neural_computer/targeted_trajectory_stop_replication
OUT=experiments/syllogimous_neural_computer/targeted_evidence_calibrated_stop
mkdir -p "$OUT"
for seed in 11 23 37; do
  "$PYTHON" -m experiments.syllogimous_neural_computer.train_consolidation \
    --controller experiments/syllogimous_neural_computer/vast_continual_sharp.pt \
    --initial-policy "$SOURCE/autonomous_seed_${seed}.pt" \
    --checkpoint "$OUT/autonomous_seed_${seed}.pt" \
    --report "$OUT/autonomous_seed_${seed}.json" \
    --train-streams 0 --calibration-streams 64 --eval-streams 256 \
    --contexts 8 --delay 0 --attempts 5 --rehearsal-groups 1 \
    --autonomous-stop --trajectory-stop --seed "$seed" --device cuda
done
