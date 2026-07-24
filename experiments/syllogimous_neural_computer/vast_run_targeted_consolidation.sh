#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
PYTHON=/venv/main/bin/python
OUT=experiments/syllogimous_neural_computer/targeted_consolidation
mkdir -p "$OUT"

for seed in 11 23 37; do
  "$PYTHON" -m experiments.syllogimous_neural_computer.train_consolidation \
    --controller experiments/syllogimous_neural_computer/vast_continual_sharp.pt \
    --checkpoint "$OUT/trained_seed_${seed}.pt" \
    --report "$OUT/trained_seed_${seed}.json" \
    --train-streams 128 --eval-streams 128 --contexts 8 --delay 0 --attempts 4 \
    --seed "$seed" --device cuda
  "$PYTHON" -m experiments.syllogimous_neural_computer.train_consolidation \
    --controller experiments/syllogimous_neural_computer/vast_continual_sharp.pt \
    --checkpoint "$OUT/untrained_seed_${seed}.pt" \
    --report "$OUT/untrained_seed_${seed}.json" \
    --train-streams 0 --eval-streams 128 --contexts 8 --delay 0 --attempts 4 \
    --seed "$seed" --device cuda
done

"$PYTHON" -m experiments.syllogimous_neural_computer.summarize_consolidation \
  --input "$OUT" --output "$OUT/summary.json"
