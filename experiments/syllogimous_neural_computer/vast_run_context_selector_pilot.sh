#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
PYTHON=/venv/main/bin/python
OUT=experiments/syllogimous_neural_computer/targeted_context_selector_pilot
mkdir -p "$OUT"
for streams in 64 256; do
  "$PYTHON" -m experiments.syllogimous_neural_computer.train_context_selector \
    --controller experiments/syllogimous_neural_computer/vast_continual_sharp.pt \
    --checkpoint "$OUT/train_${streams}.pt" --report "$OUT/train_${streams}.json" \
    --train-streams "$streams" --eval-streams 128 --contexts 8 --delay 0 \
    --context-cost 0.001 --seed 11 --device cuda
done
