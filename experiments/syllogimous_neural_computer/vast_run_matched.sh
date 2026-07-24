#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
PY=/venv/main/bin/python
$PY -m pytest -q experiments/syllogimous_neural_computer/test_neural_computer.py
for condition in no_memory random_write learned_memory; do
  $PY -m experiments.syllogimous_neural_computer.train_lifetime \
    --condition "$condition" --device cuda \
    --hidden 160 --workspace-slots 12 --heads 5 --thought-steps 4 \
    --train-lifetimes 2000 --eval-lifetimes 500 --batch-size 64 --epochs 6 \
    --associations 1 --delay 8 --choices 8 --learning-rate 3e-4 \
    --checkpoint "experiments/syllogimous_neural_computer/vast_${condition}.pt" \
    --report "experiments/syllogimous_neural_computer/vast_${condition}.json"
done
