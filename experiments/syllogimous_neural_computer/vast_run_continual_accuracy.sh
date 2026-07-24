#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
PY=/venv/main/bin/python
$PY -m experiments.syllogimous_neural_computer.train_continual \
  --device cuda --train-streams 384 --eval-streams 128 --batch-size 8 \
  --train-contexts 4 --eval-contexts 8 --delay 2 --choices 8 --epochs 8 \
  --learning-rate 1e-4 --threshold 0.01 --read-top-k 8 --write-cost 0.0 \
  --initial-checkpoint experiments/syllogimous_neural_computer/vast_hard_memory.pt \
  --checkpoint experiments/syllogimous_neural_computer/vast_continual_accuracy.pt \
  --report experiments/syllogimous_neural_computer/vast_continual_accuracy.json
