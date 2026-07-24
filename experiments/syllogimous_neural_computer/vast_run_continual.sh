#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
PY=/venv/main/bin/python
$PY -m pytest -q experiments/syllogimous_neural_computer/test_neural_computer.py
$PY -m experiments.syllogimous_neural_computer.train_continual \
  --device cuda --train-streams 256 --eval-streams 128 --batch-size 8 \
  --train-contexts 4 --eval-contexts 8 --delay 2 --choices 8 --epochs 6 \
  --learning-rate 1e-4 --threshold 0.05 --write-cost 0.0 \
  --initial-checkpoint experiments/syllogimous_neural_computer/vast_compressed_memory.pt \
  --checkpoint experiments/syllogimous_neural_computer/vast_continual_memory.pt \
  --report experiments/syllogimous_neural_computer/vast_continual_memory.json
