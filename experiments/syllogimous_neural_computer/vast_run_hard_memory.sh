#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
PY=/venv/main/bin/python
$PY -m pytest -q experiments/syllogimous_neural_computer/test_neural_computer.py
$PY -m experiments.syllogimous_neural_computer.train_lifetime \
  --condition learned_memory --device cuda \
  --hidden 160 --workspace-slots 12 --heads 5 --thought-steps 4 \
  --train-lifetimes 2000 --eval-lifetimes 500 --batch-size 64 --epochs 4 \
  --associations 1 --delay 8 --choices 8 --learning-rate 1e-4 \
  --admission-mode deterministic_hard --admission-threshold 0.01 \
  --write-cost 0.02 --write-cost-threshold 0.70 \
  --initial-checkpoint experiments/syllogimous_neural_computer/vast_learned_memory.pt \
  --checkpoint experiments/syllogimous_neural_computer/vast_hard_memory.pt \
  --report experiments/syllogimous_neural_computer/vast_hard_memory.json
$PY -m experiments.syllogimous_neural_computer.audit_durable \
  --device cuda --samples 200 --associations 1 --delay 8 --choices 8 --threshold 0.01 \
  --checkpoint experiments/syllogimous_neural_computer/vast_hard_memory.pt \
  --blob experiments/syllogimous_neural_computer/vast_durable_memory_blob.pt \
  --report experiments/syllogimous_neural_computer/vast_durable_audit.json
