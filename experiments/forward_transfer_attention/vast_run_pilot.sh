#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
OUT=experiments/forward_transfer_attention/targeted_pilot
mkdir -p "$OUT"
OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 /venv/main/bin/python \
  -m experiments.forward_transfer_attention.train \
  --initial-checkpoint experiments/syllogimous_neural_computer/vast_hard_memory.pt \
  --checkpoint "$OUT/seed_17.pt" --report "$OUT/seed_17.json" \
  --train-lifetimes 1024 --eval-lifetimes 256 --batch-size 64 \
  --epochs 10 --primitive-epochs 3 --read-top-k 1 --query-count 4 \
  --learning-rate 0.0001 --seed 17 --device cuda
