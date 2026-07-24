#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
OUT=experiments/forward_transfer_attention/targeted_consolidator_pilot
mkdir -p "$OUT"
OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 /venv/main/bin/python \
  -m experiments.forward_transfer_attention.train_consolidator \
  --controller-checkpoint experiments/forward_transfer_attention/targeted_decorrelated/seed_11.pt \
  --checkpoint "$OUT/seed_11.pt" --report "$OUT/seed_11.json" \
  --train-lifetimes 1024 --eval-lifetimes 512 --batch-size 64 \
  --epochs 12 --query-count 4 --learning-rate 3e-4 --seed 11 --device cuda \
  >"$OUT/seed_11.log" 2>&1
