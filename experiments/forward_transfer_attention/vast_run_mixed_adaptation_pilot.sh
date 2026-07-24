#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
OUT=experiments/forward_transfer_attention/targeted_mixed_adaptation_pilot
mkdir -p "$OUT"
OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 /venv/main/bin/python \
  -m experiments.forward_transfer_attention.train_consolidator \
  --controller-checkpoint experiments/forward_transfer_attention/targeted_decorrelated/seed_23.pt \
  --initial-consolidator-checkpoint experiments/forward_transfer_attention/targeted_consolidator_replication/seed_23.pt \
  --checkpoint "$OUT/seed_23.pt" --report "$OUT/seed_23.json" \
  --train-lifetimes 384 --eval-lifetimes 256 --batch-size 64 --epochs 3 \
  --query-count 4 --learning-rate 1e-4 --primitive mixed --seed 23 --device cuda \
  >"$OUT/seed_23.log" 2>&1
