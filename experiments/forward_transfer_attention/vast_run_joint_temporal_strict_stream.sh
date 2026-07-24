#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
OUT=experiments/forward_transfer_attention/targeted_joint_temporal_strict_stream
rm -rf "$OUT"
mkdir -p "$OUT"
OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 /venv/main/bin/python \
  -m experiments.forward_transfer_attention.train_joint_adapter \
  --controller-checkpoint experiments/forward_transfer_attention/targeted_decorrelated/seed_23.pt \
  --consolidator-checkpoint experiments/forward_transfer_attention/targeted_consolidator_replication/seed_23.pt \
  --checkpoint "$OUT/seed_23.pt" --report "$OUT/seed_23.json" \
  --train-lifetimes 256 --eval-lifetimes 256 --batch-size 64 --epochs 6 \
  --query-count 4 --controller-learning-rate 2e-5 \
  --consolidator-learning-rate 1e-4 --seed 23 --device cuda \
  >"$OUT/seed_23.log" 2>&1
