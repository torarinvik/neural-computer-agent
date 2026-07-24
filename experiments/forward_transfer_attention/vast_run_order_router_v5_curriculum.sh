#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
OUT=experiments/forward_transfer_attention/targeted_order_router_v5_curriculum
rm -rf "$OUT"
mkdir -p "$OUT"
OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 /venv/main/bin/python \
  -m experiments.forward_transfer_attention.train_joint_adapter \
  --controller-checkpoint experiments/forward_transfer_attention/targeted_decorrelated/seed_23.pt \
  --consolidator-checkpoint experiments/forward_transfer_attention/targeted_consolidator_replication/seed_23.pt \
  --checkpoint "$OUT/seed_23.pt" --report "$OUT/seed_23.json" \
  --train-lifetimes 256 --eval-lifetimes 512 --batch-size 64 --epochs 20 \
  --last-epochs 4 --first-epochs 4 --grounding-epochs 4 --query-count 4 \
  --controller-learning-rate 3e-4 --temporal-old-weight 0.5 \
  --temporal-future-weight 4.0 --seed 23 --device cuda \
  --order-routing --router-only >"$OUT/seed_23.log" 2>&1
