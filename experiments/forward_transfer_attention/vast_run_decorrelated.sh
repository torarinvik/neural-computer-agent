#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
OUT=experiments/forward_transfer_attention/targeted_decorrelated
mkdir -p "$OUT"
pids=()
for seed in 11 23 37 51; do
  OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 /venv/main/bin/python \
    -m experiments.forward_transfer_attention.train \
    --initial-checkpoint "experiments/forward_transfer_attention/targeted_early_transfer/seed_${seed}.pt" \
    --checkpoint "$OUT/seed_${seed}.pt" --report "$OUT/seed_${seed}.json" \
    --train-lifetimes 1536 --eval-lifetimes 1024 --batch-size 64 \
    --epochs 8 --primitive-epochs 0 --read-top-k 1 --query-count 4 \
    --learning-rate 0.00005 --advantage-weight 0.5 --advantage-margin 0.5 \
    --seed "$seed" --device cuda >"$OUT/seed_${seed}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
