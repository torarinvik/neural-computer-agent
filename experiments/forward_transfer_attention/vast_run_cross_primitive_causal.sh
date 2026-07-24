#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
OUT=experiments/forward_transfer_attention/targeted_cross_primitive_causal
mkdir -p "$OUT"
pids=()
for seed in 11 23 37 51; do
  for condition in intact empty shuffled garbage; do
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /venv/main/bin/python \
      -m experiments.forward_transfer_attention.evaluate_cross_primitive \
      --controller-checkpoint "experiments/forward_transfer_attention/targeted_decorrelated/seed_${seed}.pt" \
      --consolidator-checkpoint experiments/forward_transfer_attention/targeted_consolidator_replication/seed_23.pt \
      --report "$OUT/seed_${seed}_${condition}.json" --eval-lifetimes 256 --batch-size 64 \
      --query-count 4 --seed "$seed" --condition "$condition" --device cuda \
      >"$OUT/seed_${seed}_${condition}.log" 2>&1 &
    pids+=("$!")
  done
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
