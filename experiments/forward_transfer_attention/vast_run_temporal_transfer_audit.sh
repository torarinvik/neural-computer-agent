#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
OUT=experiments/forward_transfer_attention/targeted_temporal_transfer_audit
ADAPTER=experiments/forward_transfer_attention/targeted_temporal_adaptation_pilot/seed_23.pt
mkdir -p "$OUT"
pids=()
for seed in 11 23 37 51; do
  OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 /venv/main/bin/python \
    -m experiments.forward_transfer_attention.evaluate_cross_controller \
    --controller-checkpoint "experiments/forward_transfer_attention/targeted_decorrelated/seed_${seed}.pt" \
    --consolidator-checkpoint "$ADAPTER" --report "$OUT/seed_${seed}_spatial.json" \
    --controller-seed "$seed" --consolidator-seed 23 --eval-lifetimes 256 \
    --batch-size 64 --query-count 4 --device cuda >"$OUT/seed_${seed}_spatial.log" 2>&1 &
  pids+=("$!")
  for primitive in shape temporal; do
    OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 /venv/main/bin/python \
      -m experiments.forward_transfer_attention.evaluate_cross_primitive \
      --controller-checkpoint "experiments/forward_transfer_attention/targeted_decorrelated/seed_${seed}.pt" \
      --consolidator-checkpoint "$ADAPTER" --report "$OUT/seed_${seed}_${primitive}.json" \
      --eval-lifetimes 256 --batch-size 64 --query-count 4 --seed "$seed" \
      --primitive "$primitive" --device cuda >"$OUT/seed_${seed}_${primitive}.log" 2>&1 &
    pids+=("$!")
  done
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
