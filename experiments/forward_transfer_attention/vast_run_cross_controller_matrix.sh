#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
OUT=experiments/forward_transfer_attention/targeted_cross_controller_matrix
mkdir -p "$OUT"
pids=()
for controller in 11 23 37 51; do
  for consolidator in 11 23 37 51; do
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /venv/main/bin/python \
      -m experiments.forward_transfer_attention.evaluate_cross_controller \
      --controller-checkpoint "experiments/forward_transfer_attention/targeted_decorrelated/seed_${controller}.pt" \
      --consolidator-checkpoint "experiments/forward_transfer_attention/targeted_consolidator_replication/seed_${consolidator}.pt" \
      --report "$OUT/controller_${controller}_consolidator_${consolidator}.json" \
      --controller-seed "$controller" --consolidator-seed "$consolidator" \
      --eval-lifetimes 256 --batch-size 64 --query-count 4 --device cuda \
      >"$OUT/controller_${controller}_consolidator_${consolidator}.log" 2>&1 &
    pids+=("$!")
  done
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
