#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
OUT=experiments/forward_transfer_attention/targeted_universal_candidate
mkdir -p "$OUT"
pids=()
for controller in 11 23 37 51; do
  OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 /venv/main/bin/python \
    -m experiments.forward_transfer_attention.evaluate_cross_controller \
    --controller-checkpoint "experiments/forward_transfer_attention/targeted_decorrelated/seed_${controller}.pt" \
    --consolidator-checkpoint experiments/forward_transfer_attention/targeted_consolidator_replication/seed_23.pt \
    --report "$OUT/controller_${controller}.json" --controller-seed "$controller" \
    --consolidator-seed 23 --eval-lifetimes 1024 --batch-size 64 \
    --query-count 4 --device cuda >"$OUT/controller_${controller}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
