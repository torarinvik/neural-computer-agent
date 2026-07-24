#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
OUT=experiments/forward_transfer_attention/targeted_renderer_sanity
rm -rf "$OUT"
mkdir -p "$OUT"
pids=()
for primitive in spatial shape temporal; do
  OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 /venv/main/bin/python \
    -m experiments.forward_transfer_attention.evaluate_cross_primitive \
    --controller-checkpoint experiments/forward_transfer_attention/targeted_decorrelated/seed_23.pt \
    --consolidator-checkpoint experiments/forward_transfer_attention/targeted_consolidator_replication/seed_23.pt \
    --report "$OUT/${primitive}.json" --eval-lifetimes 64 --batch-size 64 \
    --query-count 4 --seed 23 --primitive "$primitive" --device cuda \
    >"$OUT/${primitive}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
