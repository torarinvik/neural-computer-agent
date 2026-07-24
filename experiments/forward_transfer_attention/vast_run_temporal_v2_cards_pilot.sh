#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
OUT=experiments/forward_transfer_attention/targeted_temporal_v2_cards_pilot
mkdir -p "$OUT"
pids=()
for seed in 11 23 37 51; do
  OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 /venv/main/bin/python \
    -m experiments.forward_transfer_attention.evaluate_cross_primitive \
    --controller-checkpoint "experiments/forward_transfer_attention/targeted_decorrelated/seed_${seed}.pt" \
    --consolidator-checkpoint experiments/forward_transfer_attention/targeted_consolidator_replication/seed_23.pt \
    --report "$OUT/seed_${seed}.json" --eval-lifetimes 256 --batch-size 64 \
    --query-count 4 --seed "$seed" --primitive temporal --device cuda \
    >"$OUT/seed_${seed}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
