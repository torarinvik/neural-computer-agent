#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
OUT=experiments/forward_transfer_attention/targeted_temporal_no_watermark_causal_audit
rm -rf "$OUT"
mkdir -p "$OUT"
pids=()
for condition in intact empty shuffled garbage; do
  OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 /venv/main/bin/python \
    -m experiments.forward_transfer_attention.evaluate_cross_primitive \
    --controller-checkpoint experiments/forward_transfer_attention/targeted_joint_temporal_no_watermark/seed_23.pt \
    --report "$OUT/${condition}.json" --eval-lifetimes 256 --batch-size 64 \
    --query-count 4 --seed 23 --primitive temporal --condition "$condition" --device cuda \
    >"$OUT/${condition}.log" 2>&1 &
  pids+=("$!")
done
OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 /venv/main/bin/python \
  -m experiments.forward_transfer_attention.evaluate_cross_primitive \
  --controller-checkpoint experiments/forward_transfer_attention/targeted_joint_temporal_no_watermark/seed_23.pt \
  --report "$OUT/reversed.json" --eval-lifetimes 256 --batch-size 64 \
  --query-count 4 --seed 23 --primitive temporal --condition intact \
  --reverse-temporal-query --device cuda >"$OUT/reversed.log" 2>&1 &
pids+=("$!")
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
