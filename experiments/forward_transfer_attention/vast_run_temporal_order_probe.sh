#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
OUT=experiments/forward_transfer_attention/targeted_temporal_order_probe
rm -rf "$OUT"
mkdir -p "$OUT"
pids=()
for spec in \
  original:experiments/forward_transfer_attention/targeted_decorrelated/seed_23.pt \
  adapted:experiments/forward_transfer_attention/targeted_temporal_last_curriculum/seed_23.pt; do
  name=${spec%%:*}
  checkpoint=${spec#*:}
  OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 /venv/main/bin/python \
    -m experiments.forward_transfer_attention.probe_temporal_order \
    --checkpoint "$checkpoint" --report "$OUT/${name}.json" \
    --train-lifetimes 2048 --test-lifetimes 1024 --batch-size 256 \
    --seed 23 --device cuda >"$OUT/${name}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
