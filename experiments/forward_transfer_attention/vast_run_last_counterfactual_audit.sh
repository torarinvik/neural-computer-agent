#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
OUT=experiments/forward_transfer_attention/targeted_last_counterfactual_audit
rm -rf "$OUT"
mkdir -p "$OUT"
pids=()
for mode in intact reversed counterfactual; do
  extra=()
  if [[ "$mode" != intact ]]; then extra+=(--reverse-temporal-query); fi
  if [[ "$mode" == counterfactual ]]; then extra+=(--counterfactual-temporal-labels); fi
  OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 /venv/main/bin/python \
    -m experiments.forward_transfer_attention.evaluate_cross_primitive \
    --controller-checkpoint experiments/forward_transfer_attention/targeted_temporal_last_curriculum/seed_23.pt \
    --report "$OUT/${mode}.json" --eval-lifetimes 512 --batch-size 64 \
    --query-count 4 --seed 23 --primitive temporal --temporal-stage last \
    --condition intact --device cuda "${extra[@]}" >"$OUT/${mode}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
