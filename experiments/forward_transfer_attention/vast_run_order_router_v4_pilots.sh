#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
ROOT=experiments/forward_transfer_attention/targeted_order_router_v4_pilots
rm -rf "$ROOT"
mkdir -p "$ROOT"
pids=()
for spec in route_only:1e-3:true route_and_head:3e-4:false; do
  IFS=: read -r name lr freeze <<<"$spec"
  out="$ROOT/$name"
  mkdir -p "$out"
  extra=()
  if [[ "$freeze" == true ]]; then extra+=(--freeze-answer-head); fi
  OMP_NUM_THREADS=48 MKL_NUM_THREADS=48 /venv/main/bin/python \
    -m experiments.forward_transfer_attention.train_joint_adapter \
    --controller-checkpoint experiments/forward_transfer_attention/targeted_decorrelated/seed_23.pt \
    --consolidator-checkpoint experiments/forward_transfer_attention/targeted_consolidator_replication/seed_23.pt \
    --checkpoint "$out/seed_23.pt" --report "$out/seed_23.json" \
    --train-lifetimes 256 --eval-lifetimes 256 --batch-size 64 --epochs 12 \
    --query-count 4 --controller-learning-rate "$lr" --seed 23 --device cuda \
    --temporal-old-weight 0.5 --temporal-future-weight 4.0 \
    --order-routing --router-only "${extra[@]}" >"$out/seed_23.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
