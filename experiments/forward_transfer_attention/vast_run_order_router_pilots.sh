#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
ROOT=experiments/forward_transfer_attention/targeted_order_router_pilots
rm -rf "$ROOT"
mkdir -p "$ROOT"
pids=()
run_pilot() {
  local name=$1 epochs=$2
  shift 2
  local out="$ROOT/$name"
  mkdir -p "$out"
  OMP_NUM_THREADS=48 MKL_NUM_THREADS=48 /venv/main/bin/python \
    -m experiments.forward_transfer_attention.train_joint_adapter \
    --controller-checkpoint experiments/forward_transfer_attention/targeted_decorrelated/seed_23.pt \
    --consolidator-checkpoint experiments/forward_transfer_attention/targeted_consolidator_replication/seed_23.pt \
    --checkpoint "$out/seed_23.pt" --report "$out/seed_23.json" \
    --train-lifetimes 256 --eval-lifetimes 256 --batch-size 64 --epochs "$epochs" \
    --query-count 4 --controller-learning-rate 3e-4 --seed 23 --device cuda \
    --order-routing --router-only "$@" >"$out/seed_23.log" 2>&1 &
  pids+=("$!")
}
run_pilot direct 8
run_pilot staged 10 --last-epochs 2 --first-epochs 2 --grounding-epochs 2
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
