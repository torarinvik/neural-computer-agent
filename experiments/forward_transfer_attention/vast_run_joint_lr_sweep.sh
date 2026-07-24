#!/bin/bash
set -euo pipefail
cd /root/elisa-screenwatch
ROOT=experiments/forward_transfer_attention/targeted_joint_lr_sweep
rm -rf "$ROOT"
mkdir -p "$ROOT"
pids=()
for spec in low:5e-5 medium:1e-4 high:2e-4; do
  name=${spec%%:*}
  lr=${spec#*:}
  out="$ROOT/$name"
  mkdir -p "$out"
  OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 /venv/main/bin/python \
    -m experiments.forward_transfer_attention.train_joint_adapter \
    --controller-checkpoint experiments/forward_transfer_attention/targeted_decorrelated/seed_23.pt \
    --consolidator-checkpoint experiments/forward_transfer_attention/targeted_consolidator_replication/seed_23.pt \
    --checkpoint "$out/seed_23.pt" --report "$out/seed_23.json" \
    --train-lifetimes 256 --eval-lifetimes 128 --batch-size 64 --epochs 4 \
    --query-count 4 --controller-learning-rate "$lr" \
    --consolidator-learning-rate 1e-4 --seed 23 --device cuda \
    >"$out/seed_23.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
