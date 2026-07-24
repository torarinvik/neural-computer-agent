#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" -m experiments.syllogimous_bitter_lesson.train_rl \
  --device cuda --scale 2m --memory-core event_transformer \
  --thought-dynamics gated_residual --action-count 8 --thought-steps 8 \
  --random-thought-depth --consistency-weight 0.25 \
  --train-samples 4000 --eval-samples 1000 --batch-size 64 --epochs 3 --workers 4 \
  --learning-rate 3e-5 --train-premises 2 --eval-premises 2 \
  --logic-modulus 2 --learning-signal verifier \
  --training-tasks cyclic --evaluation-tasks cyclic \
  --initial-checkpoint experiments/syllogimous_bitter_lesson/small_mixed_75_25_block2.pt \
  --checkpoint experiments/syllogimous_bitter_lesson/cyclic_mod2_learnability.pt \
  --report experiments/syllogimous_bitter_lesson/cyclic_mod2_learnability.json
