#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" -m experiments.syllogimous_bitter_lesson.train_rl \
  --device cuda --scale 2m --memory-core event_transformer \
  --thought-dynamics gated_residual --action-count 8 --thought-steps 8 \
  --random-thought-depth --consistency-weight 0.25 \
  --train-samples 4000 --eval-samples 2800 --batch-size 64 --epochs 2 --workers 4 \
  --learning-rate 5e-6 --train-premises 2,4,8 --eval-premises 2,4,8 \
  --choice-counts 2,3,4,5,6,7,8 --choice-distractors 20 --choice-delay-frames 4 \
  --choice-audio-distractors 8 --choice-target-like-distractors 6 \
  --choice-temporal-distractors 3 --reaction-fraction 0.25 \
  --reasoning-family mixed --cyclic-fraction 0.25 --logic-modulus 4 \
  --learning-signal verifier --training-tasks mixed_cognitive \
  --evaluation-tasks mixed_cognitive \
  --initial-checkpoint experiments/syllogimous_bitter_lesson/small_mixed_75_25_block2.pt \
  --checkpoint experiments/syllogimous_bitter_lesson/small_mixed_mod4_small.pt \
  --report experiments/syllogimous_bitter_lesson/small_mixed_mod4_small.json
