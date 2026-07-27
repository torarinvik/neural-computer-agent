#!/bin/bash
# Chain rungs one at a time on rectified slots, each on the previous promotion.
# Resumable: a rung whose checkpoint already exists is skipped.
set -u
cd /workspace/repos/neural-computer-agent
source /venv/main/bin/activate
export OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0
OUT=session_records/loose_ends_2026-07-27/depth/raw
CK=artifacts/checkpoints/depth
mkdir -p "$OUT" "$CK"
SEED=${1:-8600}
PARENT=artifacts/checkpoints/unified_three_skill_compounding_seed8413.pt
REPLAY="binary_mapping,visible_context,visible_context_xor"
RSUP="1,1,1"
RUNGS=(
  "4 contextual_composition 2"
  "5 context_rule_xor 2"
  "6 contextual_override 2"
  "7 contextual_mapping 2"
)
for entry in "${RUNGS[@]}"; do
  set -- $entry
  RUNG=$1; TASK=$2; SUP=$3
  REPORT="$OUT/depth_rung${RUNG}_${SEED}.json"
  NEXT="$CK/depth_rung${RUNG}_${SEED}.pt"
  if [ -f "$NEXT" ]; then
    echo "=== rung $RUNG : $TASK already promoted, skipping"
  else
    echo "=== rung $RUNG : $TASK (support $SUP) on $(basename "$PARENT")"
    echo "    replay: $REPLAY  support: $RSUP"
    python -m experiments.unified_cognitive_controller.train_fourth_primitive_transfer \
      --parent "$PARENT" --report "$REPORT" --checkpoint-out "$NEXT" \
      --device cuda --seed "$SEED" --steps 6144 \
      --new-task "$TASK" --new-support-trials "$SUP" \
      --replay-tasks "$REPLAY" --replay-support-trials "$RSUP" \
      --slot-gate-mode relu --gate-leak-initial 0.02 \
      --retention-weight 2.0 --replay-batch-size 32 --learning-rate 0.01 \
      --test-lifetimes 1024 || { echo "rung $RUNG ERRORED"; exit 1; }
    if [ ! -f "$NEXT" ]; then
      echo "    rung $RUNG did not pass its own gate; chain stops here"
      exit 2
    fi
  fi
  PARENT="$NEXT"
  REPLAY="$REPLAY,$TASK"
  RSUP="$RSUP,$SUP"
done
echo "DEPTH_CHAIN_DONE at depth $(( ${#RUNGS[@]} + 3 ))"
