#!/bin/bash
# Recreate the remote sweep lab on a fresh instance (Vast.ai or any Linux box).
#
# Usage, from the repo root on the LOCAL machine:
#   scripts/remote_lab_setup.sh <ssh-port> <user@host> [path-to-experiment-scripts]
#
# What it does: packages tracked code (minus session_records; plus the two
# artifact files the tests need), ships it, installs deps into whatever
# python3 is active remotely (CPU torch if none present), runs the full
# test suite, and prints the PYTHONPATH incantation runs need.
#
# The remote layout it creates:
#   /root/nc          the repo (tracked files)
#   /root/nc-scratch  experiment scripts (cotrained.py etc.), if a local
#                     directory was given as the third argument
#   /root/nc-results  empty, for run outputs
#
# Sweep pattern that works there (per-seed background processes; CPU cores
# are the resource that matters for the tiny plants — a GPU does not help
# at hidden=32/event=64, see the 2026-08-08 discussion):
#   for seed in 69316 ... ; do
#     OMP_NUM_THREADS=2 PYTHONPATH=/root/nc:/root/nc/src \
#       python /root/nc-scratch/cotrained.py --seed $seed ... \
#       > /root/nc-results/base-$seed.json 2> .../base-$seed.err &
#   done; wait
#
# Gotchas learned the expensive way:
#   - PYTHONPATH needs BOTH /root/nc and /root/nc/src (locally `uv run`
#     supplies src/ via the editable install).
#   - Launch long jobs with `nohup ... < /dev/null & disown` inside the
#     ssh command, or the ssh session holds them hostage.
#   - Never `pkill -f <pattern>` from a shell whose own command line
#     contains <pattern>.
#   - JSON outputs are empty until a run finishes (shell redirect creates
#     them at launch); check process count, not file existence.

set -euo pipefail

PORT="${1:?ssh port}"
HOST="${2:?user@host}"
SCRATCH_DIR="${3:-}"

SSH=(ssh -p "$PORT" -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$HOST")

git archive --format=tar.gz HEAD -o /tmp/nc-code.tar.gz \
  ":(exclude)session_records" ":(exclude)artifacts"
tar -czf /tmp/nc-art.tar.gz \
  artifacts/memory/span_multi_skill_bank_seed49011 \
  artifacts/checkpoints/span8_addressed_parent_scale1_seed32001.pt

scp -P "$PORT" -o BatchMode=yes /tmp/nc-code.tar.gz /tmp/nc-art.tar.gz "$HOST":/root/

if [ -n "$SCRATCH_DIR" ]; then
  tar -czf /tmp/nc-scratch.tar.gz -C "$SCRATCH_DIR" .
  scp -P "$PORT" -o BatchMode=yes /tmp/nc-scratch.tar.gz "$HOST":/root/
fi

"${SSH[@]}" 'set -e
mkdir -p /root/nc /root/nc-scratch /root/nc-results
tar -xzf /root/nc-code.tar.gz -C /root/nc
tar -xzf /root/nc-art.tar.gz -C /root/nc
[ -f /root/nc-scratch.tar.gz ] && tar -xzf /root/nc-scratch.tar.gz -C /root/nc-scratch
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
python3 -c "import torch" 2>/dev/null || \
  pip install -q "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu
pip install -q "numpy>=2.0" "Pillow>=11" "pytest>=8"
cd /root/nc && OMP_NUM_THREADS=4 python3 -m pytest tests/ -q 2>&1 | tail -1'

echo
echo "Remote lab ready. Run experiments with:"
echo "  PYTHONPATH=/root/nc:/root/nc/src python3 <script> ..."
