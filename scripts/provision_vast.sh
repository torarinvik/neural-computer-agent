#!/usr/bin/env bash
# Make a freshly rented box useful in one command, so boxes stay disposable.
#   VAST_HOST=1.2.3.4 VAST_PORT=41162 scripts/provision_vast.sh
# Rent by cores, not VRAM: this workload is a ~340k-parameter model whose cost
# is dominated by CPU-side evaluation, so cores per dollar sets throughput.
set -euo pipefail
VAST_HOST="${VAST_HOST:?}"; VAST_PORT="${VAST_PORT:?}"
REPO="${REPO:-/workspace/repos/neural-computer-agent}"
LOCAL_REPO="${LOCAL_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
SSH=(ssh -o ConnectTimeout=25 -o StrictHostKeyChecking=accept-new -p "$VAST_PORT" "root@$VAST_HOST")

echo "waiting for sshd"
for i in $(seq 1 40); do
  "${SSH[@]}" 'echo up' >/dev/null 2>&1 && break
  sleep 15
done
"${SSH[@]}" "mkdir -p $REPO"
echo "syncing repo"
rsync -az --exclude '.venv/' --exclude '__pycache__/' --exclude '.pytest_cache/' \
  -e "ssh -p $VAST_PORT" "$LOCAL_REPO/" "root@$VAST_HOST:$REPO/"
echo "installing deps and verifying"
"${SSH[@]}" "cd $REPO && source /venv/main/bin/activate && \
  uv pip install -q 'numpy>=2.0' 'Pillow>=11' 'pytest>=8' && \
  python -c 'import torch;print(\"torch\",torch.__version__,torch.cuda.is_available())' && \
  nproc && python -m pytest -q -p no:randomly 2>&1 | tail -2"
echo "ready: VAST_HOST=$VAST_HOST VAST_PORT=$VAST_PORT"
