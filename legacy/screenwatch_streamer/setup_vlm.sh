#!/bin/sh
# setup_vlm.sh — provision the local-VLM member (screenvlm / "violin").
#
# Creates an isolated, pinned Python venv (.venv-vlm) for Qwen2.5-VL-3B-Instruct and verifies the
# import + device. The versions below are the EXACT set measured working on this machine in the
# VJEPA2 session (torch 2.13 / transformers 5.13 / MPS) — do not float them; transformers+torch
# drift silently breaks the Qwen2.5-VL video path (see cursor_vlm_plan.md §7).
#
# The Hugging Face model cache is shared with the VJEPA2 project (same model id ⇒ no re-download;
# ~7 GB already present under ~/.cache/huggingface/hub). Only the venv is repo-local & gitignored.
#
#   ./setup_vlm.sh            provision .venv-vlm and verify
#   ./setup_vlm.sh --verify   verify only (no install)
set -e

HERE=$(cd "$(dirname "$0")" && pwd)
VENV="$HERE/.venv-vlm"
PY="$VENV/bin/python"
MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"

verify() {
    echo "verify: importing torch/transformers, checking device + model cache..."
    "$PY" - "$MODEL_ID" <<'PYEOF'
import sys, os, glob
import torch, transformers
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"  torch {torch.__version__}  transformers {transformers.__version__}  device={dev}")
mid = sys.argv[1]
cache = os.path.expanduser("~/.cache/huggingface/hub")
hits = glob.glob(os.path.join(cache, "models--" + mid.replace("/", "--")))
print(f"  model cache: {'present' if hits else 'ABSENT (first run will download ~7GB)'}")
from transformers import AutoProcessor, AutoModelForImageTextToText  # noqa: F401
print("  import OK")
PYEOF
}

if [ "$1" = "--verify" ]; then
    [ -x "$PY" ] || { echo "no venv at $VENV — run ./setup_vlm.sh first" >&2; exit 1; }
    verify
    exit 0
fi

command -v uv >/dev/null 2>&1 || { echo "uv not found (expected ~/.local/bin/uv)" >&2; exit 1; }

echo "creating venv: $VENV (python 3.12)"
uv venv "$VENV" --python 3.12

echo "installing pinned deps (proven set; shared wheel cache)..."
uv pip install --python "$PY" \
    torch==2.13.0 \
    torchvision==0.28.0 \
    transformers==5.13.1 \
    accelerate==1.14.0 \
    qwen-vl-utils==0.0.14 \
    pillow==12.3.0 \
    numpy==2.5.1

verify
echo "done. run:  ./screenvlm <batch_dir> --span t0,t1"
