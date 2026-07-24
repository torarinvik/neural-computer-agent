#!/bin/sh
# setup_aud.sh — provision the sax venv (.venv-aud) for MiDashengLM-0.6B. Separate from .venv-vlm
# because the model's remote code targets transformers 4.57 (the violin's Qwen2.5-VL uses 5.13, and
# 5.x makes MiDashengLM's generate emit garbage). uv-managed, like setup_vlm.sh.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
uv venv .venv-aud --python 3.12
VIRTUAL_ENV="$HERE/.venv-aud" uv pip install "transformers==4.57.1" torch torchaudio librosa soundfile accelerate numpy
echo "setup_aud: verifying..."
"$HERE/.venv-aud/bin/python" -c "import transformers,torch,torchaudio,librosa; print('ok tfm',transformers.__version__,'mps',torch.backends.mps.is_available())"
