#!/bin/bash
set -euo pipefail
uv pip install --python /venv/main/bin/python torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /venv/main/bin/python numpy pillow pytest
/venv/main/bin/python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())
print(torch.randn(1, device="cuda"))
PY
