#!/bin/bash
set -euo pipefail
/venv/main/bin/pip install torch numpy pillow pytest
cd /root/elisa-screenwatch
/venv/main/bin/python -m pytest experiments/forward_transfer_attention -q
