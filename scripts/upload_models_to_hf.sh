#!/usr/bin/env bash
set -euo pipefail

repo_id="${1:-torarin87/neural-computer-agent}"
visibility="${2:-private}"

if ! hf auth whoami >/dev/null 2>&1; then
  echo "Run 'hf auth login' first." >&2
  exit 1
fi

private_flag=()
if [[ "$visibility" == "private" ]]; then
  private_flag=(--private)
elif [[ "$visibility" != "public" ]]; then
  echo "Visibility must be 'private' or 'public'." >&2
  exit 2
fi

hf repos create "$repo_id" --type model --exist-ok "${private_flag[@]}"
hf upload "$repo_id" artifacts/MODEL_CARD.md README.md \
  --commit-message "Add audited neural computer model card"
hf upload "$repo_id" artifacts/checkpoints checkpoints \
  --include "*.pt" \
  --commit-message "Upload four audited research checkpoints"
hf upload "$repo_id" artifacts/manifests/curated_checkpoints.sha256 \
  checkpoints/sha256sums.txt \
  --commit-message "Add checkpoint integrity manifest"

echo "Uploaded https://huggingface.co/${repo_id}"

