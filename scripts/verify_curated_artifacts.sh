#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
shasum -a 256 -c artifacts/manifests/curated_checkpoints.sha256

