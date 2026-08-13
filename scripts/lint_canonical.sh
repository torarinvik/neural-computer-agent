#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ruff_bin="${RUFF_BIN:-ruff}"
exec "$ruff_bin" check src tests experiments/brainworkshop_canonical \
  experiments/recipe_expressibility
