#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-.venv/bin/python}"
mode="${1:-fast}"
if [ "$#" -gt 0 ]; then
  shift
fi
case "$mode" in
  fast)
    exec "$python_bin" -m pytest -q -m "not campaign" "$@"
    ;;
  campaign)
    exec "$python_bin" -m pytest -q -m campaign "$@"
    ;;
  all)
    exec "$python_bin" -m pytest -q "$@"
    ;;
  *)
    echo "usage: $0 [fast|campaign|all]" >&2
    exit 2
    ;;
esac
