#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
manifest="artifacts/manifests/curated_checkpoints.sha256"

if [[ ! -s "$manifest" ]]; then
  echo "No curated checkpoints are currently registered."
  exit 0
fi

if ! awk '
  NF != 2 || length($1) != 64 || $1 !~ /^[0-9a-fA-F]+$/ {
    print "invalid checksum manifest line " NR > "/dev/stderr"
    invalid = 1
  }
  END { exit invalid }
' "$manifest"; then
  exit 1
fi
shasum -a 256 -c "$manifest"
