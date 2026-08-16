#!/usr/bin/env bash
set -euo pipefail

exec "$(dirname "$0")/test_canonical.sh" campaign "$@"
