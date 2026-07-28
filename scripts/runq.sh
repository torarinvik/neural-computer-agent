#!/usr/bin/env bash
# Bounded-concurrency job runner. One line of the job file is one command.
# The concurrency gate uses a fifo rather than `jobs`, which does not work
# inside a command substitution and silently launched every job at once.
set -u
cd "$(dirname "$0")/.." 2>/dev/null || cd /workspace/repos/neural-computer-agent
source /venv/main/bin/activate 2>/dev/null
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
JOBS="${1:?job file}"; P="${2:-6}"
fifo=$(mktemp -u); mkfifo "$fifo"; exec 3<>"$fifo"; rm -f "$fifo"
for _ in $(seq "$P"); do printf '.' >&3; done
n=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  read -r -n1 -u3 _
  n=$((n+1))
  ( eval "$line"; printf '.' >&3 ) &
done < "$JOBS"
wait
echo "RUNQ_DONE launched=$n"
