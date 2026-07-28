#!/usr/bin/env bash
# Bounded-concurrency job runner. One line of the job file is one command.
# The concurrency gate uses a fifo rather than `jobs`, which does not work
# inside a command substitution and silently launched every job at once.
set -u
# Resolve the repo explicitly. Deriving it from $0 is wrong when this script is
# shipped somewhere else: dirname /workspace/runq.sh/.. is /, the cd succeeds,
# and every job then runs from the filesystem root writing relative paths into
# nowhere. Fail loudly instead of running in the wrong place.
REPO="${REPO:-/workspace/repos/neural-computer-agent}"
cd "$REPO" || { echo "runq: cannot enter repo $REPO" >&2; exit 1; }
[ -d experiments ] || { echo "runq: $REPO is not the repo" >&2; exit 1; }
source /venv/main/bin/activate 2>/dev/null
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
JOBS="${1:?job file}"; P="${2:-6}"
# A single job may never outlast the campaign cap. Anything slower than this is
# a design error to fix at the grid, not something to sit and wait through.
export MAX_JOB_SECONDS="${MAX_JOB_SECONDS:-300}"
fifo=$(mktemp -u); mkfifo "$fifo"; exec 3<>"$fifo"; rm -f "$fifo"
for _ in $(seq "$P"); do printf '.' >&3; done
n=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  read -r -n1 -u3 _
  n=$((n+1))
  ( timeout "${MAX_JOB_SECONDS:-300}" bash -c "$line" \
      || echo "runq: job exceeded ${MAX_JOB_SECONDS:-300}s or failed: $line" >&2
    printf '.' >&3 ) &
done < "$JOBS"
wait
echo "RUNQ_DONE launched=$n"
