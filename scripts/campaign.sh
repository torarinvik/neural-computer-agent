#!/usr/bin/env bash
# Run one campaign end to end without anyone watching it.
#
#   campaign.sh <name> <jobfile> [workers] [scorer]
#
# Ships the job file, runs it on the rented box, streams results back while it
# runs, and scores the result when it finishes. Intended to be launched in the
# background: the point is that the next thing a human or agent sees is the
# scored answer, not a progress number. Waiting on a sweep by hand was the
# single largest time sink in earlier sessions.
#
# Results stream back every STREAM_SECONDS, so a box that dies or is recycled
# costs only the runs still in flight, never the ones already finished.
#
# Environment: VAST_HOST, VAST_PORT, REPO (remote path), LOCAL_REPO.
set -uo pipefail

# Run from a snapshot of this file. Bash reads a script incrementally, so
# editing campaign.sh while a campaign is running makes the live shell resume
# into whatever the file now says -- which killed a watcher mid-run with a
# syntax error while the remote work carried on fine.
if [ "${CAMPAIGN_SNAPSHOT:-0}" != "1" ]; then
  _snap=$(mktemp "${TMPDIR:-/tmp}/campaign.XXXXXX.sh")
  cp "$0" "$_snap"
  CAMPAIGN_SNAPSHOT=1 exec bash "$_snap" "$@"
fi

NAME="${1:?campaign name}"
JOBFILE="${2:?job file}"
WORKERS="${3:-6}"
SCORER="${4:-}"

VAST_HOST="${VAST_HOST:?set VAST_HOST}"
VAST_PORT="${VAST_PORT:?set VAST_PORT}"
REPO="${REPO:-/workspace/repos/neural-computer-agent}"
LOCAL_REPO="${LOCAL_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
STREAM_SECONDS="${STREAM_SECONDS:-30}"
# Hard cap. A campaign returns within this many seconds no matter what, scoring
# whatever has landed. Every scorer here globs the reports that exist, so a
# partial score is a real partial answer rather than an error. Waiting half an
# hour for one number is what this exists to prevent.
CAMPAIGN_MAX_SECONDS="${CAMPAIGN_MAX_SECONDS:-300}"
SSH=(ssh -o ConnectTimeout=20 -o ServerAliveInterval=20 -p "$VAST_PORT" "root@$VAST_HOST")

retry() {  # a rented box refuses connections often enough to matter
  local n=0
  until "$@"; do
    n=$((n+1)); [ "$n" -ge 5 ] && return 1
    sleep $(( n * 5 ))
  done
}

echo "[campaign:$NAME] shipping $(wc -l < "$JOBFILE") jobs"
retry scp -o ConnectTimeout=20 -q -P "$VAST_PORT" "$JOBFILE" \
  "root@$VAST_HOST:/workspace/$NAME.jobs" || { echo "[campaign:$NAME] ship failed"; exit 1; }
retry scp -o ConnectTimeout=20 -q -P "$VAST_PORT" "$LOCAL_REPO/scripts/runq.sh" \
  "root@$VAST_HOST:/workspace/runq.sh" || true

retry "${SSH[@]}" "cd $REPO && setsid nohup bash /workspace/runq.sh /workspace/$NAME.jobs $WORKERS \
  > /workspace/$NAME.log 2>&1 < /dev/null & echo started" || exit 1

echo "[campaign:$NAME] running with $WORKERS workers; cap ${CAMPAIGN_MAX_SECONDS}s"
CAMPAIGN_STARTED=$(date +%s)
TIMED_OUT=0
while true; do
  sleep "$STREAM_SECONDS"
  if [ $(( $(date +%s) - CAMPAIGN_STARTED )) -ge "$CAMPAIGN_MAX_SECONDS" ]; then
    echo "[campaign:$NAME] hit the ${CAMPAIGN_MAX_SECONDS}s cap; scoring what landed"
    TIMED_OUT=1
    break
  fi
  # stream partial results home so a lost box never costs finished work
  rsync -az -e "ssh -o ConnectTimeout=20 -p $VAST_PORT" \
    "root@$VAST_HOST:$REPO/session_records/" "$LOCAL_REPO/session_records/" 2>/dev/null
  done_line=$("${SSH[@]}" "tail -1 /workspace/$NAME.log 2>/dev/null" 2>/dev/null | tr -d '\r')
  case "$done_line" in
    RUNQ_DONE*) echo "[campaign:$NAME] $done_line"; break;;
  esac
  alive=$("${SSH[@]}" "pgrep -cf '[r]unq.sh' 2>/dev/null || echo 0" 2>/dev/null | tail -1)
  if [ "${alive:-0}" = "0" ]; then
    echo "[campaign:$NAME] runner gone without a done marker; stopping watch"
    break
  fi
done

rsync -az -e "ssh -o ConnectTimeout=20 -p $VAST_PORT" \
  "root@$VAST_HOST:$REPO/session_records/" "$LOCAL_REPO/session_records/" 2>/dev/null
echo "[campaign:$NAME] results pulled to $LOCAL_REPO/session_records/"

# A campaign that produced nothing must say so. The runner once resolved its
# working directory to / and every job failed silently; the scorer then printed
# an empty table that looked like a real negative result.
produced=$("${SSH[@]}" "ls $REPO/session_records/*/raw/*.json 2>/dev/null | wc -l" 2>/dev/null | tail -1)
if [ "${produced:-0}" = "0" ]; then
  echo "[campaign:$NAME] WARNING: the run produced no reports at all."
  echo "[campaign:$NAME] last runner output:"
  "${SSH[@]}" "tail -15 /workspace/$NAME.log" 2>/dev/null
fi

if [ "$TIMED_OUT" = "1" ]; then
  echo "[campaign:$NAME] NOTE: partial. The remainder is still running on the box;"
  echo "[campaign:$NAME] rerun this command to collect and rescore, or shrink the grid."
fi
if [ -n "$SCORER" ] && [ -f "$SCORER" ]; then
  echo "[campaign:$NAME] ---- score ----"
  ( cd "$LOCAL_REPO" && PYTHONPATH="$LOCAL_REPO" .venv/bin/python "$SCORER" )
fi
echo "[campaign:$NAME] complete"
