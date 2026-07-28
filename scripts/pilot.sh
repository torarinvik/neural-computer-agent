#!/usr/bin/env bash
# Look at one run before paying for a hundred.
#
#   pilot.sh "<one full training command, {seed}/{steps} already resolved>"
#
# Runs a single configuration, reports what it cost and what it measured, and
# refuses to bless a sweep whose grid is already saturated or still at chance.
# Both failures have happened: a 168-job cell was run on a budget grid where
# every seed had already converged at the first point, and a 12-job probe sat
# at chance because the support count was wrong. Each cost a full design cycle.
#
# Rule of thumb this enforces: no batch over ~20 jobs until one run of that
# exact configuration has been seen.
set -uo pipefail

CMD="${1:?full training command}"
VAST_HOST="${VAST_HOST:?set VAST_HOST}"
VAST_PORT="${VAST_PORT:?set VAST_PORT}"
REPO="${REPO:-/workspace/repos/neural-computer-agent}"

started=$(date +%s)
out=$(ssh -o ConnectTimeout=20 -p "$VAST_PORT" "root@$VAST_HOST" \
  "cd $REPO && source /venv/main/bin/activate >/dev/null 2>&1; $CMD" 2>&1 | tail -20)
elapsed=$(( $(date +%s) - started ))

echo "$out"
echo "---- pilot verdict ----"
echo "wall clock: ${elapsed}s for this configuration"

python3 - "$elapsed" <<'PY' <<< "$out"
import sys, json, re
elapsed = int(sys.argv[1])
text = sys.stdin.read()
accs = [json.loads(m).get("new_skill") for m in re.findall(r'^\{.*\}$', text, re.M)
        if "new_skill" in m]
accs = [a for a in accs if a is not None]
if not accs:
    print("no accuracy parsed -- inspect the output above before sweeping")
    raise SystemExit(0)
lo, hi = min(accs), max(accs)
print(f"new-skill accuracy across the pilot points: {lo:.4f} .. {hi:.4f}")
if hi < 0.60:
    print("STILL AT CHANCE -- the sweep would measure nothing. Fix support "
          "count, budget or learning rate first.")
elif lo > 0.97:
    print("ALREADY SATURATED -- every point is converged, so a transfer ratio "
          "from this grid is uninformative. Lower the budget grid.")
elif hi - lo < 0.05:
    print("FLAT -- the grid barely moves; widen it before spending a panel.")
else:
    print("USABLE -- the grid spans the learning curve.")
print(f"a 100-job sweep at this cost would take about "
      f"{elapsed*100/60:.0f} worker-minutes")
PY
