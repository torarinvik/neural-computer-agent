# Generated length-six twelve-addition pressure test — rejected

Status: rejected at the cross-seed retention gate.

This rung extends the promoted generated length-six sequence from eight to
twelve sequential additions, for fourteen opaque capabilities total. The
context encoder, old route, and per-capability external credit heads remain
frozen after acquisition; the run still uses zero replay.

Seed 69316 passed the route, causal, isolated-credit, and retention-reversal
gates. Seed 69317 preserved the route and credit signals, but the final
capability reached only `0.8125` fresh route selection and failed stable
initial protection. The fully protected-bank refusal and recovery gates then
failed as a consequence. The rung is rejected rather than promoted because a
single seed-unstable final capability is evidence of a real scaling weakness.

The result localizes the next bottleneck: retention calibration and route
margin under a larger candidate bank. Increasing the retention threshold or
silently excluding the last capability would not be a valid fix. The next
experiment should improve confidence-aware admission/retention evidence and
repeat the twelve-addition audit before attempting further growth.

Evidence is in `report_seed69316.json` and `report_seed69317.json`.
