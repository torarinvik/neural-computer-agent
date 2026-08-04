# Parent-preserving counterfactual retention qualification

v41 extends the v39 counterfactual write-utility protocol with two safeguards:

1. one parent rehearsal update follows each retention update, preserving the
   mastered single-event primitive; and
2. every held-out validation checkpoint is scored on both retention utility and
   parent retention, with the first later-stable prefix selected.

The controller still receives only opaque events, opaque actions, and scalar
outcomes. Rehearsal is an ordinary single-event outcome-only episode. No
verifier target, branch position, or symbolic label enters the runtime.

The three-seed unprotected population passes the strict gate:

| seed | intact | clear | corrupt | reverse | target first | target last | missing-write cue | mastered primitive | stable bits |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1.000 | 0.494 | 0.479 | 0.996 | 0.995 | 1.000 | 0.749 | 0.988 | 34,816 |
| 18 | 0.997 | 0.521 | 0.517 | 0.997 | 0.997 | 0.995 | 0.756 | 0.996 | 28,672 |
| 19 | 0.991 | 0.521 | 0.479 | 0.989 | 0.989 | 0.999 | 0.752 | 0.988 | 25,600 |

Population means are `0.995` intact, `0.512` clear, `0.493` corrupt, `0.994`
reversed, `0.993` target-first, and `0.998` target-last. Mastered-primitive
retention averages `0.991`; random-action recall averages `0.493`. The
reward-shuffled seed-17 control remains at chance (`0.511` intact,
`0.491` target-first, `0.509` target-last) and retains no stable threshold.

Aggregate accounting for the valid population is 109,568 unique verifier bits,
48,640 unique logical lifetimes, 60,928 logical lifetime observations, 3,040
optimizer updates, 121,856 retention outcome events, 24,576 rehearsal outcome
events, 73,216 retention feedback events, 12,288 rehearsal feedback events,
52,736 diagnostic verifier bits, 20,224 diagnostic logical lifetimes, and
43.735 seconds of wall time.

This promotes the parent-preserving counterfactual write-utility protocol and
the narrow sub-minute learned retention result. No checkpoint is promoted yet.
The next rung is a longer rehearsal-preserving replication plus a matched
fresh-learner transfer curve; persistent memory remains unqualified.
