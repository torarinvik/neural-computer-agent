# Onset search lease, 192 steps (2026-08-15)

Status: **replicated, not admitted** at the time of measurement; below the
trial floor adopted afterwards. See Superseded.

Unused seeds 128017, 129017, and 130017 ran the same closed-grammar search on
the public onset rule with episode length as the only change from
`brainworkshop_onset_search_lease_2026-08-15` (192 steps instead of 48, so 191
eligible trials instead of 47). This population does not reuse 125017–127017,
122017–124017, 116017–121017, or the Dual lease. `AgentBrain.bank` was not
written and nothing was admitted.

## Results

Winner is `and:0` on every seed, bound to frontend `1ce405a0…`. Controller
digest `59c9ef2b…`. Slot 0 stayed `90e20193…`.

| Seed | Acquire | Six frozen holds | Stable prefix | retrieve slot 0 | invert slot 0 | prototype only | zeros | reversed | reward shuffled | other encoder |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128017 | 0.749 | 1.000 × 6 (1146 bits) | 191 | 0.251 | 0.749 | 0.759 | 0.749 | 0.000 | 0.492 | 0.749 |
| 129017 | 0.749 | 1.000 × 6 (1146 bits) | 191 | 0.251 | 0.749 | 0.707 | 0.749 | 0.000 | 0.497 | 0.749 |
| 130017 | 0.749 | 1.000 × 6 (1146 bits) | 191 | 0.251 | 0.749 | 0.759 | 0.749 | 0.000 | 0.508 | 0.749 |

Stable bits are the first hold prefix that remains at every later measured
prefix. All six holds were at `1.000` with zero program-file updates after the
acquire session.

## What this supports

Onset needs two families on this machine. Neither admitted delay file
(`0.251`) nor its inverse (`0.749`) clears `0.8` alone, and the acquired
current-symbol prototype alone reaches at most `0.759`. Only
`and(invert(slot 0), prototype)` reaches `1.000`, and it does so from a
two-phase acquire: the machine acts as the inverse so the verifier rewards
change trials, then the prototype is the running mean of invert-matching,
reward-1 events. The search is not told the rule, the depth, or the winner.

The prototype-only ceiling here (`0.707`–`0.759`) is the rule's base rate. The
48-step run of the same design is rejected because that control crossed `0.8`
on one seed at 47 eligible trials; the control itself was never relaxed.

## Grammar coverage

Of 19 proposals, 5 executed, 7 were recorded illegal, and 4 (retrieve slot 2,
`compose(0,0)`, `compose(1,1)`, invert slot 2) could not be installed on the
prototype-capable machine, which does not load recursive depth-2 files.

## Not claimed

- no admission: the AND child is not a curated `AgentBrain.bank` slot;
- no bits-to-threshold transfer ratio against a fresh learner;
- no learned proposer; the search still enumerates the closed grammar;
- no coverage of depth-2 files on this machine;
- no new operator types beyond retrieve, compose, invert, and, invent.

## Superseded

At 191 eligible trials a `0.75` policy still reaches `0.8` about 5.9% of the
time, so this campaign sits below the 1% floor later stated in
`PREREGISTRATION_discriminating_leases_2026-08-15.md` and is no longer
accepted under it. Its measured numbers are unchanged and its conclusion was
reproduced on a fresh block at 447 eligible trials in
`brainworkshop_onset_lease_discriminating_2026-08-15`, which is the standing
onset result.
