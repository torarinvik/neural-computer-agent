# Outcome-gated open external-compute growth

This promotion tests whether the memory side can grow as capabilities are
verified, rather than requiring a preallocated fixed bank.

## Protocol

- Start with one empty external compute file.
- Train each candidate from fresh rendered lifetimes using only scalar
  verifier outcomes.
- Admit a candidate only after every fresh retention probe reaches the stable
  mastery threshold.
- Freeze admitted files and the shared controller/frontend before the next
  candidate.
- If a candidate fails, roll back only that newest file and continue with the
  next candidate.
- After five files are admitted, train outcome-only routing and reverse the
  task behind the original cue. Probe all opaque files with fresh scalar
  outcomes, demote the stale route, and verify old-file retention.

The candidate sequence was symbol_parity, triplet_parity, parity2,
switch_binary, nback2, and symbol_parity_odd. The nback2 candidate failed
stable direct mastery in both seeds and was discarded; the later odd parity
candidate reused its physical slot and was admitted. This is a deliberate
allocation-control result, not a hidden omission.

## Results

Seeds 17 and 18 both passed every promotion gate:

| Measure | Seed 17 | Seed 18 |
| --- | ---: | ---: |
| Files admitted | 5 / 5 | 5 / 5 |
| Rejected candidates rolled back | 1 | 1 |
| Minimum routed-file accuracy | 1.0000 | 0.8693 |
| Same-cue replacement accuracy | 1.0000 | 1.0000 |
| Old-file forced retention | 1.0000 | 1.0000 |
| Route reload | exact | exact |
| Replayed examples | 0 | 0 |

The controller, event encoder, and all admitted files remained unchanged
during routing and reversal. The unknown cue used the oldest fallback and
remained at 0.0000 accuracy against the final odd-parity verifier, a valid
negative control rather than a chance claim.

Each seed accounted for 880,128 unique training verifier bits, 23,552 audit
verifier bits, 70,272 logical lifetimes, 1,152 optimizer updates, 1,092
route-memory updates, and zero replayed examples. The totals include the
rejected nback2 candidate.

## Claim boundary

This promotes replicated outcome-gated append-only capacity growth with
candidate rollback, protected-prefix retention, and same-context route
reversal. It does not establish unrestricted physical storage, arbitrary
program induction, learned compression, or general continual learning. The
remaining pressure points are repeated growth beyond this candidate budget,
harder reusable computation families such as n-back depth, and consolidation
under genuine capacity pressure.

The raw reports are seed-17.json and seed-18.json.
