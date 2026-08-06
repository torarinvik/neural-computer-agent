# Shared candidate growth router: length 6 → 8 → 10 → 12 (2026-08-06)

This audit replaces the one-scalar-extension-per-capability growth pattern
with one permutation-equivariant, key-conditioned router per temporal shift.
Each shift router scores that shift's whole opaque candidate bank; it is
frozen before the next shift, so old routes never depend on replay or on
mutable controller weights. The promoted configuration uses the rich learned
trajectory-stats route query (context, final, mean, and max recurrent state),
random opaque candidate keys, hidden width 256, and 16,384 shared route
updates per bank.

Command (per seed):

```bash
uv run python -m experiments.episodic_context_credit_amodal.shared_growth_router \
  --seed <seed> --shift-episode-lengths 8,10,12 --families-per-shift 8,10,12 \
  --context-updates 1024 --credit-updates 512 --external-credit-updates 128 \
  --route-updates 1024 --shared-route-updates 16384 --shared-route-hidden 256 \
  --route-query-representation trajectory_stats --candidate-key-bootstrap random \
  --batch-size 16 --audit-batch-size 64
```

## Promoted result

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| phase-1 minimum route selection | 0.9844 | 0.9375 |
| phase-2 minimum route selection | 0.9844 | 0.9688 |
| phase-3 minimum route selection | 0.9531 | 0.9375 |
| old route accuracy | 1.0000 | 1.0000 |
| candidate permutation accuracy | 1.0000 | 1.0000 |
| reward-shuffled false selections | 0 | 0 |
| full-bank protection/reversal/recovery | passed | passed |
| replayed examples | 0 | 0 |

Both seeds pass every hard gate: old-route retention, candidate permutation
invariance, route recovery, causal routing, reward-shuffled null, credit
survival across all shifts, zero replay, and retention reversal safety.

## What made it work

The narrow 16-dimensional projected context was the bottleneck: with it the
shared router had to memorize arbitrary query-to-row assignments and failed
the 12-way bank at every tested width. Replacing the route query with a fixed
learned trajectory summary restored the missing information without exposing
raw modality data or task labels. Query-prototype key bootstrapping and an
explicit negative-margin calibration were both tested and rejected as
controls; random keys with paired counterfactual ranking remain canonical.

## Claim boundary

This promotes one reusable routing module per shift over 30 new capabilities
in three shifts, replay-free, with all earlier state frozen. It does not
promote a single router shared across shifts, unbounded shift counts, or
sample-efficient acquisition (16,384 updates per bank is expensive). Learned
capacity scheduling and general continual learning remain open.
