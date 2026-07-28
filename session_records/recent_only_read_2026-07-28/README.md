# Recent-only ancestry read pilot — 2026-07-28

## Question

Reading one prior skill slot restored transfer at ancestry 3 → 4, but exposing
two prior slots did not improve the 4 → 5 depth differential. Does a new slot
retain the useful read-path effect if it consults only its immediately preceding
ancestor?

The controller still receives no task ID, rule label, correct unattempted
action, or symbolic state. This is an architectural ablation over opaque latent
reads. Writes remain behind the existing exact-zero gates.

## Factorial reanalysis before spending compute

The completed 4 → 5 grid already contained matched read/no-read cells at each
fixed ancestry depth. Re-pairing those cells separates the absolute value of
reading from the marginal depth advantage:

| fixed parent | raw read minus no read | paired signs |
|---|---:|---:|
| four-skill parent | **+0.09442** | 22W / 2L, p = 3.59e-5 |
| five-skill parent | **+0.07847** | 20W / 4L, p = 1.54e-3 |

Reading is therefore highly useful in absolute learning accuracy at both
depths. The bounded negative is narrower: adding a second readable ancestor
does not make the deeper parent improve *more* than the shallower parent.

Read bottlenecks of 16 and 32 dimensions also improved absolute accuracy, but
less than the raw path. Their failure was a failure to restore the depth
differential, not evidence that they carried no useful information.

## Change

`skill_adapter_prior_read_limit` is a generic nonnegative controller setting:

- `0` preserves the old behavior and reads every prior slot;
- `1` reads only the immediately preceding slot;
- larger values retain that many recent predecessors.

The default is backward-compatible. Structural tests verify both the
recent-only and read-all input widths.

## Sub-minute result

Eight matched seeds, 96 updates, 3,072 unique new-task lifetimes per seed.
Two concurrent workers finished the 768 aggregate updates in about 25 seconds
wall time (about 50 seconds summed training time).

| condition | mean new-skill accuracy |
|---|---:|
| deep parent, no read | 0.61591 |
| deep parent, all prior slots | 0.67749 |
| **deep parent, immediate prior slot only** | **0.69659** |
| shallow parent, its one prior slot | 0.71704 |

Paired differences for immediate-only:

- versus no read: **+0.08069**, 7W / 1L, two-sided sign p = 0.0703;
- versus all-prior read: **+0.01910**, 5W / 3L, p = 0.727;
- versus the shallow one-read parent: **−0.02045**, 3W / 5L, p = 0.727.

Screening retention stayed close to the all-read arm: mean deltas ranged from
−0.00347 to +0.00186 across the five inherited primitives. This pilot did not
run the larger promotion-grade retention audit.

## Decision

Do not promote to a longer run yet. Recent-only reading preserves the large
benefit of reading at all and has a small favorable direction over reading both
ancestors, but it does not yet produce a positive compounding depth
differential.

The next sub-minute diagnostic should compare which single ancestor is exposed
(immediate versus older) under identical seeds. A strong difference would
justify a task-agnostic learned latent selector. A flat result would argue that
the plateau is not caused by irrelevant-ancestor routing.
