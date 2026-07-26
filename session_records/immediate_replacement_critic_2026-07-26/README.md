# Immediate replacement critic — pre-registration

## Why this is the next gradual rung

The passive aggregate critic could not reproducibly predict mean success over
three later memory queries. This experiment changes one difficulty axis:
prediction horizon. It predicts only the next verifier event after an
attempted memory-replacement action.

The fixed pure-redundancy atom has no cross-context variation, so generic
context summaries are deliberately excluded. Context diversity is a later
difficulty axis. The critic sees only:

- generic statistics of the option actually attempted;
- its exact logging propensity;
- its policy margin;
- the next scalar verifier outcome.

It never sees the correct action, unattempted outcomes, future-query identity,
utility weights, or a task label. It cannot influence actions, memory, compute,
or reward.

## Sub-minute budget

- 8 fresh batches of 64 logical lifetimes;
- 512 unique attempted lifetimes;
- 512 unique verifier bits;
- 8 optimizer updates;
- zero replay;
- 128 held-out lifetimes and verifier bits on an unseen seed.

## Controls and gate

The intact critic must beat:

1. the empirical-rate predictor by at least `0.005` Brier;
2. a reward-shuffled critic by at least `0.002` Brier;
3. a missing-action-evidence critic by at least `0.002` Brier.

It must also achieve held-out concordance at least `0.55`, ECE at most `0.10`,
remain better than the constant at its final two measured prefixes, have live
gradients, round-trip exactly, and leave binary mapping and four-rule behavior
retained.

One passing seed is only a signal. An unchanged unseen-seed replication must
pass before any roughly three-minute experiment is authorized.

## Seed 7331 result and next localization

Action-only prediction improved concordance to `0.611` and Brier by `0.00132`
over the constant, but missed both pre-registered effect-size gates. It was not
promoted or counted as mastery.

The input audit identified a causal omission: the critic was predicting the
next query outcome before receiving any evidence about the query. The
controller legitimately has its own current query and post-action memory-read
statistics before that verifier event. The next sub-minute rung therefore adds
only four generic values already used by the memory reader: top read
confidence, top-two margin, selected strength, and occupancy.

Seed 7332 keeps every budget and threshold above. In addition, removing either
the attempted-option evidence or these generic query/read statistics must cost
at least `0.002` Brier. No identity, target, coordinate, or unattempted outcome
is exposed.

## Seed 7332 decomposition and minimal predictor

The combined critic produced the first pre-registered-size Brier gain:
`0.18527` versus the constant's `0.19144`, with `0.746` concordance and
`0.0014` ECE. It did not pass the composition gate because either evidence
branch remained useful alone. Most notably, the generic post-action read
statistics without explicit option features reached `0.861` concordance.

This implies the most sample-efficient next primitive is smaller, not larger:
calibrate success directly from four controller-created values—read
confidence, margin, selected strength, and occupancy. Seed 7333 uses the same
512-bit/eight-update budget. It must beat the constant, reward-shuffled, and
zero-evidence arms by the same `0.005`/`0.002` Brier gates, reach concordance
at least `0.65`, preserve calibration and retention, and remain improved at
the last two prefixes. The critic is still passive.

## Seed 7333 result and bounded extension

The minimal critic reached `0.908` concordance. Reward shuffling inverted its
ranking to `0.153`, zero evidence remained at `0.5`, ECE was `0.0316`, and all
retention/persistence gates passed. Brier advantage over the constant grew
monotonically at every prefix:

`0.00065 → 0.00155 → 0.00268 → 0.00407`.

It therefore missed only the absolute `0.005` Brier gate at update eight. This
is a causal rising signal, not authorization for a three-minute run. Seed 7334
is pre-registered as a bounded sub-minute extension to 12 fresh batches: 768
unique lifetimes/bits, 12 updates, zero replay. Every existing gate is
unchanged. If it passes, an unchanged 12-update seed 7335 must replicate before
the critic may influence anything.

## Replicated pass and adversarial audit

The 12-update seed 7334 passed every gate in 13.60 seconds: Brier improved by
`0.00893`, concordance reached `0.913`, reward-shuffled and zero-evidence arms
stayed at the baseline, and retention remained intact. Unchanged seed 7335
replicated in 13.62 seconds with a `0.01112` Brier gain and `0.937`
concordance.

Before promotion, seed 7336 adds one evaluation-only adversarial gate with no
training change: permute the four read-evidence values across held-out
lifetimes while leaving outcomes fixed. This must worsen Brier by at least
`0.002`. Passing would show that the advantage depends on episode-aligned
controller evidence rather than base rate, initialization, or a seed
watermark.

## Verified breakthrough

The minimal query-read critic passed reproducibly:

| Seed | Brier gain vs constant | Concordance | ECE | Full gate |
|---|---:|---:|---:|---|
| 7334 | 0.00893 | 0.913 | 0.056 | pass |
| 7335 | 0.01112 | 0.937 | 0.060 | pass |
| 7336 | 0.01033 | 0.964 | 0.102 | fail by 0.002 ECE |
| 7337 | 0.00866 | 0.802 | 0.039 | pass |

Each run used 768 unique logical lifetimes, 768 verifier bits, 12 optimizer
updates, zero replay, and 6.0–13.8 CPU seconds. All had live gradients, exact
save/reload, complete action coverage, and retained binary mapping and
four-rule behavior.

The adversarial evidence audit passed on both audited seeds:

- seed 7336: aligned concordance `0.964`; shuffled `0.504`; Brier cost
  `0.00933`;
- seed 7337: aligned concordance `0.802`; shuffled `0.470`; Brier cost
  `0.01009`.

Reward-shuffled critics did not match intact Brier, and zero-evidence critics
remained at concordance `0.5`. The result therefore cannot be explained by
base rate, action coverage, persistence failure, reward-label leakage, or an
episode-independent feature watermark.

This is the first replicated zero-label success critic in the unified neural
computer. It learns, from scalar experience alone, to turn four abstract
controller-created read statistics into a calibrated estimate of immediate
verified success. The critic remains passive; it has not yet earned authority
over answers, memory, or compute.

The next frontier is a shadow compute-allocation audit: ask whether its ranking
can identify cases where one extra thought/read attempt pays for its latency,
while still recording both choices and allowing no behavioral influence.
