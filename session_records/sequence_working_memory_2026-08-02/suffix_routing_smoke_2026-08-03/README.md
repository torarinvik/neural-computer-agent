# Suffix curriculum and successor-slot routing smoke (2026-08-03)

This is a sub-minute diagnostic record, not a promoted checkpoint. It was run
against the verified fourth-slot parent:

`artifacts/checkpoints/complement_population_fourth_slot_seed93871.pt`

The goal was to test whether an easier-prefix query curriculum could open a
causal signal for a new span-eleven slot without spending a longer run. The
learner received only the opaque attempted action and scalar outcome; no
semantic task labels or privileged rule information were supplied.

## Matched smoke arms

Both arms used seed/data-seed `93910`, 64 fresh target lifetimes, 64-lifetime
span-nine/span-ten rehearsal, span 11 complement targets, two distractors,
8 epochs, 512 batch size, MPS, and the same model/penalties.

| arm | curriculum | candidate | zeroed slot | causal gain |
| --- | --- | ---: | ---: | ---: |
| baseline | none | 72.2301% | 71.0227% | **+1.2074 pp** |
| staged | 4 warmup epochs on first 2 outputs, then full span | 71.0227% | 71.0227% | **+0.0000 pp** |

The registered promotion bar is +5 pp causal gain, so neither arm is a
candidate for scaling. Old-span changes on this tiny audit are noisy and the
run is intentionally only a routing/credit smoke test.

Saved reports from the run are the four JSON files in this directory. The
temporary model files were not promoted; their hashes are recorded for
provenance if they are still available in `/tmp`:

* baseline: `6cca4aaf306fe381c7caa2a9c307ccc2c1c845a2aaae9c8dbf24425b08d98273`
* staged: `9864f98c2be336f4bf266e8ff61a7713b93585acfcc885d7acfa061d1b16e41a`

## Routing diagnosis

The trained successor-slot gate was evaluated on feature streams without
using task labels. Its opening was high on old mixed span-nine/span-ten
streams (roughly `0.76–0.79`) and higher on the new complement stream (roughly
`2.69`), while the parent slot was essentially closed on old streams. The new
slot therefore perturbs old skills broadly instead of selectively routing to
the new event. Parent-action entropy was near zero on both old and target
streams, and event-age ranges overlapped, so neither is currently a useful
selector by itself.

**Conclusion:** the immediate bottleneck is generic slot selectivity / credit
routing, not simply more target data or a prefix curriculum. Do not scale this
curriculum. The next experiment should test a task-agnostic suffix/window or
novelty route that preserves old streams, with the existing causal, reset,
shuffle, reversal, and retention audits kept as hard gates.

## Reproduction

The exact command lines and full nested metrics remain in the archived JSON
reports. The baseline used `train_sequence_reward_buffer` with
`--append-skill-slot`; the staged arm added
`--query-curriculum-warmup-epochs 4 --query-curriculum-cutoff 2`.
