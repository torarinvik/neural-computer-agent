# Replicated multi-step episodic context, credit, and route growth (2026-08-05)

This record extends the single-append context/credit result to two sequential
fresh capability additions. The recurrent episode context and the original
opaque router are frozen after the old procedures. Each new artifact receives
isolated event-credit state and an isolated route extension trained only from
fresh paired scalar outcomes. A later procedure must pass the old route and
the earlier extension before its own extension can activate.

The verifier privately generates four temporal procedures with identical
single-event statistics but distinct order. Candidate artifact keys are random
opaque vectors. The model sees no task IDs, correct rows, semantic operation
names, or unattempted-action labels.

## Promoted result

| Metric | Seed 69316 | Seed 69317 |
| --- | ---: | ---: |
| context old-route accuracy | 0.9688 | 1.0000 |
| pooled-event baseline | 0.5000 | 0.5000 |
| candidate permutation accuracy | 1.0000 | 1.0000 |
| new route 2 / route 3 selection | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| old-route failure on new procedures | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| prior-extension attempts | pass | pass |
| new-route ablation | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| shuffled-extension selection | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| old / new credit-position accuracy | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| replay after append | 0 | 0 |
| unique verifier bits | 131,072 | 131,072 |
| unique logical lifetimes | 28,928 | 28,928 |

All cumulative route, prior-extension, isolated-credit, causal ablation,
retention, permutation, shuffled-outcome, and no-replay gates passed across
both seeds. The fixed activation threshold is a safety boundary: a near-zero
shuffled extension cannot become executable merely because an earlier route
failed.

## Claim boundary

This promotes bounded two-step external growth with isolated episodic credit
state. It demonstrates replay-free addition of two sequential capabilities
while preserving the old context and credit state. It is not unrestricted
memory growth, arbitrary program induction, learned eviction, nonstationary
task discovery, natural-modality learning, or general continual learning.

Reports are in `report_seed69316.json` and `report_seed69317.json`.
