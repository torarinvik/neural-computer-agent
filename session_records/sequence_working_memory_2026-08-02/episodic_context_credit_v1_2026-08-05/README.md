# Replicated episodic context and causal credit (2026-08-05)

This is the first promoted pressure test of the missing task-context and
credit loop. A replaceable memory-side `EpisodicContextEncoder` consumes only
ordered learned event tensors, opaque action vectors, scalar outcomes, and
presence. It is trained with augmented episode views and paired
common-random-number write utilities. An opaque route then learns candidate
artifact addressing from attempted-row outcomes; a new route is appended from
fresh outcomes while the context encoder and old router remain frozen.

The verifier privately generates three procedures with identical single-event
statistics but different ordered patterns. Candidate artifact keys are random
opaque vectors. Task families, correct rows, and semantic procedure names are
not inputs to the encoder or router.

## Promoted result

| Metric | Seed 69316 | Seed 69317 |
| --- | ---: | ---: |
| context old-route accuracy | 0.9688 | 1.0000 |
| pooled-event baseline | 0.5000 | 0.5000 |
| candidate permutation accuracy | 1.0000 | 1.0000 |
| new route selection after append | 1.0000 | 1.0000 |
| selection without new extension | 0.0000 | 0.0000 |
| reward-shuffled extension selection | 0.0000 | 0.0000 |
| decisive-position credit accuracy | 0.6667 | 0.6667 |
| replay after extension | 0 | 0 |
| unique verifier bits | 57,344 | 57,344 |
| unique logical lifetimes | 16,640 | 16,640 |

All context, permutation, new-route, causal ablation, retention, shuffled
outcome, credit, and no-replay gates passed on both seeds. The extension uses
a predeclared confidence threshold so a near-zero shuffled score cannot
activate a new capability after an old-route failure.

## Claim boundary

This promotes a bounded memory-side episodic-context and counterfactual-credit
mechanism for ordered synthetic procedures. It demonstrates that a reusable
trajectory representation can support opaque external routing and a fresh
no-replay append while protecting the old route. It is not unrestricted
memory growth, arbitrary program induction, natural-modality learning, or
general continual learning. The next frontier is scaling the same loop to
more procedures, longer and nonstationary episodes, and genuinely learned
write/read consolidation rather than a single calibrated extension.

Reports are in `report_seed69316.json` and `report_seed69317.json`.
