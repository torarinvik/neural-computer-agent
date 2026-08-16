# A remembered self must know when it no longer applies (2026-08-16)

Status: **development-seed diagnostic; current safety mechanism rejected.**
Nothing admitted. `AgentBrain.bank` remained byte-identical at `07319eb1`.

The integrated navigation result improved when a soft self model accumulated
across episodes, but that result began from a favourable initialization in one
unchanging world. This audit attacks the assumptions that made the improvement
possible before any holdout seed is spent.

## Result

Three development worlds, forty twenty-step episodes per distinct stream.
`precision` is conditional on being confident enough to name a track;
`confidently wrong` is measured over every episode.

| condition | named | abstained | precision when named | confidently wrong |
| --- | ---: | ---: | ---: | ---: |
| honest | 0.992 | 0.008 | 0.891 | 0.108 |
| transplanted model | 0.025 | **0.975** | 0.500 | 0.008 |
| dynamics changed | 0.175 | 0.825 | **0.071** | **0.150** |
| poisoned initialization | 1.000 | 0.000 | 0.817 | 0.183 |
| exact dynamic mimic | 0.000 | **1.000** | n/a | **0.000** |

Two controls work for the intended reason. A model from an unrelated world
mostly refuses to name anything, and two dynamically identical tracks produce
complete abstention rather than a tie-broken 50% "accuracy".

The changed-world condition fails the safety claim. The model usually
abstains, but when it does name a track it is wrong almost every time. A 15%
confident-error rate is not usable by a downstream world model. This run tests
the frozen model immediately after the change; it does **not** claim online
recovery, because no causal invalidation-and-relearning mechanism exists yet.

## What the new terms bought, and cost

The exact pre-audit likelihood posterior is retained as an ablation.

| posterior | honest precision | poisoned precision | confidently wrong when poisoned |
| --- | ---: | ---: | ---: |
| likelihood only | **0.925** | **0.000** | **1.000** |
| applicability guard + controllability | 0.891 | **0.817** | 0.183 |

The old posterior is a self-confirming loop: start it on the wrong track and
six re-fitting passes leave every episode confidently wrong. The guarded
posterior breaks that fixed point decisively, but pays 0.034 of honest
precision and does not solve dynamics reversal. This is a real mechanism-level
improvement and an overall rejection of the current configuration as a safe
persistent self model.

The applicability margin (`0.25`) and controllability weight (`2.0`) were
chosen on these already-consumed development worlds. They are not eligible for
further tuning on a future holdout block.

## Accounting

- unique verifier bits: 4,800;
- unique logical lifetimes: 240;
- optimizer updates: 0;
- replayed episode histories: 9,600;
- wall time: 35.77 seconds;
- stable bits-to-threshold: not applicable; no mastery threshold;
- primitive retention: not claimed;
- transfer ratio against a fresh learner: not claimed.

Only the current-world and alternate-world interaction streams are unique.
The mimic and posterior ablations transform or re-read those already-paid
streams and are counted as replay rather than new experience.

## Cleanup findings

The first end-to-end test appeared to show that the run mutated
`AgentBrain.bank`. It had not: a transition-loop variable named `before`
shadowed the stored pre-run digest, causing the fail-closed comparison to reject
an unchanged file. The test caught the false alarm; the variable is now named
for its actual meaning, and the bank digest is asserted before and after the
run.

The earlier reversal arm retrospectively re-fitted one model over episodes
from both worlds. That is not change detection. It was replaced with the
causal question this mechanism can currently answer: what does the frozen
self model believe on the first post-change episodes, before hindsight?

## Claim boundary and next gate

This does not spend a holdout block and does not justify promotion, admission,
or persistent use in the live runtime. The next mechanism must explicitly
invalidate or quarantine a self model when its predictive evidence collapses,
then measure detection and recovery online. Only a frozen mechanism with a
pre-registered confident-error gate should receive new holdout worlds.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m \
  experiments.brainworkshop_canonical.self_model_adversarial
```

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_slot_alignment.py tests/test_self_model_adversarial.py -q
```
