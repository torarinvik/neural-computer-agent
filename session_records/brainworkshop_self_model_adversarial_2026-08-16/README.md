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
| dynamics changed, causal quarantine/recovery | 0.800 | 0.200 | **0.885** | **0.092** |
| poisoned initialization | 1.000 | 0.000 | 0.817 | 0.183 |
| exact dynamic mimic | 0.000 | **1.000** | n/a | **0.000** |

Two controls work for the intended reason. A model from an unrelated world
mostly refuses to name anything, and two dynamically identical tracks produce
complete abstention rather than a tie-broken 50% "accuracy".

The causal reversal now fits before the change, scores each post-change episode
against the live model, quarantines on applicability collapse, and rebuilds
from post-change evidence only. It detects the first failure in post-change
episodes 0, 1, and 0 across the three worlds, and first recovery occurs at
episodes 1, 2, and 1. The mechanism can re-enter quarantine, but it still has a
9.2% confidently-wrong rate after recovery and therefore fails the safety gate.
Detection and recovery are measured, not claimed as solved.

The near-mimic controls are deliberately harder than exact symmetry:

| control | named | abstained | precision when named | confidently wrong |
| --- | ---: | ---: | ---: | ---: |
| delayed copy | 0.975 | 0.025 | 0.906 | 0.092 |
| partial response | 0.983 | 0.017 | 0.865 | **0.133** |
| stochastic copy | 0.983 | 0.017 | 0.907 | 0.092 |
| independent controller | 1.000 | 0.000 | 0.958 | 0.042 |

These controls show that exact-mimic abstention does not generalize to
near-equivalent causal ambiguity. They remain diagnostics, not promotion
claims.

## What the new terms bought, and cost

The exact pre-audit likelihood posterior is retained as an ablation.

| posterior | honest precision | poisoned precision | confidently wrong when poisoned |
| --- | ---: | ---: | ---: |
| likelihood only | **0.925** | **0.000** | **1.000** |
| applicability guard + controllability | 0.891 | **0.817** | 0.183 |

The complete baseline ladder is recorded in the JSON report: episodic identity
only; remembered likelihood; likelihood plus applicability gating; and the
full likelihood-plus-controllability mechanism. The honest precisions are
0.792, 0.925, 0.933, and 0.891 respectively. The poisoned confidently-wrong
rates are 1.000, 1.000, 1.000, and 0.183.

Starting from the poisoned posterior and applying the guarded update repeatedly
reduces confidently-wrong from 1.000 to 0.817 after six passes, but never to a
safe zero-error condition. This is recovery from a poisoned initialization as a
measured curve, not a success claim.

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
- replayed episode histories: 19,740;
- wall time: 75.65 seconds;
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
from both worlds. That is not change detection. It is now a causal stream with
frozen pre-change weights, explicit quarantine, post-change-only rebuilding,
per-episode detection/recovery events, and repeated invalidation when the
replacement becomes inapplicable. Exact and near-mimic transforms are
recomputed from synthetic traces without paying new verifier interactions.

## Claim boundary and next gate

This does not spend a holdout block and does not justify promotion, admission,
or persistent use in the live runtime. The current mechanism detects some
changes and escapes the poisoned fixed point, but its post-recovery confident
errors and near-mimic failures are too large for a downstream world model.
Only a new frozen mechanism with a pre-registered confident-error gate should
receive new holdout worlds.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m \
  experiments.brainworkshop_canonical.self_model_adversarial
```

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_slot_alignment.py tests/test_self_model_adversarial.py -q
```
