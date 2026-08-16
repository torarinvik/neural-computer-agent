# Persistent identity v3 in the rendered navigation loop (2026-08-16)

Status: **development composition diagnostic; not promoted**. This run uses
the consumed development world block and leaves the reserved integrated
self-model holdout untouched. `AgentBrain.bank` stayed at
`07319eb13c9cac58bbbe94258548e64f13ba1b3512ca2a01fbdc675c49e2e7c9`.

## What was composed

The v3 artifact was called at the intended external seam:

```text
rendered frame
  -> learned slot event tensors
  -> tracker
  -> persistent causal identity v3 (or explicit abstention)
  -> external relational world model and successor policy
  -> opaque decoder action
  -> rendered transition and scalar verifier feedback
```

All five matched arms used the same sampled worlds, relations, starts, frozen
frontend, and policy machinery:

- `episode_local`: current episode-only causal scorer;
- `persistent_v3`: fresh state-conditioned action graph per navigation world;
- `stale_v3`: one graph carried across worlds, so changed dynamics can trigger
  quarantine;
- `told_all`: evaluation-only identity oracle;
- `random`: blind action control.

The identity adapter saw only learned event tensors and one-hot opaque actions.
Coordinates, place symbols, verifier state, relation names, and scoring truth
remained outside that adapter. The fixed-size event buffer marks tracks born
after a merge as missing evidence; it never pads missing history with a zero
event.

## Result

Four worlds, twenty-step episodes, six relations, six starts, and 40 exploration
episodes per world produced:

| block | episode-local | persistent v3 | stale v3 | told-all | random |
| --- | ---: | ---: | ---: | ---: | ---: |
| trained relations | 0.472 | 0.276 | 0.267 | 0.709 | 0.263 |
| held-out relations | 0.385 | 0.211 | 0.211 | 0.595 | 0.211 |

The persistent arm was **not** an improvement over the episode-local scorer:
`-0.196` trained and `-0.174` held out. It abstained on 95.3% of trained
steps and 100% of held-out steps after the merged/born-track evidence caused a
safe quarantine. Confidently wrong assignments were correspondingly low
(1.9% trained, 0% held out), but abstention dominated behavior. This is a
useful safety signal and a failed behavioral promotion signal.

The stale model also quarantined once and had 0% confident wrong assignments
on held-out relations. That does not establish recovery: the current composed
loop does not yet contain the preregistered fresh-rerender recovery audit.

## Decision

Do **not** promote v3, freeze it, or spend the reserved holdout. The bounded
fixture result remains a mechanistic signal, but this real navigation
composition exposes an applicability gap: the graph currently treats common
track birth/merge history as missing evidence and then has no productive
relearning path inside the episode horizon.

Next work is to improve that applicability without weakening abstention:
fresh pixel rerenders with explicit crossing and occlusion, action-shuffled and
missing-evidence controls, corrupted-memory and reversal recovery, a fresh
learner comparison, and stable-prefix accounting. Only a preregistered
development pass that beats `episode_local` without increasing confident errors
can justify reconsidering the holdout.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.integrated_navigation_v3
```

The companion `integrated_navigation.json` contains the full per-task and
per-relation rows. `persistent_identity_v3_navigation.json` contains the
composition claim boundary and the matched-arm summary.
