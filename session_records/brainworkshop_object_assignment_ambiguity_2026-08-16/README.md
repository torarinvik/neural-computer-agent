# Action-conditioned assignment under appearance collision (2026-08-16)

Status: **development diagnostic; not a holdout and not a promotion**.
This is the follow-up to the variable-object-count audit. It removes the
unique-appearance shortcut while keeping the objects separately visible: the
controlled object and a distractor emit the same opaque appearance symbol,
cross by swapping ring positions, and never merge because their separation is
kept odd.

## Method

Each frame exposes an unordered pair of `(appearance, position)` events plus
the ordinary action log. The controller-facing trackers never receive latent
lifetime labels or the verifier's controlled position. The causal beam keeps
several assignments and scores a candidate controlled track by whether its
signed displacement matches the preceding action. The controls use only the
first event or nearest continuity. The distractor is either independent or
responds to the action on a fixed 35% of steps.

| condition | distractor | purpose |
| --- | --- | --- |
| independent collision | fixed independent motion | appearance collision and crossing |
| approximate collision | action response on 35% of steps, otherwise independent motion | calibrated causal ambiguity |

The normalized identification score must remain at least 0.75 at every later
prefix to count as stable. Seed blocks are disjoint across prefixes, and the
record reports unique verifier bits, logical lifetimes, optimizer updates,
replay, wall time, latency, stable bits, and retention.

## Development result

| condition / arm | stable bits (replicate 1 / 2 / 3) | final score range |
| --- | ---: | ---: |
| independent / causal beam | **144 / 144 / 144** | 0.970–0.991 |
| independent / nearest | none | 0.135–0.212 |
| independent / appearance | none | 0.486–0.531 |
| approximate / causal beam | **144 / 144 / 144** | 0.957–0.976 |
| approximate / nearest | none | 0.283–0.457 |
| approximate / appearance | none | 0.446–0.528 |

The action-conditioned assignment signal survives both crossings and a
partially responsive distractor. Continuity alone often stays on the wrong
lifetime after a swap, and the appearance control is near chance because the
symbols are identical. This is evidence for a causal assignment operator, not
evidence that the rejected persistent self model is safe: the beam is a small,
explicit two-object diagnostic and has not been integrated into the live
runtime.

## Decision and next step

Retain causal assignment as a candidate replaceable tracking operator and keep
the appearance/nearest controls in the regression suite. Do not promote or
admit it. The next step should be a live amodal vertical-slice test with this
operator behind the event boundary, followed by a reactive-target/other-agent
test only after identity error and abstention are measured in that runtime.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.object_assignment_ambiguity
```

The canonical report is `object_assignment_ambiguity.json`; the companion
ledger is `sample_efficiency_ledger.json`.
