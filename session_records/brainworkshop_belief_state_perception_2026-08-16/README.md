# Belief-state perception under aliased events (2026-08-16)

Status: **development diagnostic; not promoted**.

The fixture has six latent places but only five learned event symbols: places
0 and 3 are perceptually aliased. The controller-facing evidence is only the
opaque event symbol and opaque action. A verifier-side latent place is used
only to score predictions.

`BeliefStatePerception` keeps weighted context hypotheses rather than forcing a
single current-symbol interpretation. It uses the full recent event/action
context when present, backs off to shorter learned contexts when an event is
missing, and abstains below a confidence floor. Missing evidence is represented
as `None`; it is never converted to a zero-valued event.

## Result

With 16 training episodes, 40 evaluation episodes, and 20% missing events:

| arm | accuracy when acting | coverage | expected correct rate |
| --- | ---: | ---: | ---: |
| merged current-symbol, clean | 0.897 | 1.000 | 0.897 |
| hard history, clean | 0.961 | 1.000 | 0.961 |
| belief, clean | 1.000 | 0.915 | 0.915 |
| hard history, missing | 0.950 | 0.606 | 0.576 |
| belief, missing | 1.000 | 0.823 | 0.823 |

The belief arm improves the useful missing-evidence rate from 0.576 to 0.823
while retaining perfect accuracy on the predictions it emits in this fixture.
That is a perception/abstention signal, not yet evidence of closed-loop
navigation or experience savings.

## Boundary and next gate

This artifact is not wired into the rendered navigation controller yet. The
next development gate is to carry the same explicit belief/mask through the
learned-event bus and persistent identity adapter, then test fresh pixel
rerenders with occlusion, aliasing, corrupted history, and a shuffled-action
control. The reserved holdout remains untouched.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.belief_state_perception
```
