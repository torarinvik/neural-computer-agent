# Within-lifetime dynamics reversal (2026-08-16)

Status: **development diagnostic; not a holdout and not a promotion**.
Perceptual complexity is held fixed. The agent explores one six-place ring,
then the two controllable directions reverse without notification while the
same place-event protocol continues.

## Arms

| arm | update behavior |
| --- | --- |
| recovery | reusable operator clears the model on a known-cell contradiction, then probes and replans |
| mixed-model control | keeps contradictory transition counts and carries the old model forward |

Both arms receive the same eight pre-change episodes, twelve post-change
episodes, and five fresh evaluation prefixes. The threshold is normalized return
0.75 held at every later measured prefix. The verifier emits one binary arrival
outcome per step; accounting separates verifier bits, logical lifetimes,
optimizer updates, replay, wall time, decision latency, stable bits, and
retention.

## Development result

All three replicates agree:

| arm | stable bits to threshold |
| --- | ---: |
| rebuild on contradiction | **384** |
| mixed-model control | 512 |

The recovery operator therefore shortens the stable post-change prefix by 25%
in this small diagnostic. It is a mechanism signal, not a claim about robust
online reversal in the rendered runtime; the reversal is synthetic and the
model reset policy is deliberately explicit.

## Decision and next step

Keep contradiction-triggered invalidation as a reusable operator candidate. Do
not promote it yet. The next escalation should add perceptual aliasing while
keeping reversal out of the same run, so only one difficulty axis changes.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.dynamics_reversal
```
