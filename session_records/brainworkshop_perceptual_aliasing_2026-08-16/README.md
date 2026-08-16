# Perceptual aliasing with history-conditioned state (2026-08-16)

Status: **development diagnostic; not a holdout and not a promotion**.
This changes one axis only: six latent places remain deterministic, but latent
places 0 and 3 render to the same learned event symbol. The controller-facing
models receive only observed symbols and opaque actions.

## Arms

| arm | state key |
| --- | --- |
| merged | current observed symbol and action |
| history | previous observed symbol, previous action, current symbol, action |
| corrupted history | same history key with the previous event replaced by missing/corrupt data |

The verifier scores next-observation predictions on fresh traces. A score is
stable only when normalized accuracy remains at least 0.95 at every later
prefix. Latent positions and the alias map are used only to generate the
observation stream and scoring truth.

## Development result

The current-symbol table never reaches a stable prefix: it plateaus around
0.88–0.90 because it merges incompatible transitions. The history-conditioned
model reaches the stable threshold in **920, 1840, and 2760 verifier bits** in
the three replicates. Corrupted history never passes. The result is a
mechanistic signal that short event/action context is necessary for this alias,
not a claim that the full runtime has solved belief-state planning.

Accounting is serialized per arm for unique verifier bits, logical lifetimes,
optimizer updates, replay, wall time, prediction latency, stable bits, and
retention. Nothing is admitted to the curated bank.

## Decision and next step

Keep a history/belief-state boundary as the next architecture candidate, but
do not promote it from this finite event audit. The next escalation should add
variable object count and occlusion while leaving the alias mapping and
dynamics fixed.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.perceptual_aliasing
```
