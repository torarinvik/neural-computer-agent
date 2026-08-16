# Variable object count, birth/death, and occlusion (2026-08-16)

Status: **development diagnostic; not a holdout and not a promotion**.
Dynamics reversal and perceptual aliasing were held out of this run. The only
new environment pressure is that the number of visible objects changes through
birth, death, and temporary missing evidence.

## Method

Each frame exposes only opaque appearance symbols and positions. The verifier
keeps latent lifetime state so it can score the next position even when the
object is occluded; lifetime identifiers and latent positions never enter a
tracker. The persistent tracker carries a bounded track and infers one of the
four synthetic ring velocities. Controls reinitialize every frame or replace a
missing track with position zero.

| condition | stream change | expected pressure |
| --- | --- | --- |
| stable | two objects, no missing frames | baseline tracking |
| lifecycle | up to five objects with random births/deaths | variable count |
| occluded | lifecycle plus 25% frame-level occlusion | missing evidence |

The normalized position score must remain at least 0.75 at every later prefix
to count as stable. Every arm records verifier bits, logical lifetimes,
optimizer updates, replay, wall time, decision latency, and stable bits.

## Development result

| condition / arm | stable bits (replicate 1 / 2 / 3) | final score range |
| --- | ---: | ---: |
| stable / persistent | **248 / 248 / 248** | 0.968 |
| lifecycle / persistent | **395 / 415 / 445** | 0.952–0.954 |
| lifecycle / reinitializing | none | 0.000 |
| occluded / persistent | **408 / 472 / 414** | 0.872–0.900 |
| occluded / zero-missing | none | 0.647–0.688 |
| occluded / reinitializing | none | 0.000 |

Persistent state remains above threshold when objects appear and disappear,
and it recovers latent positions during occlusion. The zero-filled missingness
control fails the occluded gate, showing that retaining a track is doing the
work rather than merely attaching a state slot. The stable/lifecycle zero
control is intentionally uninformative because those conditions have no
occlusion.

This is not identity induction: every synthetic lifetime has a distinct opaque
appearance symbol. It therefore validates bounded temporal persistence under
variable count, not robust association when appearances collide or objects
cross. Nothing is admitted to the curated bank.

## Decision and next step

Keep a replaceable track/memory boundary that can abstain or carry a bounded
track through missing evidence. Reject zero-imputation as a missingness
strategy. Do not promote this finite event diagnostic. The next pressure test
should remove the unique-appearance shortcut with crossing distractors and
appearance collisions while retaining the same birth/death and occlusion
protocol.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.object_lifecycle
```

The canonical report is `object_lifecycle.json`; the accounting companion is
`sample_efficiency_ledger.json`.
