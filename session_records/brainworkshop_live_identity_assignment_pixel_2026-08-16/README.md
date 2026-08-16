# Rendered-pixel identity-assignment integration (2026-08-16)

Status: **interface pixel diagnostic; not a holdout and not a promotion**.

This is the first check that the new identity seam sits behind the real
rendered frontend rather than a hidden-state swap. Eight RGB scenes were
rendered with two markers, decomposed into separate slots, encoded by a frozen
`RenderedBrainWorkshopEncoders` frontend, and passed to
`PolicyFreeAmodalLiveMachine` only as learned `AmodalEventCollection` tensors.

## Controls and result

| path | result |
| --- | ---: |
| assignment evidence with a clear margin | 8/8 opaque proposals emitted |
| no-assignment control | 8/8 opaque proposals emitted |
| equal evidence | 8/8 explicit abstentions, 0 guessed proposals |

All eight ticks carried two learned event tensors. The frontend digest was
unchanged before/after, and the curated `AgentBrain.bank` digest stayed
`07319eb1…e2e7c9`. No verifier bits, logical lifetimes, optimizer updates, or
bank writes were claimed.

## Boundary

The pixel path validates transport and fail-closed behavior only. The caller
still supplies the evidence vector; this run does **not** learn identity from
pixels, establish a behavioral return advantage, or validate the causal beam
on a rendered holdout. The next experiment must provide a temporary external
assignment artifact that computes evidence from learned event histories, then
compare its stable curve with a matched fresh/no-assignment learner.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.live_identity_assignment_pixel
```

The canonical report is `live_identity_assignment_pixel.json`; the accounting
companion is `sample_efficiency_ledger.json`.
