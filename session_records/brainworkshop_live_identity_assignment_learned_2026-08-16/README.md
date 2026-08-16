# Learned-event identity evidence (2026-08-16)

Status: **interface diagnostic; not a holdout and not a promotion**.

This run replaces the previous caller-supplied evidence placeholder with a
versioned external artifact. Eight RGB scenes were rendered with a moving
marker and a stationary marker, decomposed into separately bound slots, and
encoded by a frozen `RenderedBrainWorkshopEncoders` frontend. The artifact
received only the resulting learned event history and opaque action-feature
history, then supplied a score to the existing fail-closed identity gate.

## Result

| path | result |
| --- | ---: |
| learned event/action evidence | 5/8 opaque proposals; 3 warm-up/tie abstentions |
| passive zero-evidence control | 8/8 explicit abstentions |
| constant-action control | 8/8 explicit abstentions; all evidence `[0, 0]` |

After the minimum four-frame history, the artifact selected slot 0 whenever
the score margin cleared the 0.15 gate. The frontend digest and curated
`AgentBrain.bank` digest were unchanged. No verifier bits, logical lifetimes,
optimizer updates, or bank writes were claimed.

## Boundary

This establishes only that a replaceable external artifact can turn bound
learned event histories into opaque identity evidence and preserve explicit
abstention. The position sequence and one-hot action features are synthetic
feeder inputs, not learner-visible coordinates or a behavioral holdout. Slot
order never crosses in this diagnostic. It does not establish causal identity
learning, transfer, retention, promotion, or a return advantage. The next
experiment should use fresh pixel rerenders with crossings/occlusion and
matched shuffled-action and fresh-learner controls before any promotion.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.live_identity_assignment_learned
```

The canonical report is `live_identity_assignment_learned.json`; the
accounting companion is `sample_efficiency_ledger.json`.
