# Suffix-window credit arm (pre-registration, 2026-08-03)

## Hypothesis

The accepted missing-evidence frontier still has a position-shaped credit
bottleneck: later span-11 queries are substantially weaker than the early
queries. A generic suffix window over the fresh target stream may concentrate
scalar-outcome updates on the residual positions that need them, while the
existing parent preserves the earlier positions.

This is a stream-coordinate intervention only. Positions are not exposed as
semantic labels to the controller, and the controller still receives learned
events plus opaque attempted actions and scalar outcomes.

## Pre-registered arm

- parent: `artifacts/checkpoints/span11_missing_evidence_rehearsal_seed996047.pt`;
- fresh target: 128 span-11 mixed-operation lifetimes;
- protected rehearsal: 128 span-10, 128 span-9, and 128 blank span-11 lifetimes;
- train only fresh target query positions `[4, 11)`; all protected rows remain
  available to the retention penalties;
- 32 epochs, batch size 512, learning rate 0.0005;
- binary complement and outcome-conditioned critic losses, gate/logit
  protection 0.1, no new input read or semantic branch;
- corrected audit first at 256 lifetimes, then only if the screen passes at
  1,024 and 4,096 lifetimes.

## Promotion criteria

The arm must pass acquisition, positive paired causal contribution, old-span
retention within two points, blank and full-memory-reset controls within five
points of binary chance, and replication. A short-run gain with a failed blank
control is a rejection, regardless of its objective score.
