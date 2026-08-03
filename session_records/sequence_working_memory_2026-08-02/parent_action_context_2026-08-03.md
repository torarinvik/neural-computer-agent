# Parent-action context experiment — 2026-08-03

The next retention-control hypothesis is now implemented in the controller and
replay collector. A successor skill slot can read the inherited controller's
current action vector before its own residual is applied. This is a generic
confidence/context signal, not an operation label or verifier answer: the
intention is to let the plasticity gate distinguish contexts where the frozen
parent is already confident from contexts where a new residual may be useful.

The first tiny run used **raw parent action logits**. It was deliberately
bounded at 512 target lifetimes and audited independently:

| Arm | Complement | Zeroed-slot | Causal gain |
| --- | ---: | ---: | ---: |
| parent-action logits, seed 93770 | 51.94% | 50.48% | +1.46 pp |

The result is below the registered +5 pp causal bar, so this arm is not
promoted. The inherited span-nine/span-ten skills remained usable, but the raw
logit scale is a poor context representation for this gate. The full reports
are `complement_parentaction_512_93770.json` and
`complement_parentaction_512_93770_audit.json` in this directory; the
unpromoted checkpoint is retained only for reproducibility.

## Next bounded fork

Repeat the same 512-lifetime experiment with the inherited action vector
normalized to a probability simplex. Use a freshly regenerated replay buffer
with the identical feature layout, then apply the same causal, reset, shuffled,
cue-blank, and span-nine/span-ten retention audits. A single promising seed is
not enough: promotion still requires an independent replicate and the existing
two-point retention gate.

This keeps the investigation small and reversible. If normalized context also
fails, stop expanding the feature list and return to the task-agnostic gate or
population-selection alternatives; do not scale the complement task to 2,048
lifetime exposures while retention remains seed-sensitive.
