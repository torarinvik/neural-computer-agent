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

## Normalized follow-up

The matched probability-simplex follow-up used the same 512 target lifetimes,
freshly regenerated 2,432-transition replay buffer, penalties, and audit:

| Arm | Complement | Zeroed-slot | Causal gain | Span 9 Δ | Span 10 Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| normalized parent probabilities, seed 93771 | 53.29% | 50.94% | +2.35 pp | −6.76 pp | −12.53 pp |

This also misses the +5-point causal bar and violates the two-point retention
gate. It is rejected, not promoted. The exact reports are
`complement_parentaction_probability_512_93771.json` and
`complement_parentaction_probability_512_93771_audit.json`; the checkpoint and
normalized replay buffer are retained only to make the negative result
reproducible.

## Decision

Both raw-logit and normalized-parent-action context fail. Normalizing the
signal does not make it selective; it makes the old-skill interference worse.
Stop expanding this feature list and return to a genuinely context-selective
plasticity mechanism or an explicit verifier-side promotion/rejection
population. Do not scale the complement task to 2,048 lifetime exposures
while the retention gate remains seed-sensitive.
