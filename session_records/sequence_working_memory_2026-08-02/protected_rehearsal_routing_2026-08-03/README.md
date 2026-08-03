# Protected-rehearsal routing correction (2026-08-03)

## Implementation finding

The reward-buffer trainer historically marked only persisted disk rows as
`replay_mask=True`. Freshly collected span-nine/span-ten rehearsal rows were
therefore treated as fresh target data, so replay weights and residual/gate
retention penalties silently skipped the very old skills they were intended
to protect. This record accompanies the opt-in `--protect-rehearsal` fix in
`train_sequence_reward_buffer.py`.

The new protected mask leaves the historical default unchanged. When enabled,
it marks all non-target rows in the freshly collected buffer as protected while
preserving persisted-row provenance. The learner still receives only latent
features, opaque attempted actions, and scalar attempted-action outcomes.

## Tiny causal tests

All runs started from
`artifacts/checkpoints/complement_population_fourth_slot_seed93871.pt` and
used two distractors, 64 target lifetimes, 64-lifetime span-nine/span-ten
rehearsal, eight epochs, and the same 256-wide appended slot.

| arm | protected | source weight | residual penalty | causal gain | span-9 Δ | span-10 Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| matched baseline | no | 0 | 0 | +1.21 pp | −1.48 pp | −1.05 pp |
| provenance-only | no | 10 | 0 | +1.38 pp | −4.38 pp | −4.92 pp |
| corrected gentle arm | yes | 1 | 0.01 | **+2.13 pp** | −1.56 pp | −0.59 pp |
| residual 0.003 | yes | 1 | 0.003 | +0.67 pp | −2.00 pp | −0.12 pp |
| residual 0.03 | yes | 1 | 0.03 | +0.71 pp | −1.95 pp | −0.12 pp |
| strong gate penalty | yes | 0 | gate 10 | +0.00 pp | +0.08 pp | +0.00 pp |

The corrected gentle arm is the best local Pareto point, but it remains below
the registered +5-point promotion bar. It justifies a short escalation, not a
promotion.

## Short escalation and adversarial control

With 256 target lifetimes and 128-lifetime rehearsal, the real-outcome arm
reached only **+1.53 pp** causal gain, with span-nine/span-ten changes of
−1.45/+0.02 pp. The matched outcome-shuffled arm reached **−7.48 pp** and
−2.97/−2.60 pp retention changes. The negative shuffle result confirms that
the small real signal depends on reward correspondence, but the signal does
not scale into a promotion at this budget.

## Decision

The mask bug is fixed and tested, and the correction improves the diagnostic
interpretability. The protected-rehearsal recipe is not yet a successful
span-eleven learner. Do not scale it blindly. The next frontier remains a
selective, task-agnostic credit route that can preserve old behavior while
using the observed outcome stream; any candidate must pass causal gain,
shuffle, blank/reset, reversal, and retention gates.

All JSON reports in this directory are the complete evidence for these arms.
