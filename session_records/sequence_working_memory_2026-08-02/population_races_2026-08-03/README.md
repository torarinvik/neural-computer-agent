# Complement-slot population races — 2026-08-03

This experiment tests whether a small population of independently seeded
successor-slot learners gives a more reliable retention-safe promotion than a
single seed. Every arm receives the same freshly generated target stream
(`--data-seed`) but a different model/optimizer seed. A private verifier-side
audit selects the best arm; the selected arm then receives a larger untouched
audit, a reset-memory control, and a matched outcome-shuffled control. Only a
full-audit pass is promotable.

The learner still receives only controller-visible latent features, opaque
attempted actions, and scalar attempted-action outcomes. Correct answers,
operation labels, and audit labels remain verifier-side. The population is a
reliability mechanism, not free sample efficiency: three clones cost roughly
three training exposures, so it must earn its cost through fewer unsafe runs
and more reliable promotion.

## Results

| Race | Target lifetimes | Data seed | Arm seeds | Selected arm | Full complement | Causal gain | Span 9 Δ | Span 10 Δ | Reset | Shuffled | Decision |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | 1,024 | 93783 | 93783/93784/93785 | 93785 | 66.81% | +16.28 pp | −1.97 pp | −1.61 pp | 50.00% | 51.68% | **promoted** |
| B | 1,024 | 93787 | 93787/93788/93789 | 93789 | 67.64% | +16.90 pp | −1.58 pp | −1.22 pp | 50.00% | 50.50% | **promoted** |
| C | 2,048 | 93791 | 93791/93792/93793 | 93791 | 69.88% | +19.05 pp | −2.67 pp | −1.63 pp | 50.00% | 52.17% | **rejected** |

The 1,024-lifetime recipe therefore replicated population selection twice on
disjoint sensory streams. Both promoted children pass the causal, complement
accuracy, reset, shuffled, and two-point old-skill retention gates. The
2,048-lifetime race improved the new-task score but failed the span-nine
retention gate, so it is deliberately not promoted.

The private reset check is advisory at 256 examples because its estimate is
too noisy (the arms commonly land near 0.40). The full audit always enforces
the 0.45–0.55 reset band. Retention is expressed in percentage points; the
two-point gate is applied as a ±0.02 fraction internally.

## Accounting and next frontier

Population selection is now a credible robustness tool for this adjacent
primitive, but it has not yet demonstrated a lower verifier-bit cost than a
well-chosen single seed. The 2,048 rejection also confirms that scaling target
data without a stronger context-selective plasticity constraint can amplify
interference. Keep the two promoted 1,024 children as capability checkpoints;
do not scale blindly to 4,096. The next high-ROI fork is to measure whether a
retention-aware population or promotion/rejection rule improves *accepted
capability per verifier bit*, with the same shuffled, blank, reset, causal,
and old-skill retention audits.

All JSON reports and private/full audit summaries are in this directory. The
promoted checkpoints are local, intentionally ignored weight files:

- `artifacts/checkpoints/complement_population_winner_seed93785.pt`
- `artifacts/checkpoints/complement_population_winner_seed93789.pt`

The matched shuffled controls and the rejected 2,048 candidate are also kept
under `artifacts/checkpoints/` for local follow-up.
