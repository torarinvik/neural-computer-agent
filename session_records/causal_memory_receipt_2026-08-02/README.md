# Causal memory receipts and learned plasticity

## Result

The next memory boundary is now implemented and passed a causal probe:
verified outcomes can be attributed to the physical row that actually supplied
a read, even when unequal write strengths redirect ordinary retrieval.

The probe used two latent rows. The exact content match had low admission
strength (`0.05`); a slightly less similar row had strength `1.0`. Re-resolving
the query with the strength prior credited the wrong row and selected the stable
row for replacement. A receipt captured at read time credited the exact content
row and selected the decoy for replacement instead.

| policy | stable-row volatility | replacement row | stable evicted |
|---|---:|---:|---:|
| ordinary re-resolution | `1.0000` | `0` | yes |
| physical receipt | `0.1678` | `1` | no |
| shuffled receipt | `1.0000` | `0` | yes |

The receipt-corrected memory survived save/reload byte-for-byte. Shuffling the
receipts destroyed the protection behavior, so the result is causal rather
than a fixed-row shortcut.

## Learned-controller integration

The promoted eight-feature controller was then evaluated against physical
`DiskLatentMemory` histories with unequal admission strengths. No controller
weights were changed during this audit. The learned plasticity policy remained
causal and preserved the old behavioral skills:

| policy | update accuracy | stable eviction | stable volatility | decoy volatility |
|---|---:|---:|---:|---:|
| receipt-attributed | **97.07%** | **4.69%** | 0.344 | 0.836 |
| ordinary re-resolution | 96.29% | 28.91% | 0.452 | 0.851 |
| shuffled receipts | 88.09% | 59.38% | 0.633 | 0.547 |

Binary mapping and four-rule retention both remained 100%. The audit consumed
7,680 physical verifier bits and completed in 6.12 seconds. All gates passed:
receipt accuracy ≥95%, stable eviction ≤10%, receipt better than ordinary
re-resolution, shuffled-receipt loss ≥6 points, retention, and the five-minute
cap.

## Implementation

- `PersistentMemory.read_with_receipt()` returns values, confidence, and the
  physical top-1 row index.
- `PersistentMemory.record_outcomes_from_receipts()` applies verifier outcomes
  without resolving retrieval a second time.
- `DiskLatentMemory` exposes both methods without changing the existing read
  API.
- The new targeted tests cover unequal-strength redirection and repeated-row
  outcome accounting.
- The executable probe is
  `experiments/unified_cognitive_controller/probe_causal_memory_receipt.py`.
- The learned-controller integration audit is
  `experiments/unified_cognitive_controller/audit_receipt_volatility_controller.py`.

No semantic task IDs, correct actions, or row labels enter the learner-facing
interface. The receipt is generic provenance metadata, analogous to a CPU
register or memory address.

## Existing learned-plasticity baseline

The earlier full-controller volatility experiment remains the stronger learned
baseline: with only one new volatility coefficient trained, held-out valid
replacement reached `98.83%` and accuracy `97.07%` versus a `97.36%` oracle.
Shuffling volatility fell to `48.83%` valid replacement, and reversing outcome
order caused `99.6%` stable-row eviction. Binary and four-rule behavioral
retention also passed. The short eight-update smoke in this session was
intentionally undersized and failed its gates; it is retained as a warning that
the phase-transition budget matters.

## Next experiment

Train a fresh small volatility/plasticity head directly on
receipt-attributed outcomes with unequal admission strengths. Keep the
controller frozen, compare against the promoted head and an oracle, and use
the same retention and shuffled-receipt gates. Only then consider letting the
main controller predict the scalar online.
