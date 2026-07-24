# Tiny-experiment breakthrough — 2026-07-23

## Main result

The frozen integrated controller's intermediate recurrent states contain the
ingredients needed for temporal binding, but the trained write/memory path
does not preserve a decodable rule.

Two independent 1,024-lifetime disposable-probe runs found:

| seed | first identity | rewarded identity | learned composition | true-ingredient composition | shuffled mean |
|---:|---:|---:|---:|---:|---:|
| 41 | 96.78% | 86.33% | 83.40% | 100% | 46.58% |
| 97 | 95.80% | 84.96% | 81.45% | 100% | 50.81% |

The 512-lifetime run was already positive (91.41%, 76.95%, and 73.63%),
showing a clear data-scaling curve. The direct raw-write/compact/recall probe
at 128 lifetimes stayed at or below the 53.9% majority baseline, including
after 2 and 4 demonstrations. Intermediate-state localization also stayed at
baseline for the derived rule.

## Interpretation

This is representation evidence, not a behavioral milestone. The capability
exists in the recurrent representation and is learnable by a disposable
supervised binder. The remaining problem is integration/credit assignment:
the in-agent write path has not learned to encode the composition into memory.
No 3-minute training run is justified yet.

## Next tiny experiment

Use the successful 1,024-lifetime two-stage diagnostic as a teacher only for a
throwaway write-path probe: train a small head on the actual raw-write output,
then test raw-write → compact row → recall with 1,024 held-out lifetimes. Keep
the agent weights frozen and discard the head. If raw writes can be made
decodable, the next repair is the writer interface; if not, the signal must be
connected to the writer before any consolidation change.

## Integration follow-up

A one-epoch, 1,024-lifetime event-binder-only supervised bootstrap ran for
about three minutes. Gradients were healthy and the residual stayed bounded,
but the temporary write-rule head averaged 49.44%. A follow-up 512-lifetime
raw-write probe reached 49.41% held out versus a 52.54% majority baseline;
compact and recalled rows were also at baseline. This is a bounded negative:
the representation is learnable, but one short write-path bootstrap does not
transfer it into memory.
