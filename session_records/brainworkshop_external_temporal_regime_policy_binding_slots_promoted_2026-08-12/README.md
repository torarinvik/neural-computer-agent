# Opaque binding-routed residual slots — promoted boundary

Status: `PROMOTED` as a narrow multi-capability routing result.

The geometry-only detector cannot distinguish two bindings with the same
current/incoming relational structure. `GatedResidualRegimePolicyBank` adds a
versioned opaque binding-context contract: each context key selects an
independent zero-initialized residual slot while the frozen detector remains
the fallback. Slot A is trained first from fresh scalar utilities; slot B is
then trained separately. No semantic slot name, task ID, or earlier example is
provided to the policy.

| seed | A partial after A | B partial before B | A partial after B | B partial after B | stable/disjoint after B |
| ---: | ---: | ---: | ---: | ---: | --- |
| 17 | `0.9844` | `0.0391` | `0.9688` | `0.9375` | `1.0000/1.0000` and `0.9844/1.0000` |
| 18 | `0.9453` | `0.0391` | `0.8828` | `0.8516` | `1.0000/1.0000` and `1.0000/1.0000` |

Across both seeds, slot B remained byte-identical while slot A learned, and
slot A remained byte-identical while slot B learned. The base detector,
controller, and event encoder stayed frozen; routes selected distinct slots;
stable keep and disjoint replacement were retained for both bindings; and
144 online scalar utilities per seed were consumed with zero replay. After
each fresh capability passed its retention probe, its slot was frozen. Further
updates to frozen slots were rejected, and a third allocation was rejected at
the configured two-slot capacity.

The `rejected_overadaptation_seed-17.json` calibration used 96 updates per
slot and is retained as a warning: new partial replacement reached 1.0, but
stable keep collapsed for the adapted slots. Slot growth therefore still needs
capacity-aware stopping and consolidation.

The promoted lifecycle run additionally builds copy-on-write replacement
candidates. An unsafe candidate is rejected without mutation; a verified
candidate reuses slot B's physical capacity for binding C. Binding A remains
retained, binding B is evicted, and binding C reaches at least `0.8984`
partial replacement while stable/disjoint behavior remains retained.

This promotes isolated opaque binding routing, not autonomous binding
discovery, arbitrary skill composition, unrestricted slot growth, or general
continual learning.

Reports and accounting:

- `seed-17.json`
- `seed-18.json`
- `rejected_overadaptation_seed-17.json`
- `sample_efficiency_ledger.json`
