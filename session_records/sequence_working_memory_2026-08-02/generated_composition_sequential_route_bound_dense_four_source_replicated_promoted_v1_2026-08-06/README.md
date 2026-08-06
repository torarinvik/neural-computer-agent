# Four-source dense expansion with opaque route binding

This audit addresses the rejected dense expansion control. A single routed
neural artifact grows from two to five composition slots while each newly
arrived source is trained without replaying earlier examples. External memory
provides an opaque slot-eligibility mask at execution time: an alias may use
the slots that existed when it was admitted, while the new alias may use the
new slot and all earlier slots. The controller receives no source ID, task
label, grammar, or raw protocol format.

The full sequence is source `0`, then `2`, `3`, and `4`; target `1` is acquired
after the four-source artifact is stable. The runtime-private grammar is the
same five-program grammar used by the slot-isolated control. The mask binding
is versioned as `optional_opaque_external_slot_mask_v1` in the external
composition contract.

Both full replicas passed every gate:

| seed | source behavior after reload | inherited target bits | fresh target bits | inherited target after reload |
| ---: | --- | ---: | ---: | ---: |
| 69316 | 0.9570 / 1.0000 / 0.9375 / 1.0000 | 2,048 | 14,336 | 1.0000 |
| 69317 | 0.9805 / 1.0000 / 0.9844 / 1.0000 | 2,048 | 8,192 | 1.0000 |

The sequential stages all adopted, all prior aliases retained behavior after
reload, the four aliases resolved to one physical artifact row, the frozen
controller digest was unchanged, memory corruption was rejected, and both
alias-specific reversal/recovery tests passed. Replayed examples were zero.
The target was admitted through capacity growth and recovered independently.

This is a real gain over the rejected dense control: source `0` no longer fell
from `0.9531` to `0.6250` when source `2` was learned on new data only. The
matched two-source run is included as a causal check, and the short rung is
included as a rejected curriculum control.

The action-feedback boundary also received a numerical robustness fix. A
policy probability that underflows to exact zero is clamped to the smallest
positive representable value before it is logged as propensity, preserving the
feedback contract for long rollouts.

This promotes bounded replay-free dense slot growth with external route
binding. It does not establish unrestricted memory growth, neural compression,
arbitrary program induction, or general continual learning. The current mask
is correctness-first and now skips globally ineligible slots. A matched
seed-`69316` sparse-execution audit passed the same four-source gates with
identical source behavior, target behavior, masks, and accounting. Wall time
fell from `961.3s` to `831.4s` in the paired runs. Batch-divergent masks still
execute the union of active slots, so finer-grained grouped execution remains
a future optimization.

A Python-level per-mask subset/scatter implementation was tested and rejected
for performance: it preserved every semantic gate but took `935.2s` versus
`831.4s` for the sparse baseline. The full semantic pass and rejection record
are retained in `report_grouped_execution_semantic_pass_seed69316.json` and
`report_grouped_execution_rejected_seed69316.json`. The next attempt should
use compiled or batched grouped execution.

The opaque binding is now durable rather than trainer-local. Alias binding
metadata survives manifest save/load, growth, compaction, consolidation,
`promote()`, and `promote_view()`. A full seed-`69316` audit consumed the
binding from reloaded `ArtifactHandle` metadata and reproduced every gate and
metric. Its wall time was `1,244.6s`, so repeated retention/manifest writes
are now the main persistence bottleneck.
