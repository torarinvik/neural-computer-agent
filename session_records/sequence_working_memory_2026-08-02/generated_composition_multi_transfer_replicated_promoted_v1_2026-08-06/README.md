# Multi-source external transfer and verified finite-capacity compaction

This audit tests the next boundary after single-file replay-free transfer:
two protected external artifacts are compared as initializations for a new
runtime program, then compacted into one physical memory row with separate
opaque executable views. The frozen parent controller receives no raw program
IDs, correct actions, or replayed examples.

The source programs were:

1. `reverse -> adjacent_xor -> complement -> prefix_parity`
2. `global_parity -> reverse -> adjacent_xor -> rotate`

The target was `prefix_parity -> global_parity -> rotate -> complement`.
Each replica trained source 0, source 2, and a fresh candidate independently
from fresh target outcomes. A stable-prefix selector chose one unique target
candidate before admission. The source bank was capacity two; consolidation
rewrote it transactionally to one row with `source_0` and `source_2` views,
verified both views after reload, and then grew to capacity two to admit the
target without evicting either protected source.

| seed | source-0 stable bits | source-2 stable bits | inherited target stable bits | fresh target stable bits | selected | source-0 reload | source-2 reload | target reload |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 69316 | 10,240 | 2,048 | 6,144 | 10,240 | source 0 | 0.9336 | 1.0000 | 1.0000 |
| 69317 | 12,288 | 2,048 | 4,096 | 10,240 | source 0 | 0.9492 | 1.0000 | 1.0000 |

Both replicas passed every gate: source address separation, source mastery and
protection, unique inherited transfer, one-row savings, independent view
reload, retention protection, target growth, target reload, frozen parent
digest, and zero replay. The fresh-over-inherited stable-bit ratios were
`1.667x` and `2.500x`.

This promotes multi-source bounded external transfer and logical storage
compaction. It is not neural weight compression, unrestricted memory growth,
arbitrary program induction, or general continual learning.

The authoritative per-seed reports are `report_seed69316.json` and
`report_seed69317.json`. The short rung rejected a source view at `0.625`
fresh retention and was not promoted.
