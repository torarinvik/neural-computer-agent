# Corrected external capability composition — 2026-08-06

Status: rejected as a replicated promotion; retained as a seed-sensitive
diagnostic.

This is the corrected rerun of the external capability composition audit. The
fresh pipeline arm now receives gradients through its recurrent external
programs; the earlier reports in the neighboring `...composition_rejected...`
directory are retained only as provenance because their fresh arm was inside a
`no_grad` scope.

Two independently acquired programs (`complement4` and `reverse4`) were frozen
and composed before a fresh decoder learned `complement_reverse4`. The shared
controller stayed frozen, all composition episodes were fresh, and replay was
zero.

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| composed stable bits | 2,048 | 14,336 |
| composed accuracy | 0.9492 | 0.8828 |
| fresh stable bits | 6,144 | 6,144 |
| fresh accuracy | 1.0000 | 1.0000 |
| fresh-over-composed ratio | 3.000× | 0.429× |
| blank-pipeline accuracy | 0.5508 | 0.6641 |
| zero-first accuracy | 0.8477 | 0.8984 |
| zero-second accuracy | 0.5117 | 0.5273 |

The composed pipeline beats the blank control on both seeds and all reload,
checksum, frozen-core, shuffled-outcome, and zero-replay controls pass. But the
transfer effect reverses across seeds: one seed shows a strong 3× gain, while
the other is slower than a fresh pipeline and the first primitive is not
causal. This fails the replicated promotion gate. The immediate bottleneck is
safe selection of inherited composition state: a candidate must earn a fresh
held-out learning-curve advantage before it is installed.

Full reports and accounting are in `report_seed69316.json`,
`report_seed69317.json`, and `sample_efficiency_ledger.json`.
