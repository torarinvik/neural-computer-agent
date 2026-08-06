# External capability composition diagnostic — 2026-08-06

Status: rejected as a general composition promotion; retained as a decisive
partial result.

This audit learned two separate memory-side programs (`complement4` and
`reverse4`), froze them, and serially composed them through the new
`ExternalCapabilityPipeline`. A fresh decoder then learned the novel
`complement_reverse4` target from fresh rendered episodes and scalar outcomes.
The shared controller, frontend, and parent output path stayed frozen. No old
examples were replayed.

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| composed stable bits | 2,048 | 14,336 |
| composed held-out accuracy | 0.9492 | 0.8828 |
| blank-pipeline accuracy | 0.5508 | 0.6641 |
| prior fresh-pipeline stable bits* | none | none |
| prior fresh-pipeline accuracy* | 0.5703 | 0.6758 |
| shuffled-outcome accuracy | 0.5547 | 0.5273 |
| zero-first-program accuracy | 0.8477 | 0.8984 |
| zero-second-program accuracy | 0.5117 | 0.5273 |

The composed pipeline clearly beats the blank pipeline on both seeds, and the
second primitive is causal on both. The first primitive is causal only on seed
69316; on seed 69317 it is redundant or mildly harmful. The prior fresh-pipeline
control was invalid: a harness `no_grad` scope prevented its pipeline weights
from receiving gradients. Its values are retained for provenance but cannot be
used to calculate a fresh-learner transfer ratio. These results support a
useful frozen-program composition signal, but not robust arbitrary composition
or positive transfer against a fresh learner.

\* Invalid prior control; the corrected rerun is pending.

Persistence and safety controls passed on both seeds: exact pipeline/decoder
reload, behavior-preserving reload, checksum corruption rejection, unchanged
controller digest, and zero replay. The implementation cost was high
(`430.35s` / `463.84s`), so the next step is a cheaper shared-rollout
diagnostic before increasing task breadth. That diagnostic rehydrated the
seed-69317 artifact without primitive replay: normal execution scored `0.8828`,
while hiding raw events from downstream programs scored `0.5195`, a `0.3633`
drop. This confirms that the current composition result still relies heavily
on later programs rereading the original event stream.

Full reports and accounting are in `report_seed69316.json`,
`report_seed69317.json`, `event_visibility_audit_seed69317.json`, and
`sample_efficiency_ledger.json`.
