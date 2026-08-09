# Promoted context-isolated external calibration rung

Two seeds (`69801`, `69802`) show a narrow continual-learning capability on
the memory-side evidence boundary. A frozen
`ExternalTransitionEvidenceEvaluator` was trained on a source verifier
regime. During the target phase, only the scalar calibration state addressed
by a distinct opaque context was updated from one deterministic verifier bit
at a time.

The target held-out accuracy improved from `0.477`/`0.496` to
`0.926`/`0.928`, while source accuracy stayed at `1.0` for both seeds. The
source calibration slot and base evaluator were byte-stable, the controller
received zero parameter updates, the wrong context did not receive the target
gain, and the calibration state round-tripped exactly through its external
payload.

The target phase used 256 unique verifier bits, 256 optimizer updates, and
zero replayed target examples per seed. Source evaluator pretraining used
1,024 rows for 500 updates; its repeated rows are accounted for in the JSON
reports and are not misrepresented as replay-free evaluator learning.

This promotes context-isolated external calibration, not general continual
learning. Contexts are supplied as opaque tensors, growth is append-only, and
there is no learned context discovery, capacity management, consolidation, or
compression yet.
