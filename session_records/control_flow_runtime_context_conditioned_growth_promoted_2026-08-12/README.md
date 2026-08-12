# Sequential context-conditioned external-file growth

This record promotes the short-rung `ControlFlowProgramAmodalRuntime` audit
implemented in
`experiments/recipe_expressibility/control_flow_runtime_context_conditioned_growth.py`.

One frozen amodal controller addressed four protected generic control-flow
files through an external checksummed context-route table. Four learned event
contexts were acquired sequentially. Training used only the scalar verifier
outcome for the selected opaque file, with no replay of earlier contexts. A
final phase changed context zero's correct file and tested reversal recovery.

Results across seeds `17`, `18`, and `19`, with both forward and reversed
physical-file order:

- all six verifier arms: `1.0000` retention after growth and reversal;
- fresh-bank controls: `0.2500`;
- reward-shuffled nulls: `0.2500` paired fresh accuracy;
- controller optimizer updates: `0`;
- replayed examples: `0`;
- scalar verifier bits per arm: `160`;
- external contexts: `4`; protected files: `4`.

The first implementation failed reversal because lifetime-average promotion
made earlier failures permanent. The promoted implementation adds a persisted
recovery streak, allowing a previously bad slot to become preferred after a
fresh stable success run without clearing unrelated context evidence.

This is bounded context-conditioned external-memory growth and reversal
recovery. It does not establish unrestricted memory growth, content search,
arbitrary new computation, or general continual learning. The independent
reward-shuffled arms are retained as negative controls and are not expected to
promote.

The complete machine-readable report is `report.json`; its SHA-256 is recorded
in `SHA256SUMS`.
