# Synchronized external control-flow growth — narrow promotion

This three-seed in-repository audit validates the new
`ControlFlowProgramAmodalRuntime.admit_program_verified()` lifecycle. A
scalar verifier prefix admits one new opaque control-flow file through a
copy-on-write transaction. The transaction synchronizes executable memory,
context-route evidence, router capacity/state, and per-file execution
counters while the amodal controller remains frozen.

All seeds passed the evidence-backed and router-capacity growth arms. The
source file remained intact, the target file executed, memory reload was exact,
rejected candidates did not mutate the source, and the controller state was
byte-identical before and after growth. Each seed charged 18 unique verifier
bits, with zero optimizer updates and zero replayed examples.

This is a narrow external-memory lifecycle promotion. It does not establish
unrestricted memory growth, autonomous arbitrary program induction, or general
continual learning.

The runnable audit is
`experiments/recipe_expressibility/control_flow_runtime_program_growth.py`.
