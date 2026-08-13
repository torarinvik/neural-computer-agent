# Outcome-only program acquisition through the canonical runtime

This record promotes the audit implemented in
`experiments/recipe_expressibility/control_flow_runtime_acquired_program.py`.

The structural frontier first acquired a generic counter-machine program from
scalar verifier outcomes. The acquired file was then admitted beside a
protected source and decoys, and a frozen amodal controller routed opaque
intentions through the canonical control-flow runtime. Route evidence was
learned from selected-file scalar outcomes only; the controller and external
files were not updated.

Across seeds `17`, `18`, and `19`, with both forward and reversed physical
file order:

- acquired-program held-out accuracy: `1.0000` in all six verifier arms;
- canonical acquired-file route selection: `1.0000` in all six verifier arms;
- canonical acquired-file execution: `1.0000` in all six verifier arms;
- source route and execution retention: `1.0000` in all six verifier arms;
- matched fresh acquired-file route/execution control: `0.0000`;
- reward-shuffled route mastery: `0.0000` in all six null arms;
- optimizer updates: `0`; replayed examples: `0`;
- controller, files, route reload, and checksum gates: all passed.

The opaque codec derives its bounded counter input from the complete opaque
intention. This keeps the route challenge non-degenerate when one coordinate
lands at zero; it does not expose a target amount, program identity, or
privileged state to the controller.

This promotes bounded outcome-only structural acquisition followed by
canonical frozen-controller execution and route learning. It does not
establish arbitrary program induction, unrestricted memory growth, or general
continual learning. The complete report is `report.json`; accounting is in
`sample_efficiency_ledger.json`; checksums are recorded in `SHA256SUMS`.
