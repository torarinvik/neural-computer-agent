# Outcome-only composition search through the canonical runtime

This record promotes the v2 audit implemented in
`experiments/recipe_expressibility/control_flow_runtime_composed_program.py`.

The structural frontier first acquired a generic external component. The
composition stage then searched opaque ordered file-slot sequences, materialized
each candidate through the generic control-flow ABI, and admitted the first
candidate that passed a stable scalar verifier prefix. Search state was bound
to the source file-memory checksum and retained candidate identities and
aggregate quality only; verifier rows were not persisted. The accepted
composition was then routed through the canonical frozen amodal runtime.

Across seeds `17`, `18`, and `19`, with both forward and reversed physical
file order:

- composition search evaluations: `7` in all six verifier arms;
- acquired component and composed artifact held-out accuracy: `1.0000` in all
  six verifier arms;
- canonical composed route selection and execution: `1.0000` in all six
  verifier arms;
- source route and execution retention: `1.0000` in all six verifier arms;
- matched fresh composed-file route/execution control: `0.0000`;
- reward-shuffled composed route mastery: `0.0000` in all six null arms;
- optimizer updates: `0`; replayed examples: `0`;
- controller and external file memory remained frozen after admission.

The accepted sequence is an opaque behavioral candidate. In this run it was a
behaviorally equivalent factor order rather than the old hand-written
provenance order, which is expected: scalar verification establishes behavior,
not semantic factor identity. This promotes bounded outcome-only composition
search over existing files through the canonical runtime. It does not
establish arbitrary program induction, unrestricted memory growth, or general
continual learning. The complete report is `report.json`; the compact result
and accounting are in `report_summary.json` and
`sample_efficiency_ledger.json`; checksums are recorded in `SHA256SUMS`.
