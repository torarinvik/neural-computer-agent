# Reusable external control-flow composition through the canonical runtime

This record promotes the audit implemented in
`experiments/recipe_expressibility/control_flow_runtime_composed_program.py`.

The structural frontier first acquired a generic transfer loop from scalar
verifier outcomes. A second external file was then composed with that acquired
file using the typed control-flow ABI, with jump targets relocated and
terminal halts transferred safely. The composed artifact was admitted through
scalar verifier evidence and routed through the canonical frozen amodal
runtime. The controller received no program IDs, target labels, or protocol
fields.

Across seeds `17`, `18`, and `19`, with both forward and reversed physical
file order:

- acquired component held-out accuracy: `1.0000` in all six verifier arms;
- composed artifact held-out accuracy: `1.0000` in all six verifier arms;
- canonical composed route selection and execution: `1.0000` in all six
  verifier arms;
- source route and execution retention: `1.0000` in all six verifier arms;
- fresh composed-file route/execution control: `0.0000`;
- reward-shuffled composed route mastery: `0.0000` in all six null arms;
- optimizer updates: `0`; replayed examples: `0`.

The composition primitive is domain-general: it relocates internal and
terminal jump targets, rejects incompatible counter widths and ambiguous
internal halts, and returns an ordinary checksummed control-flow file. It does
not add a task-specific reasoning branch.

This promotes bounded reusable external computation through the canonical
runtime. It does not establish arbitrary program induction, unrestricted
memory growth, or general continual learning. The complete report is
`report.json`; accounting is in `sample_efficiency_ledger.json`; checksums are
recorded in `SHA256SUMS`.
