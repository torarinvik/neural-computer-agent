# External fast-weight plasticity: bounded primitive promotion

Date: 2026-08-09

Seeds: `69316`, `69317`

The repository now exposes `ExternalFastWeightPlasticity`, an independently
versioned memory-side delta rule. It reads learned opaque query/value tensors
and updates an external per-capability fast-weight matrix only when scalar
verifier evidence is present and positive. The controller and the plasticity
rule parameters remain unchanged during the acquisition stream.

## Result

Both seeds passed the bounded primitive pressure test:

- source and target associations reached the stable readout gate in `1`
  verifier bit each;
- the source state remained unchanged while the target state was acquired;
- failed outcomes and missing evidence made no computation-state write;
- tensor-only state persistence was exact;
- the rule parameter digest remained unchanged;
- replayed examples: `0`.

Per seed, the audit accounted for `18` unique verifier bits and `16` logical
lifetime updates. This promotes an isolated external associative-plasticity
primitive. It does not promote general continual learning, arbitrary new
computation, compression, or positive transfer to a new task: the source and
target states were independently addressed and the target began from a fresh
state.

The next test must connect this state to a learned capability adapter and
compare its no-replay learning curve with a matched fresh learner, while
retaining protected source capabilities. Evidence summaries are in
`report_seed69316.json` and `report_seed69317.json`.
