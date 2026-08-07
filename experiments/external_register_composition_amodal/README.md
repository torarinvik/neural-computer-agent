# External register composition pressure test

`train.py` tests the factorized `ExternalCapabilityRegisterMachine` on the
actual rendered sequence-memory stream. A frozen parent controller emits
learned event tensors and opaque intentions. Two external instruction vectors
learn reverse and complement separately with separate decoders; the machine
and instructions are then frozen while a fresh decoder learns the held-out
`complement_reverse` composition from the final register.

The first sub-minute run on 2026-08-07 used 96 updates per acquisition stage.
It was rejected as undertrained: reverse was `0.6172`, complement `0.6250`,
composition `0.6250`, and reward-shuffled composition was also `0.6250`.
Those numbers do not test the composition gate because the primitives had not
reached mastery. The report is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_rendered_undertrained_rejected_v1_2026-08-07/`.

The next run should increase only the acquisition budget to the registered
next rung. It must not be promoted unless both primitive retention and the
held-out composition clear stable-prefix gates, with fresh-learner,
reward-shuffled, missing-evidence, exact-reload, corruption, and frozen-core
controls.

## Recurrent-context follow-up (2026-08-07)

The first failure was architectural: the register seeded once and ignored
later events. `ExternalCapabilityRegisterMachine` now owns a recurrent
external context that reads each active learned event and writes the working
register before register-only instruction execution. With paired
counterfactual credit, two seeds reached composition `0.9844/0.9805`, reverse
retention `0.9844/0.9688`, reward-shuffled `0.4336/0.2891`, and fresh matched
composition `0.9492/0.8750`. Exact reload and frozen-parent checks passed.

This is retained as a replicated composition signal, not a promotion: stable
prefix bits, missing-evidence, corruption, and the full fresh control ladder
remain outstanding. Evidence is in
`session_records/sequence_working_memory_2026-08-02/external_register_rendered_factorized_composition_replicated_signal_v1_2026-08-07/`.
