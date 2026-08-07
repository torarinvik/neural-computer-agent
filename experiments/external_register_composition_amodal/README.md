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

## Promoted narrow rung (2026-08-07)

The full promotion ladder now passes on seeds 69316 and 69317. Stable
composition mastery is `4,096` verifier bits for the inherited register path
versus `8,192` for the matched fresh learner on both seeds (`2.0x`
fresh-over-inherited). Reverse retention, reward-shuffled, missing-evidence,
exact reload, checksum-corruption, frozen-parent, and zero-replay gates all
pass. This promotes bounded factorized register composition and positive
transfer, not general continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/external_register_rendered_factorized_composition_promoted_v1_2026-08-07/`.

## Three-instruction pressure test

`train_three_instruction.py` extends the same frozen-parent protocol to an
opaque reverse -> complement -> rotate program and its reversed-order control.
The first short rung was correctly rejected as undertrained: reverse retained
at `0.9219`, but complement and rotate were below the `0.8` mastery gate and
triple composition reached only `0.6172`. The next rung acquired all three
primitives without replay and retained them after the third instruction:
reverse `0.9961`, complement `0.9648`, and rotate `0.9180`.

The triple composition decoder nevertheless reached only `0.6758`, while a
fresh three-instruction learner reached `1.0000` and the reversed-order control
reached `0.6836`. Stable composition mastery was not reached in `114,688`
unique verifier bits; the fresh learner crossed the same threshold at
`16,384` bits. Reward shuffling (`0.4531`), missing evidence (`0.5000`),
exact reload, checksum corruption rejection, frozen-parent equality, and
zero replay controls behaved as expected.

This is a decisive diagnostic, not a promotion. The bottleneck has moved from
primitive acquisition and retention to reusable serial execution at depth
three: the frozen external interpreter preserves the learned instructions,
but a newly trained decoder cannot reliably route the resulting triple-chain
state. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_three_instruction_rejected_v1_2026-08-07/`.
