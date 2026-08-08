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

## Promoted read/execute snapshot frontier (2026-08-07)

The depth-three failure isolated a mutable-state problem, so the register now
exposes `observe_register()` and `read_execute_register()`. Observation state
persists learned events and feedback; instruction execution runs on a
transient register snapshot and cannot write its result back into the evidence
store. The legacy in-place `step_register()` remains available for explicit
compatibility, while `step()` and this harness use the snapshot path.

The original two-instruction regression passes on both seeds with stable
inherited composition at `4,096` verifier bits versus `8,192` for fresh
(`2.0x` fresh-over-inherited). The three-instruction reverse -> complement ->
rotate rung also passes on both seeds: inherited mastery is `8,192` versus
`16,384` fresh on seed 69316 (`2.0x`) and `4,096` versus `12,288` on seed
69317 (`3.0x`). All primitive-retention, order-sensitivity, reward-shuffled,
missing-evidence, exact-reload, checksum-corruption, frozen-parent, and
zero-replay gates pass.

This promotes the read/execute state boundary and a bounded three-instruction
compositional growth result. It does not establish arbitrary program
induction, unrestricted memory growth, or general continual learning. The
curated reports and accounting ledger are in
`session_records/sequence_working_memory_2026-08-02/external_register_read_execute_promoted_v1_2026-08-07/`.

## Four-instruction nonlinear boundary (2026-08-07)

The next runtime-grammar program was
`reverse -> adjacent_xor -> complement -> prefix_parity`. The canonical
factorized low-rank interpreter retained the simpler primitives but plateaued
at adjacent-XOR `0.7734`, and its inherited four-step composition did not beat
fresh. A factorized FiLM candidate raised primitive retention to `0.8125` but
tied fresh composition. A low-rank-plus-zero-initialized-FiLM hybrid retained
all four primitives at `0.9414`, yet its serial composition remained unstable.
Deeper shared blueprint pretraining made primitive retention `0.9336` while
collapsing inherited composition to `0.4805`.

All runs passed persistence, causal, frozen-parent, and zero-replay controls,
but none passed the positive stable-transfer and composition gates. This is
archived as a rejected diagnostic in
`session_records/sequence_working_memory_2026-08-02/external_register_four_instruction_rejected_v1_2026-08-07/`.
The production default remains factorized low-rank read/execute. The next
architecture task is a nonlinear operator with an explicit compositional
invariant, not more unconstrained depth or blueprint updates.
A short composition-aware blueprint probe was also rejected: retention was
`0.7813` and inherited composition was `0.7344` versus fresh `0.9844`.

## Bounded residual operator control (2026-08-08)

The next operator candidate normalizes the register before a low-rank update,
bounds the proposal with `tanh`, and lets each opaque instruction choose a
feature-wise residual gate. This directly tests whether bounded state change
improves serial compositional stability. The matched four-instruction audit
passed every primitive-retention, shuffled-outcome, missing-evidence,
reload, corruption, frozen-parent, and zero-replay control on both seeds, but
failed positive transfer: inherited composition stabilized at `20,480` and
`24,576` verifier bits versus fresh at `16,384` and `8,192`. Final inherited
composition was `0.8867` on both seeds.

Fresh-outcome shared-blueprint pretraining raised inherited composition to
`0.9883` and `0.9688`, with all safety controls still passing, but fresh
learners reached the same threshold in `4,096` bits on both seeds while the
inherited path required `8,192`. It is therefore rejected as a transfer
mechanism. The result supports bounded execution stability, not learned
blueprint reuse or genuine new computation. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_four_instruction_bounded_residual_rejected_v1_2026-08-08/`.
