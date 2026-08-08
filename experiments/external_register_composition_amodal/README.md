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

## Held-out new-procedure transfer (2026-08-08)

The next audit acquired four verifier-private source procedures, froze the
shared interpreter, and trained only a new opaque instruction code and decoder
for a fifth unseen procedure. A matched fresh interpreter received the same
target outcomes. The depth-four rung was rejected because source acquisition
was under the mastery floor. Reducing only program depth to two made the
source gate valid on seed `69316`: inherited target accuracy was `0.9531`,
fresh was `0.9844`, and stable cost tied at `12,288` bits. Seed `69317` left
one source at `0.7617`; increasing only source updates to `384` repaired that
gate, but inherited target still required `12,288` bits versus `8,192` fresh.

All valid controls passed: source retention, target mastery, reward-shuffled
rejection, missing-evidence rejection, exact reload, corruption rejection,
frozen parent, and zero replay. The strict held-out-transfer gate therefore
remains rejected. This localizes the next bottleneck: one new opaque code is
not a reusable blueprint for an entire unseen procedure. New composition from
already learned instruction data is the next higher-ROI test. Evidence is
archived in
`session_records/sequence_working_memory_2026-08-02/external_register_heldout_new_procedure_transfer_rejected_v1_2026-08-08/`.

## Promoted held-out composition transfer (2026-08-08)

The composition formulation succeeds. Four opaque instructions are acquired
sequentially for `reverse`, `adjacent_xor`, `complement`, and `prefix_parity`.
After the interpreter is frozen, a new held-out order
`prefix_parity -> complement -> reverse -> adjacent_xor` is acquired using
the existing instruction vectors and a fresh decoder; no new instruction code
is added. A matched fresh interpreter learns the same composition directly.

With the registered 384-update source rung, both seeds promote. Inherited
composition reaches stable mastery in `8,192` verifier bits on both seeds,
while fresh learners require `12,288`, a replicated `1.5x`
fresh-over-inherited transfer ratio. Final inherited accuracies are `0.8867`
and `0.8750`. Source retention, composition mastery, reward-shuffled null,
missing-evidence, exact reload, corruption rejection, frozen-parent, and
zero-replay gates all pass. The matched 256-update source control is retained:
it ties or fails the transfer gate, and one seed fails source mastery.

This promotes reusable compositional computation for one held-out order, not
arbitrary program induction or general continual learning. Evidence is
archived in
`session_records/sequence_working_memory_2026-08-02/external_register_heldout_composition_transfer_promoted_v1_2026-08-08/`.

## Multi-order composition pressure test (2026-08-08)

The same four frozen instructions were reused across three held-out orders.
Seed `69316` transferred all three orders at the 128-update composition rung,
with inherited stable costs of `8,192`, `4,096`, and `4,096` bits versus fresh
costs of `12,288`, `12,288`, and `16,384`. Seed `69317` transferred two of
three; the first order tied fresh at `8,192` bits. Doubling only composition
updates for that seed preserved the tie while the other two orders transferred.
All mastery, source retention, shuffled-null, missing-evidence, reload,
corruption, frozen-parent, and zero-replay controls passed.

This is strong partial evidence for reusable multi-order composition, but the
strict replicated all-target promotion gate is rejected. The remaining
bottleneck is order-robust transfer efficiency, not basic composition
execution. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_multi_heldout_composition_rejected_v1_2026-08-08/`.

## Held-out primitive transfer (2026-08-08)

The strict new-computation test acquired four source primitives, froze the
interpreter, and trained a fifth unseen `rotate` instruction from fresh
outcomes. Both seeds mastered all sources and the new target and passed every
retention, shuffled-null, missing-evidence, reload, corruption, frozen-parent,
and zero-replay control. Transfer nevertheless failed: seed `69316` tied
fresh at `8,192` stable bits, while seed `69317` required `24,576` versus
`4,096` fresh. This is a clean boundary result: learned instruction
composition transfers, but the current shared operator does not yet invent a
new primitive computation. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_heldout_primitive_transfer_rejected_v1_2026-08-08/`.

## Continual shared-blueprint update control (2026-08-08)

The next intervention kept the shared operator trainable on later source
primitives and added an anchor penalty to its previous weights, while freezing
all old instruction codes. Anchor weights `1.0` and `10.0` were tested across
both seeds. The new `rotate` target often transferred quickly (for example,
`4,096` versus `8,192` bits), but source primitives fell below mastery before
target acquisition in every run. The safety controls themselves passed, but
the source-retention gate rejected the intervention. This rules out naïve
whole-blueprint online updates; future operator-family learning needs isolated
meta-state or protected subspaces rather than a scalar anchor. Evidence is
archived in
`session_records/sequence_working_memory_2026-08-02/external_register_continual_blueprint_anchor_rejected_v1_2026-08-08/`.

## Protected operator-family and code-prior control (2026-08-08)

The `factorized_protected_meta` operator isolates a zero-initialized,
code-conditioned residual family while freezing the mastered base operator.
It preserves all four source primitives across both seeds, but new `rotate`
transfer ties or loses to fresh. Initializing the new code from the mean of
the mastered instruction vectors does not improve the result: seed `69316`
ties at `8,192` bits and seed `69317` needs `12,288` versus `4,096` fresh.
This rejects the simplest protected operator-family and code-geometry repairs.
The next design must add an explicit expandable computation basis rather than
expecting a new vector to elicit an unseen primitive from a fixed operator.
Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_protected_meta_code_prior_rejected_v1_2026-08-08/`.

## Expandable external computation basis (2026-08-08)

The register now exposes an append-only `ExternalRegisterComputeBasis` slot
interface. A slot is independently addressable external computation capacity:
it receives only the persistent register and an opaque instruction vector,
and adding one does not resize the controller or alter existing instruction
codes. The read/execute and in-place APIs accept memory-side slot bindings,
and the configuration schema is versioned as external-register v3.

The first unseen-`rotate` pressure test added one fresh slot only for the new
primitive. At the established 384/256 rung, all four source primitives were
retained and the target reached 0.9922 accuracy; shuffled outcomes, missing
evidence, reload, corruption, frozen-parent, and zero-replay controls passed.
However, stable-prefix promotion failed while the matched fresh learner
reached 8,192 stable verifier bits. This promotes the architecture’s ability
to add isolated computation safely, not positive transfer or general
continual learning. The short curriculum probe is also retained as a
non-promotion diagnostic. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_expandable_basis_probe_2026-08-08/`.

The basis boundary also exposes verifier-gated memory-side selection: fresh
candidate probes may select an opaque existing slot, while a failed probe
requests growth. Mastered slots can be frozen, and an unpromoted newest slot
can be rolled back without touching earlier slots.

## Promoted basis reuse (2026-08-08)

The two-seed reuse probe promoted the next boundary. A first `rotate`
instruction learned one external basis slot; a second fresh opaque
instruction reused that frozen slot without replay. Reused stable costs were
`4,096` verifier bits on both seeds, versus `12,288` and `8,192` for matched
fresh learners. All admission, mastery, retention, frozen-slot, shuffled-null,
missing-evidence, reload, frozen-parent, and zero-replay gates passed. This
promotes bounded reuse of mastered computation, not arbitrary new computation
or general continual learning. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_basis_reuse_probe_2026-08-08/`.

The distinct-operation follow-up reused the frozen `rotate` slot for a fresh
`global_parity` instruction. Both seeds retained the old capability and
passed all safety controls, but only one transferred faster than fresh
(4,096 vs 8,192 bits); the other was slower (16,384 vs 8,192). Cross-operator
reuse is therefore rejected as a strict promotion and remains the next
bottleneck.

The executable route-vs-grow follow-up now passes both seeds: the slower case
appended and trained slot 1, recovering 8,192 stable bits while retaining the
old capability at 1.0000; the faster case reused slot 0 at 4,096 stable bits,
also retaining the old capability at 1.0000. This promotes efficiency-aware
capacity routing for this pressure test, while general cross-operator priors
remain open.

Efficiency-aware admission now separates the two cases: seed `69316`
requests growth when reuse costs `16,384` versus `8,192` fresh bits, while
seed `69317` reuses at `4,096` versus `8,192`. This closes the correctness-only
admission gap; the append-and-retrain grow branch is covered by the executable
route-vs-grow result above.

The next implementation adds `ExternalRegisterBasisCompatibilityPrior`, a
replaceable opaque candidate screen that learns slot ordering from attempted
scalar outcomes. It remains screening-only; fresh stable verifier evidence
still controls reuse versus growth.

The two-seed held-out opaque screening audit reduced mean trials on admissible
queries from `2.074` to `1.116` and from `2.143` to `1.029`, while preserving
the exact verifier-admissibility rate. This promotes bounded screening
efficiency, not verifier-free admission or general cross-operator transfer.
Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_compatibility_prior_audit_2026-08-08/`.

The real acquisition follow-up trained three source primitives into separate
basis slots, updated the prior from their actual verifier outcome matrix, and
routed a held-out `prefix_parity` candidate through the live register
scheduler. Both seeds correctly requested growth because no existing slot
passed fresh verification; no replay was used. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_real_basis_acquisition_2026-08-08/`.

Executing the growth slot reached high final accuracy and preserved every
source capability, but both seeds failed stable-prefix promotion and failed
the shuffled-outcome rejection control. This growth result is rejected; the
next bottleneck is causal credit/verification dependence during new-slot
acquisition.

The causal repair uses `attempted_bce` for new-slot training, exposing only
delivered scalar outcomes. Shuffled-training controls then collapse to `0.4766`
and `0.5000`, while normal target accuracy remains `0.9375` and `0.9063` with
source retention intact. Stable-prefix promotion still fails, so growth is
not promoted; the credit-path repair is retained.
