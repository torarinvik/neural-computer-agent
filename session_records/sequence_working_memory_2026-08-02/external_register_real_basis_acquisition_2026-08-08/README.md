# Real external-register basis acquisition — 2026-08-08

Three opaque source primitives (`rotate`, `global_parity`, `complement`) were
trained into independent external basis slots. Their fresh verifier outcome
matrix was used to update the compatibility prior once, then a held-out
`prefix_parity` acquisition was routed through the live register scheduler.

Both seeds produced distinct source outcome rows, preserved fresh-verifier
admission as the authority, and correctly found no passing existing basis for
the unseen target. The target therefore requested growth rather than being
incorrectly reused. No replayed examples were used.

This promotes real multi-slot opaque acquisition and no-false-admission
behavior. It does not yet demonstrate positive transfer to a genuinely new
primitive; the correct result here is verified growth.

## Growth execution follow-up — rejected

The selected slot-3 growth branch was then trained on held-out `prefix_parity`.
Both seeds reached high final target accuracy (`0.9766` and `0.9453`) and
retained all source capabilities, with the old basis digests unchanged.
However, neither reached a stable-prefix threshold, and shuffled-outcome
controls remained above the rejection floor (`0.9922` and `0.9531`). The
growth result is therefore rejected for promotion. The next bottleneck is
causal credit/verification dependence in new-slot acquisition, not retention
or append-only capacity.

The causal follow-up switched only new-slot training to `attempted_bce`, so
the optimizer received delivered scalar outcomes rather than verifier-private
correct-action utilities. Shuffled-training controls then collapsed to
`0.4766` and `0.5000`, confirming causal dependence. Normal target accuracy
remained `0.9375` and `0.9063`, with source retention intact, but stable-prefix
promotion still failed. The corrected result remains rejected for stability,
while the credit-path repair itself is retained.

## Staged scalar-credit follow-up — rejected

The next rung used a two-stage curriculum: a short-span warmup followed by
full-span target training, with source retention checked after warmup and
again after growth. Seed `69316` reached `0.8125` final target accuracy and
seed `69317` reached `0.8320`; both retained all source skills, rejected
shuffled training, and left old basis digests unchanged. Neither produced a
stable full-span prefix, and the reward-shuffled control remained too strong.
The staged curriculum therefore does not promote new-skill acquisition yet.
It establishes that the current failure is not simply catastrophic forgetting:
the remaining blocker is reliable scalar-credit learning and control
sensitivity on the full target distribution.

The audit also corrected propensity accounting: sampled actions use an
epsilon-smoothed behavior policy, and the exact smoothed propensity is now
carried in the opaque action record and used by policy-gradient credit.

## Exact-propensity REINFORCE follow-up — rejected

Replacing attempted-outcome BCE with exact-propensity REINFORCE did not
produce stable full-span acquisition in either seed (`69316`: `0.8281`,
`69317`: `0.7969` final accuracy; both had no stable prefix). Source
retention and shuffled-training rejection remained intact. This path is
therefore retained as a valid baseline, not promoted as the solution.

## Fixed-baseline policy-gradient follow-up — rejected

A second scalar-only policy-gradient path used an action-independent `0.5`
verifier baseline and a small entropy floor, avoiding the batch-centered
advantage used by the earlier REINFORCE path. It still failed stable full-span
acquisition (`69316`: `0.8164`; `69317`: `0.7813`) while retaining source
skills and rejecting shuffled training. The estimator is retained as a valid
option, but the bottleneck is now localized to routing credit through the
new basis and decoder representation.

## Eligibility-trace scalar credit follow-up — rejected

Discounted return-to-go credit was added so each delivered outcome could
credit earlier selected actions through the sequence. It performed worse than
the one-step estimators (`69316`: `0.7930`; `69317`: `0.7656` final accuracy;
neither had a stable prefix). Retention and shuffled-training rejection still
passed. Temporal credit accumulation is therefore not promoted as the default
new-slot learner.

## Basis-focused acquisition follow-up — rejected

The new slot was given a dedicated scalar-credit phase with the decoder
frozen between warmup and final joint training. The focus phase remained near
chance-plus (`69316`: `0.6563`, `69317`: `0.6563`), and final target accuracy
was `0.8477` and `0.8086`, with no stable prefix. Source retention and causal
shuffle controls passed. Isolating basis updates does not solve acquisition;
the next bottleneck is the representational interface between the frozen
controller/register and a freshly added computation slot.

## Near-identity fresh-slot initialization — rejected

A fresh basis slot was initialized as a gated near-identity residual to avoid
perturbing the established register before learning. It did not improve the
causal growth rung (`69316`: `0.7578`; `69317`: `0.8164`), and one seed lost
source retention. The production initialization remains unchanged; the
interface problem requires a learned representation path, not only safer
initialization.

## Bounded learned-event window follow-up — positive mechanism, not yet promoted

The external register v4 now carries a bounded window of standardized learned
event tensors and masks it through quiet ticks. New computation slots can read
that window alongside the register and opaque instruction, while raw modality
formats remain outside the boundary. In the two-seed causal audit, final
target accuracy rose to `0.9180`/`0.9063`, source retention was
`0.9570`/`0.9336`, shuffled-training remained below threshold, and old basis
digests were unchanged. The full-span progress checkpoints still failed the
stable-prefix gate, so this is a promoted interface mechanism and a positive
direction—not yet a promoted continual-learning capability.

The longer 256-update follow-up reached final accuracy `0.9980`/`0.9766`,
but full-span checkpoints later fell to `0.5684`/`0.6133`. This confirms that
the event window improves access to task information without solving protected
continual acquisition; final-sample accuracy must not be used as the mastery
criterion.

An attempted length-9 window retained the full rendered episode but performed
worse: full-span progress ended at `0.6289`/`0.6055` and both candidates were
rolled back. The shorter length-4 window remains the empirically stronger
representation; more context is not automatically more useful.

The event-window state also now preserves the entire prior window on quiet
ticks instead of shifting and duplicating its final token. The corrected
two-seed audit produced the same all-active scores (`0.9180`/`0.9063`) and
rolled both unstable candidates back, confirming that this was a state
correctness repair rather than an unverified capability gain.

## Verifier-gated consolidation/rollback

External growth now uses the shared retention gate transactionally. An
unstable candidate is removed from the newest slot, its opaque instruction is
restored, and previously mastered slots remain untouched; only a candidate
that passes stable-prefix mastery and the retained-capability floor is frozen
as consolidated. A rollback smoke audit rejected an unstable candidate with
candidate prefix minimum `0.5` and confirmed `rollback_applied: true`.

## Action-independent actor–critic follow-up — rejected

A trainer-only value head was added to estimate expected scalar verifier
success from the learned register, reducing policy-gradient variance without
correct-action labels. Across two seeds it reached `0.8320`/`0.8281` with no
stable prefix. Source retention and causal controls passed, but it did not
beat attempted-outcome BCE; the value head remains an optional experiment,
not the default learner.

## Action-conditioned Q-credit follow-up — rejected

A trainer-only action-conditioned verifier model learned scalar outcomes for
the actually attempted opaque action and supplied a learned advantage to the
policy. Checkpoint progress improved modestly to `0.7500`/`0.7539`, but final
accuracy was only `0.8555`/`0.8750`, with no stable prefix. This remains an
optional outcome-only baseline; attempted-outcome BCE is still stronger.

## Paired scalar-probe credit follow-up — rejected

The trainer actively obtained both opaque action outcomes for each fresh
rendered query and optimized the two scalar verifier results directly. This
removed single-action credit sparsity, but doubled verifier bits and therefore
does not constitute a single-trial continual-learning result. Across seeds
`69316` and `69317`, final target accuracy was `0.8203`/`0.8750`, while the
stable full-span checkpoints remained below threshold (`0.6563`/`0.6328` at
the final checkpoints). Source retention passed in only one seed, and both
unstable candidates were rolled back. The result localizes the remaining
bottleneck beyond scalar credit sparsity: the new computation path cannot yet
consolidate a stable capability while the old basis remains protected.

Each run used `69,632` verifier bits, `8,704` logical lifetimes, `577`
optimizer updates, and zero replayed examples. Paired probing is retained as a
valid trainer-only diagnostic, not as the default learner.

### Fixed-suite retention measurement repair

The first paired-probe reports compared source retention on different random
held-out batches. The audit now evaluates the exact same source suite before
and after growth and applies an explicit `0.02` maximum regression tolerance.
The corrected rerun measured zero regression in both seeds. This confirms that
the failed consolidation is not caused by old-slot forgetting; it is caused by
the new candidate failing stable-prefix mastery. The absolute source floor
still remains a separate promotion gate.

## No-basis-focus curriculum follow-up — rejected

Removing the dedicated basis-focus phase and going directly from short-span
warmup to joint full-span acquisition did not solve the bottleneck. Final
target accuracy was `0.7578`/`0.8438`, but neither seed achieved a stable
full-span prefix. The full-span checkpoint minima were `0.6484`/`0.6563`, and
the first seed also failed the absolute source floor. Source scores were
unchanged on the fixed retention suite. This rejects phase ordering as the
primary cause: the remaining blocker is the representational/capacity path
for a newly acquired computation, not scalar credit or basis-focus scheduling.

Each run used `61,440` verifier bits, `7,680` logical lifetimes, `513`
optimizer updates, and zero replayed examples.

## Lower-plasticity and span-3 warmup screens — rejected

Reducing growth learning rate to `3e-4` made acquisition slower (`0.6719`/
`0.8203` final target accuracy) and still produced no stable prefix. Changing
warmup from span 2 to span 3 was also worse, ending at `0.8047`/`0.9219` with
unstable full-span checkpoints. The remaining issue is not simple optimizer
overshoot or the two-to-four span curriculum jump.

## Diverse frozen-parent pretraining — useful foundation, not promoted

The parent was pretrained on held-out `reverse`, `complement`, and
`adjacent_xor` auxiliaries while `rotate` remained unseen. With 128 parent
updates, source mastery and final target accuracy improved (`0.9063`/`0.9219`),
and fixed-suite retention was exact, but stable-prefix target mastery still
failed (`0.6875`/`0.6250` minima). This is a promising foundation direction,
but not yet a continual-growth capability.

## Controller-state event boundary screens — rejected

Two optional standardized event modes were tested. Appending the frozen
controller state to the frontend event did not produce stable acquisition
(`0.6641`/`0.6563` final checkpoints). Delivering controller state alone was
worse (`0.6250`/`0.5625` minima) and damaged source mastery. A raw state dump is
not the correct bridge; the next design needs a learned task-agnostic adapter
that preserves recoverable information without exposing an unstructured
controller implementation detail.

The corrected runs include parent pretraining and growth accounting. The
ordinary state-only runs used `70,656` verifier bits and `9,216` logical
lifetimes per seed; the diverse-parent runs used `86,016` verifier bits and
`16,896` logical lifetimes, with zero replayed examples.

## Candidate learned event bridge — promising, not promoted

A zero-initialized external adapter was trained only with the new target
candidate. It maps the frontend event plus controller latent state back into
the existing standardized event width, so the frozen controller and inherited
slots remain untouched and rollback discards the adapter. One seed improved
through checkpoint `0.7656` before drifting; the other remained near `0.66`.
Final target accuracy was `0.8125`/`0.8594`, but neither seed passed stable
prefix mastery. This validates the isolation and behavior-preserving design,
but the adapter needs a more stable learning/consolidation objective before
it can become the canonical bridge.

## Wider fresh-slot capacity follow-up — rejected

Increasing the fresh compute-slot hidden width from `64` to `128` did not
solve acquisition. Full-span progress remained below stable mastery in both
seeds, with final checkpoint scores of `0.6719`/`0.7109`; source retention on
the fixed suite remained unchanged. More parameters in the same one-step
residual MLP are therefore insufficient. The next representation test should
add structured iterative state or microsteps to the new slot rather than only
increasing width.

Each run used `69,632` verifier bits, `8,704` logical lifetimes, `577`
optimizer updates, and zero replayed examples.

## Verifier-gated snapshot consolidation — positive mechanism, mixed replication

The candidate path now has a genuine staging boundary. During mutable growth,
the trainer may retain the best held-out snapshot, but that snapshot is not
admitted immediately. It is restored, treated as frozen external capability
state, and subjected to four independent post-freeze verifier probes. Admission
also requires the fixed source-retention floor; an unstable or retention-
breaking candidate is rolled back transactionally.

At a longer 512-update target budget, both seeds found a high-performing frozen
candidate and passed all four target probes: seed `69316` scored
`0.9531`/`0.9531`/`0.9453`/`0.9648`, while seed `69317` scored
`1.0` on every probe. Seed `69317` passed the inherited source floor
(`0.8281` minimum) and was promoted. Seed `69316` was correctly rejected
because one inherited source remained below the absolute floor (`0.7578`),
despite zero measured regression from its own baseline. Thus the mechanism
solves the transient-candidate admission problem in one seed, but the strict
two-seed promotion gate remains mixed and this is not yet a general continual-
learning claim.

The target training curves themselves stayed below threshold, which is
important: promotion came only from the frozen snapshot's independent
verification window, not from relabelling a drifting training trajectory.
Each run used `121,856` verifier bits, `15,360` logical lifetimes, `961`
optimizer updates, and zero replayed examples; `2,048` verifier bits were
spent on post-freeze consolidation probes.

## Attention-pooled event reader follow-up — rejected

Fresh slots were given a generic attention reader over the bounded event
window: the opaque register/instruction state formed a query, and event tokens
formed masked keys and values. This is a more structured interface than
flattening the window, but it reduced acquisition: stable-prefix checkpoint
minima were `0.6484`/`0.6250`, with final target accuracy `0.7656`/`0.7969`.
Fixed-suite source retention remained unchanged. The optional attention mode
is retained as an architectural probe, not promoted as the default path.

## Bounded microstep fresh-slot follow-up — rejected

Each new external slot was allowed two bounded residual microsteps over its
register state, while retaining the same opaque instruction and event-window
interface. This performed worse than the one-step slot: final full-span
checkpoint scores were `0.5703`/`0.5000`, with no stable prefix. Fixed-suite
source retention remained unchanged. Iterating the same residual interface is
therefore not sufficient; the next design must improve what the slot can
represent and learn from, rather than only repeating its update.

Each run used `69,632` verifier bits, `8,704` logical lifetimes, `577`
optimizer updates, and zero replayed examples.

## Online EMA stabilization follow-up — rejected

The candidate learned event bridge was trained with online exponential moving
average parameter updates (`decay=0.9`) across target warmup, basis focus, and
full-span growth. This preserved the fixed source suite exactly in both seeds,
but it did not produce stable target mastery: the full-span checkpoint paths
were `0.625`/`0.6563`/`0.6719`/`0.6484` and
`0.4219`/`0.4688`/`0.5625`/`0.5938`. Final target accuracy was `0.6641` and
`0.7266`, and both unstable candidates were rolled back. EMA therefore does
not repair the bridge's consolidation problem by itself. The bridge remains
an isolated architectural direction, while this optimizer mechanism is not
promoted.

Each run used `70,656` verifier bits, `9,216` logical lifetimes, `577`
optimizer updates, and zero replayed examples.

## Source-floor repair with longer acquisition — promoted replicated rung

The mixed snapshot result localized the remaining failure to source mastery:
seed `69316` had not retained its inherited `adjacent_xor` capability above
the absolute floor before target growth. Increasing source acquisition from
`96` to `192` updates, while retaining fixed-suite best-checkpoint selection,
repaired that failure without changing the frozen controller or replaying old
examples.

Both seeds now pass the complete promotion gate. Seed `69316` retains a
minimum source score of `0.875` and its frozen target snapshot passes probes
`1.0`/`0.9961`/`0.9961`/`1.0`; seed `69317` retains a minimum of `0.8594` and
passes `1.0` on all four probes. Both target snapshots are promoted, and all
causal controls, source-retention, zero-replay, and rollback invariants pass.
This is the first replicated rung where source retention and new target
consolidation succeed together under the frozen-controller boundary.

Each run used `224,768` verifier bits, `28,992` logical lifetimes, `1,761`
optimizer updates, and zero replayed examples. This total includes the
shuffled-training control (`65,536` bits), source-selection suite (`4,608`
bits), and post-freeze target consolidation (`2,048` bits).
This proves a robust single new-capability acquisition rung, not yet
unrestricted continual learning; the next pressure test is a second unseen
capability acquired after the first has been promoted.

## Sequential two-target acquisition — promoted replicated rung

The next pressure test appended a second opaque instruction only after the
first target had been promoted. It acquired `rotate`, froze it, then acquired
`prefix_parity` with a new basis slot, new decoder, and new event bridge. The
first target, all three sources, and the frozen controller were retained while
the second target learned from fresh outcomes only.

Both seeds promoted both targets. Seed `69316` reached target probe minima of
`0.9688` for `rotate` and `0.8203` for `prefix_parity`; seed `69317` reached
`0.8594` and `0.8750`. Every retained capability stayed above `0.8594` after
the second acquisition. Shuffled-outcome training controls stayed below
`0.516`, missing-evidence controls stayed at `0.5`, and zero replay was used.
This is the first replicated evidence that the external memory can grow by
more than one promoted capability without catastrophic forgetting in this
pressure test.

Each run used `370,176` verifier bits, `47,680` logical lifetimes, `2,912`
optimizer updates, and zero replayed examples. The next frontier is a third
sequential capability plus reversal testing, to distinguish durable growth
from a two-stage benchmark effect.

## Sequential three-target acquisition — promoted replicated rung

The next audit appended a third instruction only after the first two targets
had been promoted. It acquired `rotate`, `prefix_parity`, and then
`global_parity`, adding a fresh basis slot, decoder, and learned event bridge
for each target. Candidate restarts were allowed, but rejected candidates were
rolled back before the next attempt; no old examples were replayed.

Both seeds promoted all three targets. Every retained source and target
capability stayed above the `0.8` floor after the final acquisition. The
shuffled-outcome controls stayed at `0.4531`–`0.5156`, missing-evidence stayed
at `0.5`, and all measured fixed-suite retention deltas were zero. Seed
`69316` used `518,656` verifier bits and `4,064` optimizer updates; seed
`69317` used `809,472` verifier bits and `6,368` optimizer updates because
the first two candidate attempts were rejected before successful restarts.
Both used zero replayed examples.

The corrected pressure test uses a pixel-level sequence-order rerender rather
than the earlier no-op operation-cue flip. `rotate` and `prefix_parity` had
changed rendered labels under reversal, while `global_parity` correctly stayed
label-invariant because parity is order-invariant. The retained capabilities
remained strong under these rerenders, so this is evidence for durable
generalization—not evidence that unrestricted continual learning is solved.
The next frontier is genuinely novel composition and longer, interleaved
acquisition with bounded memory growth.

## Sequential four-target acquisition — promoted replicated rung

The fourth target was deliberately compositional: `complement_rotate` combines
the previously acquired rotation and complement factors, but was not admitted
as a previously mastered target. It received its own appended instruction,
basis slot, decoder, and event bridge after `rotate`, `prefix_parity`, and
`global_parity` had already been frozen.

Both seeds promoted all four targets. The final retained minima were `0.8359`
and `0.8516`; the new target's frozen consolidation probe minima were `0.9844`
and `0.9023`. Shuffled controls remained below `0.540`, missing evidence stayed
at `0.5`, fixed-suite retention deltas were exactly zero, and the frozen parent
was unchanged. Seed `69316` used `665,088` verifier bits and `5,216`
optimizer updates; seed `69317` used `955,904` verifier bits and `7,520`
optimizer updates because it required one extra restart for the first target.
Both used zero replayed examples.

This is evidence that the isolated external memory can retain four promoted
capabilities and acquire a novel factor composition. It still does not test
interleaved learning of multiple mutable capabilities or unrestricted program
induction; those are the next bottlenecks.

## Interleaved mutable-capability acquisition — promoted replicated rung

The next audit trained two unseen targets, `complement_rotate` and
`prefix_parity`, in alternating local updates after the three source slots had
been frozen. Each target kept independent optimizer state and its own
instruction, basis, decoder, and learned event bridge. A paired transactional
gate admitted both candidates only if both passed retention and causal controls.

Both seeds promoted both targets. Seed `69316` reached candidate accuracies
`0.9844`/`0.8984`; seed `69317` reached `0.9688`/`0.9453`. Shuffled-training
controls were `0.6172`/`0.5938` and `0.5547`/`0.5313`; missing-evidence scores
were exactly `0.5`; the corrected paired-suite retention deltas were all zero.
Each run used `406,272` verifier bits, `53,152` logical lifetimes, `3,168`
optimizer updates, and zero replay. The accounting includes the independently
trained shuffled-outcome pair.

During this work the external event bridge was also repaired: the frozen
parent remains detached, but bridge parameters now receive gradients. A
regression test proves both properties. This makes the interleaved result a
valid learned-boundary measurement rather than a fixed random bridge screen.

## Interleaved three-target acquisition — promoted replicated rung

The three-way audit trained `complement_rotate`, `prefix_parity`, and
`global_parity` in the same round-robin schedule. The source machine was built
with source instructions only; target instructions and basis capacity were
appended only after source acquisition. The operator family was the
factorized low-rank implementation, because the bounded-residual family did
not meet the source floor in this configuration.

Both seeds promoted all three concurrent targets with exact zero retention
deltas. Seed `69316` candidate accuracies were `0.9688`, `0.9063`, and
`0.9922`; seed `69317` reached `1.0000`, `0.9844`, and `1.0000`. Shuffled
controls stayed below `0.586`, missing evidence remained `0.5`, and the
source floors were `0.8086`/`0.8477`/`0.8828` and
`0.9961`/`0.9063`/`0.9688`. Each run used `570,368` verifier bits,
`74,720` logical lifetimes, `4,448` optimizer updates, and zero replay.

This is the strongest concurrent-plasticity result in this path so far. The
remaining frontier is concurrent compositional transfer with more than one
composition family, not merely more independent target slots.

## Concurrent compositional transfer — promoted replicated rung

The follow-up kept the round-robin schedule but replaced the third direct
target with a genuinely compositional candidate. The candidate received a
fresh decoder and event bridge, while its computation was a frozen chain of
the three acquired source instructions for `complement -> reverse ->
adjacent_xor`. Two fresh direct capabilities (`complement_rotate` and
`prefix_parity`) learned concurrently beside it. The composition candidate
therefore had no new instruction or basis slot and could succeed only by
reusing the learned external program chain.

Both seeds promoted all three candidates. Seed `69316` reached direct target
accuracies `0.8984` and `0.9453`, and composition accuracy `0.8281`; seed
`69317` reached `1.0000`, `0.9844`, and `0.9688`. Frozen consolidation probe
accuracies were `0.8945`/`0.9453`/`0.8047` and `0.9922`/`0.9844`/`0.9766`.
Shuffled-outcome controls stayed below `0.727`, missing-evidence scores were
`0.5`, and all source retention deltas were exactly zero. Each run used
`570,368` verifier bits, `74,720` logical lifetimes, `4,448` optimizer
updates, and zero replayed examples.

This promotes concurrent reuse of a learned frozen program chain under fresh
plasticity. It is still bounded external-memory growth: the next pressure
test is multiple independently sampled composition programs, followed by
longer sequences and transfer against matched fresh learners.

## Multiple concurrent composition programs — promoted replicated rung

The next pressure test interleaved four candidates: the two direct targets
above plus two fresh decoder/bridge candidates for independently ordered
programs, `complement -> reverse -> adjacent_xor` and
`adjacent_xor -> complement -> reverse`. Both programs executed only the
three frozen source instructions; neither received new compute capacity.

Both seeds promoted all four candidates. Direct target accuracies were
`0.9063`/`0.9453` and `0.9688`/`0.9844`. The two composition candidates
reached `0.8281`/`0.9531` in seed `69316` and `0.9531`/`0.9609` in seed
`69317`; every consolidation probe exceeded `0.855`. Shuffled controls
remained below `0.680`, missing-evidence scores were `0.5`, source retention
deltas were exactly zero, and replay was zero. Each run used `732,160`
verifier bits, `96,000` logical lifetimes, and `5,728` optimizer updates.

This strengthens the promotion from one favorable composition to multiple
ordered programs under concurrent plasticity. It remains bounded: the next
test should sample composition programs from a larger grammar and measure
transfer against matched fresh learners.

## Full three-source permutation grammar — promoted replicated rung

The composition family was expanded to all six permutations of the three
frozen source operations. The audit therefore interleaved eight candidates:
two direct capabilities and six fresh decoder/bridge candidates, one for each
program ordering. No composition candidate received a new instruction or
basis slot.

All eight candidates promoted in both seeds. The weakest consolidation probe
was `0.8711` in seed `69316` and `0.9297` in seed `69317`; shuffled controls
stayed below `0.633`, missing-evidence scores stayed below the mastery floor,
and every source-retention delta was exactly zero. Each run used `1,379,328`
verifier bits, `181,120` logical lifetimes, `10,848` optimizer updates, and
zero replayed examples.

This promotes concurrent compositional reuse across the complete finite
permutation family tested here. It is still a bounded grammar result; larger
program depth, novel primitives, and transfer against matched fresh learners
remain open.

## Matched fresh-learner transfer diagnostic — not promoted

The full-grammar audit now includes a strict matched fresh control. For each
of the six compositions, a fresh machine with three trainable instructions,
bases, decoder, and event bridge receives the same phase/update budget as the
inherited candidate. The inherited path retained its full promotion, but the
transfer gate did not pass: seed `69316` had fresh stable-prefix failure on
most programs, while seed `69317` showed mixed results when both arms crossed
the threshold. The strict positive-transfer flag was false for both seeds.

This is a useful negative result. The frozen external program bank can preserve
and expose learned capability, but we cannot yet claim that it reliably makes
learning a new output path cheaper than a matched fresh learner. The next
bottleneck is adaptation efficiency at the memory-to-decoder/event-bridge
boundary, not source-program retention. Evidence is in
`interleaved_full_grammar_transfer/`; its fresh-control accounting adds
`466,944` verifier bits per run, with zero replay.

## Mastered decoder prior diagnostic — rejected

As a targeted adaptation intervention, the two-program audit initialized each
new composition decoder from mastered source decoder 0 while keeping the
composition bridge fresh. The shuffled-outcome control received the same
initialization. The prior was not stable: seed `69316` degraded one
composition from `0.8828` to `0.8125`, while seed `69317` improved another
from `0.9141` to `0.9844`; the full transaction was rejected in seed `69316`
and accepted in seed `69317`, so no replicated positive transfer gate passed.

This rejects raw decoder-weight reuse as a general interface prior. It is
retained as an explicit opt-in diagnostic, not as the production default. The
next intervention should learn a protocol-agnostic interface representation or
adaptation rule, rather than copying action-decoder weights across skills.
Evidence is in `interleaved_decoder_prior_diagnostic/`.

## Shared outcome-trained event bridge diagnostic — rejected

The next intervention trained one reusable `AmodalEventBridge` from attempted
scalar outcomes on the three mastered source capabilities, froze it, and
provided the same bridge to new compositions and their fresh controls. Source
retention stayed unchanged, but the composition gate was seed-dependent:
seed `69316` rejected both composition candidates at `0.7578`/`0.7734`
consolidation accuracy, while seed `69317` passed at `0.8828`/`0.9258`.

This rejects a simple shared bridge prior as a general adaptation solution.
The opt-in mechanism remains useful for diagnostics, but the next design must
learn an interface prior that conditions on capability state without copying
protocol-specific weights or relying on one fixed bridge. Evidence is in
`interleaved_shared_bridge_prior_diagnostic/`.

## Capability-conditioned bridge prior diagnostic — mixed, not promoted

The next design uses shared bridge weights conditioned by an opaque aggregate
of the learned instruction/program vectors. This preserves a reusable
interface while allowing capability-specific adaptation; the context has no
hand-assigned semantic fields. Seed `69317` passed both compositions and the
strict fresh-transfer comparison, with inherited stable bits of `32,768` and
`16,384` versus fresh `40,960` each. Seed `69316` rejected both compositions
at `0.7852` and `0.7695` consolidation accuracy, and its fresh learners also
failed stable mastery.

The mechanism is promising but not replicated. It remains opt-in; the
production default and promotion claim are unchanged. Evidence is in
`interleaved_conditioned_bridge_prior_diagnostic/`.

## Conditioned bridge source-restart repair — promoted bounded rung

The mixed conditioned-bridge result was rerun with only source acquisition
robustness changed: two source restarts instead of one. Both seeds then
promoted both compositions. Seed `69316` reached consolidation probes
`0.9492`/`0.9805`; seed `69317` reached `0.9805`/`0.9453`. Source selection
floors were `0.9961`/`0.8125`/`0.9023` and `0.9961`/`0.9648`/`0.9648`, and
all source retention deltas remained exactly zero. Fresh controls still did
not reach stable mastery, so this promotes robust bounded behavior—not
positive sample-efficiency transfer. Each run used zero replay.

Evidence is in `interleaved_conditioned_bridge_restart_repair/`.

## Curriculum-matched fresh transfer — rejected

The fresh control was strengthened to acquire the same three source
primitives before learning each composition. This removes the confound that
the inherited path had mastered source programs while the prior fresh arm
had not. The result is negative for transfer: in both seeds, fresh target
adaptation reached stable mastery in `8,192` target bits for both programs,
while inherited adaptation required `8,192`/`16,384` in seed `69317` and
`16,384`/`24,576` in seed `69316`. Fresh source acquisition cost an additional
`73,728` bits per composition, but the inherited path still did not pass the
strict positive-transfer gate.

Behavioral promotion and retention remain intact. The next bottleneck is
aligning learned source-state geometry with new program/output paths, not
simply storing more source skills. Evidence is in
`interleaved_conditioned_curriculum_transfer/`.

## Canonical register readout diagnostic — rejected for transfer

The raw executed register is now separated from the decoder by an explicit,
identity-initialized `CanonicalRegisterReadout`. A single readout was trained
from scalar outcomes across the three mastered source skills, frozen, and
then supplied to two new composition decoders. Both seeds passed causal,
retention, and behavior gates with zero source regression and zero replay.

The matched fresh curriculum control did not replicate positive transfer:
seed `69316` favored the fresh path on both compositions, while seed `69317`
favored it on one and tied on the other. The readout is retained as clean
interface infrastructure and a diagnostic boundary, but the learned
source-state geometry remains the bottleneck. Output normalization alone is
not enough to make independently acquired external programs composable.

## Shared interpreter family diagnostic — strongest direction, not promoted

The next intervention changed the external execution contract itself. In
`factorized_shared_interpreter` mode, addressed basis slots resolve through
one shared factorized operator family; only the opaque instruction vector
selects the operation. This removes the independent per-slot MLP geometries
that made serial composition ill-conditioned.

The matched two-seed result is mixed. Seed `69317` passed both composition
transfer gates, with inherited stable costs of `16,384` and `40,960` bits
versus fresh costs of `57,344` and `49,152`. Seed `69316` failed transfer on
both programs and rejected one composition for mastery; source floors were
also weaker than the independent-slot baseline. Retention deltas remained
exactly zero and shuffled-outcome controls stayed causal.

This is the strongest current architectural direction, but not yet a
promoted continual-learning capability. The next bottleneck is robust source
acquisition under a shared operator family, including order sensitivity and
capacity allocation—not another decoder or readout prior.

## Shared-interpreter capacity screen — rejected

Increasing the shared factorized operator rank from `8` to `16` did not
repair the mixed result. Across both seeds, source floors weakened (including
`0.7734` and `0.8047` in seed `69316`), composition candidates became less
stable, and positive transfer was not replicated. Retention deltas remained
exactly zero and the causal controls remained valid.

This rejects raw rank expansion as the next intervention. The shared
interpreter’s main asymmetry is now acquisition order: the first source skill
trains the shared operator weights, while later skills mostly train their
opaque instruction vectors. The next diagnostic will test order sensitivity
and seek a way to acquire the shared operator contract without privileging
the first capability.

## Acquisition-order screen — rejected adjacent-first order

The audit now exposes source acquisition order as an explicit control. Moving
the harder `adjacent_xor` capability ahead of `reverse` did not improve the
shared contract. Seed `69316` had no positive transfer, while seed `69317`
failed source mastery for all three source floors (`0.8594/0.7656/0.875`) and
had no stable inherited composition. Retention deltas remained exactly zero,
so the failure is acquisition quality rather than forgetting.

This confirms that the shared interpreter is order-sensitive. The original
reverse-first order remains the stronger baseline, but the real fix is to
learn or calibrate the shared operator contract without allowing the first
capability to define it.

## Balanced joint calibration upper bound — rejected for composition transfer

To separate order poisoning from operator expressivity, the audit gained an
explicit balanced calibration mode. All three source instructions and the
shared interpreter were trained round-robin for `576` updates, then frozen
before composition learning. This is deliberately an upper bound, not a
continual-learning claim, because it uses every source procedure up front.

Source mastery became strong in both seeds (`0.9844/0.9805/0.9609` and
`0.9688/0.9570/0.9531`) with exact zero retention deltas. Nevertheless,
positive composition transfer was not replicated: seed `69316` rejected one
composition at `0.6719`, and neither seed passed the strict fresh-transfer
gate. The shared operator can therefore learn the individual primitives, but
their register states do not yet form a reliable compositional algebra.

This localizes the next major bottleneck: the representation and execution
semantics between serial instructions, not source acquisition order, rank, or
decoder normalization.

## Bounded shared-state transitions — stability improvement, not transfer

The shared interpreter now has an explicit `factorized_shared_bounded` mode.
It normalizes the register before each instruction and bounds the opaque
instruction’s residual proposal and gate, preventing unbounded serial drift.

Under the balanced joint-calibration upper bound, both seeds passed every
candidate, retention, and causal gate. Positive transfer was still mixed:
seed `69316` required `57,344` stable target bits for both compositions versus
fresh `40,960`/`16,384`, while seed `69317` improved to `24,576`/`16,384`
versus fresh `32,768`/`32,768`. The sequential rung then failed in both
seeds, with weak source floors and no stable inherited composition.

This promotes bounded transitions as a useful state-stability primitive, not
as the compositional algebra or continual-learning solution. The remaining
problem is semantic binding: serial instructions need a state representation
whose intermediate results remain meaningful to the next instruction, not
merely numerically bounded.

## Preserved execution trace — useful binding interface, not transfer

The external register now exposes a versioned execution-trace API. A chain
can preserve every intermediate register state as an opaque positional bank;
composition decoders can opt into the bank while direct source paths remain
unchanged. The fresh control receives the same trace path.

With bounded shared transitions and balanced source calibration, both
composition candidates passed in seed `69317`; seed `69316` also passed both
composition candidates, though one direct target candidate failed. Strict
positive transfer appeared in only one of four composition/seed comparisons.
Source retention remained exact, but the transfer gate was not replicated.

The trace confirms that preserving intermediate states helps binding and
stability, but a positional bank alone is not a learned state algebra. The
next step is learned, opaque addressing over the bank so the next instruction
can select relevant intermediate state without relying only on execution
position.

## Internal canonical state contract — mixed, not promoted

The shared interpreter gained an opt-in `factorized_shared_canonical` mode
that applies a learned LayerNorm state contract after every bounded
instruction transition. This tests internal content compatibility rather
than decoder-side normalization; the trace decoder and fresh control remain
matched.

Both seeds passed composition behavior and exact retention gates, but strict
positive transfer appeared in only one of four composition/seed comparisons.
Seed `69316` had inherited stable costs of `57,344`/`24,576` versus fresh
`24,576`/`40,960`; seed `69317` had `16,384`/`8,192` versus fresh
`8,192`/`24,576`. The global contract helps one ordering and hurts another,
so it is not a universal content invariant.

The result rejects imposed normalization as the final solution. The next
design must learn content structure—such as separately bindable roles or
relations—rather than force every intermediate into one undifferentiated
normalized vector.

## Learned bank addressing — useful interface, not transfer

The register now has an opt-in `factorized_shared_banked` mode. Each opaque
instruction produces a learned query over prior intermediate states, reads a
weighted opaque value, and applies the bounded shared transition. No task,
operation, or semantic address is supplied. The preserved trace decoder and
matched fresh control use the same interface.

Both seeds passed the composition behavior and retention gates, but strict
positive transfer appeared in only one of four composition/seed comparisons.
Seed `69316` required inherited stable costs of `32,768`/`49,152` versus fresh
`16,384`/`49,152`; seed `69317` improved one composition to `16,384` versus
fresh `32,768`, while the other inherited path tied or lost. The attention
router is retained as extensible infrastructure, not promoted as the learned
compositional solution.

The next bottleneck is not access to prior states but the content of those
states: the system needs a learned representation with invariants that make
intermediate results reusable across operators, rather than merely a better
addressing mechanism for incompatible tensors.
