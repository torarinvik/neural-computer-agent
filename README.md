# Neural Computer Agent

A compact research repository for building a real-time neural computer that
learns reusable cognitive primitives from sensory streams and deterministic
outcomes.

Audited model checkpoints are stored in the private Hugging Face repository:
<https://huggingface.co/torarin87/neural-computer-agent>.

The whole system's frontends receive rendered vision/audio/text streams. The
controller receives only their learned event representations, its own opaque
actions, its own latent state and memory, and scalar verifier outcomes. Neither
the controller nor its adapters receive game state, coordinates, semantic task
labels, rule IDs, correct-action labels, English chain-of-thought, or
counterfactual labels for actions it did not attempt.

## Canonical architecture

The target is an **amodal N-to-M neural computer**:

```text
N modality-specific encoders -> shared neural event bus
                              -> one modality-independent controller
                              -> shared intention bus -> M decoders/actuators
```

Both input and output counts vary at runtime. Encoders are learned frontends;
decoders and actuators are learned or calibrated backends. Adding a sensor or
output protocol must not resize the controller or add a modality-specific
reasoning branch. This is analogous to many language frontends and machine
backends sharing LLVM IR, except that the semantic content of our neural IR is
learned rather than hand-specified.

The complete normative specification, current implementation gap, terminology,
and required causal audits are in
[`docs/AMODAL_N_TO_M_ARCHITECTURE.md`](docs/AMODAL_N_TO_M_ARCHITECTURE.md).

**Current-status boundary:** the proven legacy class still supports its bundled
`step(frame)` API, but a behavior-preserving extracted runtime now owns the
vision encoder, controller core, and action decoder as three disjoint,
independently serialized components. The real five-capability checkpoint is
bit-identical through that path. A promoted algebraic migration now folds the
old two-action residual into the 24-dimensional base intention with no examples
or optimizer updates; its compatibility suffix is structurally zero and all
five repertoire gates pass at 4,096 held-out lifetimes. A runtime-variable
output bus now fans that clean intention into the inherited action decoder and
an independently reward-calibrated reversed protocol decoder simultaneously.
Three decoder seeds crossed after 64 verifier bits and passed the five-skill
closed loop; reward-shuffled and intention-ablation controls failed. The system
now also accepts runtime-variable synchronous event collections through a
generic set bus. A 4,817-parameter residual learned complementary N=2 relation
composition from scalar outcomes: 96.46% fused accuracy versus 55.84%/45.02%
for either stream alone, with three acquisition seeds crossing after
768–1,344 verifier bits. The selected bus transfers above 90% to two unseen
renderers, although that transfer is not yet replicated. The legacy feedback
API still consumes canonical action IDs, and delayed/asynchronous streams remain
unqualified. Timestamp-preserving out-of-order delivery is now qualified:
96.36% behavior is action-identical to synchronous delivery, and mismatched
timestamps remain separate. Two outcome-only attempts to adapt the input bus
to erased pixels retained clean behavior but worsened the held-out corruption
curve, so no adapted bus was promoted; the frozen bus remains the baseline.
Learned delay compensation, corruption-aware frontends, and noisy streams
remain open. Current results therefore establish synchronous N=2,
timestamp-aware transport, and M-output operation—not yet unrestricted amodal
N-to-M behavior. A new confidence-routing audit qualifies N=3 distractor
rejection: 96.58% with two streams, 58.42% with an opaque full-confidence
third stream, and 96.40% when that stream carries generic confidence 0.01.
The confidence value was supplied by the frontend in this audit; learning that
quality estimate remains open. A tiny self-supervised confidence head now
learns that quality from clean/corrupted latent consistency: across two seeds,
80%-corrupted N=3 streams improve 5.24–6.61 points while N=2 remains
98.77–99.04%. A self-supervised same-frame pair-agreement head now handles a
valid irrelevant third stream: N=3 rises from 57.45% to 89.25–89.59% while
N=2 remains 98.51–98.55%. Cross-modality relevance and more than three streams
remain open. See
`session_records/amodal_input_noise_2026-08-01/README.md` for the rejected
noise-adaptation evidence,
`session_records/amodal_n3_confidence_2026-08-01/README.md` for the interface
gate, and
`session_records/amodal_learned_confidence_2026-08-01/README.md` for the
learned-confidence audit, and
`session_records/amodal_pair_agreement_2026-08-01/README.md` for the learned
relevance breakthrough.

## North star

Maximize verified reusable capability gained per unique interaction.

The project distinguishes:

- unique verifier/reward bits;
- unique logical lifetimes;
- replayed examples;
- optimizer updates;
- GPU and wall time;
- action latency;
- retention and forward-transfer ratios.

Final accuracy alone is not an adequate score.

## Current audited frontier

The protected `next item` learner has broken its 71.6–71.7% local saturation
boundary. Error decomposition showed that the remaining failure was sharply
asymmetric: causal conflicts were already about 82.5%, while non-conflicts
were only 59–61% and the non-conflict/action-zero cell was at chance.

A verifier-side 3:2 conflict/non-conflict loss allocation is the first targeted
repair to replicate. On the high-precision matched audit it raises `next`
72.15% → 73.50%, causal conflicts 81.60% → 82.02%, non-conflicts 62.55% →
64.85%, and the hardest cell 52.99% → 56.81%. Previous-item retention remains
96.30% overall / 95.65% on conflicts, and complete memory reset is chance.
An independent lineage improves every new-skill measure. The exact paired
truthful model also beats both its unchanged parent and shuffled-outcome
control.

Focal loss, non-conflict-only weighting, and equal subgroup weighting were
rejected. The frontier is now the residual asymmetric action-zero boundary,
not more blind training duration. See
`session_records/procedural_shape_next_operation_2026-07-30/README.md`.

The procedural-shape controller now has a replicated constraint-only
rehearsal milestone. Rehearsal outcomes define protected gradient directions
without taking old-skill optimizer steps; only four target batches change the
weights. A 0.000025 trust-region learning rate keeps those updates inside the
locally valid protection region.

Across two matched seeds, aligned `next item` rises from 63.93% and 64.71% to
69.99%, while causal conflicts rise from 70.43% and 74.29% to 78.58% and
80.49%. Every redundant-anchor, first-next, and previous-item overall and
causal-conflict gate remains above 95%; complete memory reset stays at chance.
A fully paired third seed scores 69.79% with truthful target outcomes versus
66.99% with shuffled outcomes and 64.45% unchanged. The next frontier is to
compound these four-update safe increments toward mastery, halving the trust
region immediately whenever retention approaches its gate. See
`session_records/procedural_shape_next_operation_2026-07-30/README.md`.

The controller now has its first replicated protected-plasticity result at
the independent third-item `next item` frontier. Aggregate rehearsal-gradient
projection removes only target-gradient components that oppose verified old
skills; compatible learning directions remain untouched. From the same clean
parent, two independent 1,536-target-bit runs improved aligned `next` from
58.12% zero-shot to 64.00% and 64.45%, while redundant-anchor, first-next, and
previous-item overall and causal-conflict gates all remained above 95%.

A matched target-only shuffled-outcome control leaves the new relation at
52.99% and causal conflicts at 50.86%. Complete memory reset remains at
chance. A longer continuation reached 92.36% new `next` and 91.52% causal
conflicts, but previous-item conflicts slipped to 94.43%, so it was not
promoted. The frontier is therefore precise: preserve the replicated
direction-aware gain while crossing from partial acquisition to mastery, not
add stronger blanket freezing. See
`session_records/procedural_shape_next_operation_2026-07-30/README.md`.

The procedural-shape controller now has a replicated sequence-manipulation
primitive at fixed span three. An arbitrary visual operation glyph selects
either direct lookup or the relative operation `previous item`; the controller
must read its fast memory and compare the candidate with the resulting item.
It still receives only pixels, its own opaque actions, and scalar attempted-
action outcomes.

Two independent lineages score 98.30% and 98.59% on 24,576 fresh natural
lifetimes. Previous-item accuracy is 97.78% and 98.23%, conflict accuracy is
96.65% and 97.43%, and the weakest query-position/target cells are 96.44% and
96.88%. Both anchor-specific operation cells pass. Complete memory reset
returns to chance, and valid operation counterfactuals score 97.57–97.84% while
flipping 93.89–94.84% of predictions whose correct answer changes.

The replica exposes compounding acquisition: learning the second anchor in
isolation required 41,472 target verifier bits, delaying it to query two
required 23,040, and delaying it to query three required only 5,760 after a
94.92% zero-shot start. A matched shuffled-outcome control leaves conflict
accuracy near chance at 52.57% and damages old skills. The next frontier is the
adjacent `next item` operation at the same span, not a larger memory load. See
`session_records/procedural_shape_previous_operation_2026-07-30/README.md`.

The procedural-shape controller now has a replicated three-query short-term
memory primitive. It stores three independently rerendered shapes and answers
three sequential visual equality queries using only pixels, its own opaque
actions, and scalar outcomes. An adaptive three-rung curriculum introduced
immediate repeated lookup, delayed repeated lookup, and finally a novel third
lookup. The final novel rung reached stable mastery after only 5,760 additional
verifier bits in both lineages.

Final held-out accuracy is 99.70% and 99.45%; the hardest third-query cells are
98.78% and 98.00%. One- and two-query retention remains 99.69–99.87%.
Missing presentation and complete memory reset return to chance, valid
counterfactuals remain causal, and matched reward-shuffled training collapses
to 55.33%. The next frontier is minimal sequence manipulation at fixed span
three, not a larger memory load. See
`session_records/procedural_shape_three_query_curriculum_2026-07-30/README.md`.

The procedural-shape controller now has a replicated ultra-gradual nuisance
curriculum. Starting from the audited span-2 checkpoint, the scalar was mapped
in increments of 0.005. Already-mastered rungs received no updates; only the
first sub-90% rung was trained, interleaved with its immediately preceding
mastered rung.

The frontier advanced from randomness 0.090 to 0.135. Rungs 0.095 and 0.120
each needed only 512 target verifier bits; 0.135 needed 1,536 bits, and an
independent replica needed 1,024. A fresh learner remained at exactly 50%
after 8,192 target bits, so the conservative new-rung transfer advantage is
greater than 5.33x (greater than 8x in the replica). Shuffled outcomes also
remained at chance.

The final model scores 95.70% at 0.135 while retaining 100% at the original
floor, 98.73% at 0.120, 99.93% at 0.090–0.095, and 100% on visible identity
and span-1 recognition. Missing evidence and full memory reset return to
chance. Any future instability now halves the scalar increment to 0.0025.
See
`session_records/procedural_shape_randomness_staircase_2026-07-30/README.md`.

The procedural-shape memory track now demonstrates compounding acquisition
under an exact, shortcut-resistant design. A single controller first learns
two visible shape identities, then one-item recognition, then two-item
ordinal recognition. Shapes are independently rerendered with nonzero
position, scale, rotation, colour, and background variation; query order is
crossed with every identity/answer pattern so time step cannot reveal the
requested ordinal.

Two inherited seeds master span 2 at 20,480 and 16,384 new verifier bits,
finishing at 100% and 93.75%. A matched fresh learner remains at 50.68% after
32,768 bits, giving a conservative greater-than-1.6x transfer advantage.
Shuffled outcomes remain at 50%. The best model falls to 50.39% with missing
presentation and 50.05% after complete fast-memory reset; valid presentation
and candidate counterfactuals retain 100% accuracy and flip every affected
prediction. It also retains visible identity and span-1 recognition at 100%.
This is the first rung only: increasing nuisance randomness and span 3 remain
unproven in this checkpoint; the successor staircase above establishes the
first randomness expansion. See
`session_records/procedural_shape_span_2026-07-30/README.md`.

The working-memory branch is now position invariant and distractor resistant.
Balanced nuisance positions introduced during the forward-retention primitive
remove the previous stride-phase shortcut: forward span reaches 100% at the
base, intermediate, and fully shifted layouts. Retrofitting the same invariance
after specialization failed, establishing that diverse early experience—not
more duration—is the effective intervention.

That invariant retention skill compounds into manipulation. After only 4,096
new verifier bits, two forward-parent replicas reach 93.70% and 91.55% mixed
forward/reverse accuracy; matched fresh learners score 53.99% and 44.73%.
The first fresh learner needs 16,384 bits to reach the same plateau, a measured
4x new-skill sample-efficiency gain.

One-distractor adaptation then preserves zero-distractor accuracy and
generalizes without additional training to two distractors: two replicas score
93.58% and 93.48%. Blank evidence, complete fast-memory reset, and
reward-shuffled adaptation all return to chance. Four distractors reduce
accuracy to 82.12%, defining the next selective-retention boundary. See
`session_records/sequence_working_memory_robustness_2026-07-30/README.md`.

The controller now has its first causally audited working-memory atom. It
observes two abstract visual events and conditionally emits either the original
sequence or its reversal. The sequence, requested operation, and correct
actions remain verifier-private; learning uses only RGB streams, opaque
attempted actions, and scalar outcomes. Fast state and workspace tensors stay
resident in RAM/VRAM throughout the episode.

Mastering forward retention before mixed forward/reverse training produces a
large and replicated transfer gain. At 16,384 new verifier bits, curriculum
seeds reach 99.11% and 93.84% held-out versus 81.51% and 75.34% from fresh
weights. Both curriculum runs retain forward recall at 100%. The strongest run
crosses a stable 90% gate at 14,336 new bits; the matched fresh learner never
crosses within budget.

The result survives valid cue and sequence reversals, blank evidence, complete
fast-memory reset, and shuffled-outcome training. Removing all fast memory
returns performance to 50%; independently disabling the differentiable
workspace or recurrent carrier causes smaller losses, showing redundant use of
both. This is deliberately bounded: an unseen distractor reduces performance
to 84.08%, and disjoint object positions fail. The next frontier is gradual
position invariance followed by selective distractor-resistant retention—not
another memory architecture.

See
`session_records/sequence_working_memory_2026-07-30/README.md`.

The same/different repertoire now robustly spans bars, diamonds, and
disconnected dot pairs in one unchanged controller architecture.  A fixed
64-update acquisition plus 32-update consolidation recipe replicated on all
three fresh seeds.  Mean held-out accuracy was 99.61%, 96.62%, and 97.81%;
every inherited primitive remained above 90%.

The robust threshold uses 3,072 new lifetimes and 64,512 total verifier bits:
3.33 times fewer new lifetimes and 2.86 times fewer total outcomes than the
preceding diamond bridge.  A fresh 8,192-lifetime audit of the promoted
checkpoint scored 99.65%, 97.85%, and 97.78%, while missing-second-object
controls remained at 49.41–50.01%.

Matched controls showed that a proposed additive gate extension was not
necessary.  The gain came from letting the existing architecture cross its
ignition valley and then consolidating it, not from adding capacity or running
a population search.  Inference is already at one controller pass per event
with zero optional thoughts.  The next frontier is a genuinely new relation
on familiar appearances, with experience-to-threshold measured before any
execution-step compression.  See
`session_records/pair_relation_robust_compound_2026-07-29/README.md`.

The repertoire now includes its first cross-family simultaneous visual
relation. The same one-controller lineage learned whether two rendered objects
were the same or different from its own opaque attempted actions and scalar
outcomes—no semantic relation labels, task IDs, coordinates, or correct
unattempted actions. Three independent promoted runs reached 99.02%, 99.56%,
and 99.46% on held-out colors and positions while all three inherited behavior
gates passed. Valid second-object counterfactuals passed; blank vision and
removing the second object returned performance to 48–49%.

The audit rejected an earlier false positive whose relation was accidentally
constant within a lifetime: recurrence could infer every later answer from one
reward while ignoring vision. After making the relation vary on every event, a
generic residual-locality price—not extra replay—removed the remaining
retention interference. The honest next boundary is contour abstraction:
zero-shot diamonds remain at ~26% and disconnected dot pairs at 68–71%.
See
`session_records/repertoire_pair_relation_2026-07-29/README.md`.

The persistent-memory line now has a verified-use plasticity milestone. Every
physical row carries a modality- and task-agnostic volatility scalar. Successful
retrieval gradually protects a row, failure thaws it, and disuse slowly restores
plasticity; access frequency alone cannot freeze a memory. The field survives
RAM/VRAM movement, selection, growth, disk save/reload, and old v1-v3 memories
load as fully plastic.

In a non-stationary bounded-memory atom, three stable skills, three equally
frequent but consistently failing decoys, and two stale skills competed with
four new skills. Verified-use volatility retained 100% of stable skills while
acquiring 100% of the new skills over 64 seeds. Uniform replacement scored
77.46%, access-only plasticity 71.43%, and shuffled row/volatility
correspondence 79.02%. This directly falsifies the tempting shortcut “frequent
means important”: stable and decoy rows had identical access counts, but only
verified usefulness separated them.

A fresh 321-parameter generic replacement selector then learned to use the
scalar from final verifier reward alone—no task identity, semantic row label,
or correct replacement action. Four independent 192-update runs each reached
100% perfect episodes on 512 held-out environments. Physical disk-backed audits
retained 100% of stable rows and installed 100% of new rows; shuffling
volatility among rows reduced stable retention to 51.0–53.4%. Access-only and
uniform controls lost roughly one stable skill per episode. This establishes
learned selective stability in external memory.

That mechanism is now integrated into the full visual controller. The parent
replacement policy already saw age, strength, similarity, access frequency,
and aggregate reliability. A zero-initialized eighth feature added volatility;
the only trainable quantity was its single scalar coefficient. Stable and decoy
rows were deliberately matched on access count and on total successes/failures
(five each). Only the *order* of scalar verifier outcomes differed.

Two independent 32-update runs reached 99.61% and 98.83% valid replacement on
held-out visual lifetimes, within 0.3 points of their perceptual oracles. Both
crossed and retained 95% after 24 updates, 6,144 unique verifier bits, and
10,752 logical contexts, with no replay. A matched reward-shuffled run remained
at 57.81% and never crossed threshold. Shuffling volatility reduced correct
replacement to 47–49%; reversing outcome order made the controller evict the
previously stable row on 98.4–99.6% of banks. All inherited behavioral and
memory-utility gates remained intact. Each complete run took 72–80 seconds,
well inside the five-minute cap.

Three physical disk-backed audits then achieved 100% valid replacement,
91.80–95.12% visual accuracy, exact persistence of keys, values, access,
success, failure, and volatility fields, and zero capacity growth. Shuffled or
constant volatility fell to 46.9–52.3%; reversing histories flipped 100% of
replacement choices. This physical claim currently assumes equal admission
strength while histories are accumulated. With unequal learned strength priors,
content retrieval redirected some exact queries to other rows and correct
replacement fell to 67.19%. Credit attribution under unequal retrieval priors
was therefore the next explicit frontier, not part of that promoted claim.

That frontier is now closed for the audited exact-content regime. Retrieval's
write-strength prior now has a backward-compatible controller scalar, initialized
at the old value `1.0`. A five-candidate population race compared scales
`0, 0.25, 0.5, 0.75, 1` on matched physical banks using only pixel-task reward.
Two independent races selected content-first scale `0.0` from 1,280 verifier
bits and 448 unique logical contexts in 23.8–24.2 seconds.

Under unequal learned admission strengths, the unadapted parent reached 64.06%
valid replacement. Both selected controllers reached 100%, with 98.05% and
99.02% visual accuracy. Volatility shuffling returned replacement to
46.88–50%; reversing histories flipped every choice. A reward-shuffled race
selected `0.5`, reached only 69.53%, and failed the causal gates. Two independent
512-context selective-disk audits retained the older loop at 92.77–93.55%
first-reload and 93.36% repeat-reload accuracy, with value corruption causal and
duplicate rates below 12%.

That per-query frontier is now closed on a balanced exact-versus-ambiguous
retrieval atom. A dormant 49-parameter policy observes four generic retrieval
statistics and chooses whether each query should be content-first or should
also use verified row strength. It is trained only from the controller's
attempted retrieval and scalar visual-task outcome.

Two independent 80-update runs reached 100% held-out accuracy on both arms and
remained perfect from update 40 onward: 5,120 unique verifier bits to stable
95%, 10,240 bits total, no replay, and 27.4–34.6 seconds end to end. Either
global rule is insufficient: fixed content-first reached 73.0–74.2%, while
fixed strength-aware retrieval reached 74.4–74.8%. Shuffling the policy's
generic features reduced its action accuracy to chance and task accuracy to
73.8%; corrupting retrieved values reduced task accuracy to 47.9–48.6%.

The result survives 256 independently saved and reloaded two-row physical
banks per seed at 100% accuracy with exact persistence. A reward-shuffled run
never learned the conditional action and stayed at 73.6%. The earlier
selective-disk loop still passed at 92.8–94.1% reload accuracy, and the
unequal-strength volatility audit retained 100% valid replacement with exact
histories.

The binary policy has now compounded into continuous resource control. A
hardened task makes any constant scale impossible: exact queries require usage
influence below `0.12–0.18`, while ambiguous queries require it above
`0.35–0.55`. Fixed scale zero and one therefore each retrieve the correct row
exactly 50% of the time. Correctness remains the primary reward, with a smaller
generic cost for stronger historical influence.

Two independent eight-update runs retained 100% exact and ambiguous retrieval
while reducing mean scale from the inherited binary policy's `0.50` to `0.312`
and `0.347`. Both reached the joint correctness-and-efficiency gate at update
five and remained above it through updates six, seven, and eight: 640 unique
verifier bits to stable improvement, 1,024 total, no replay, and about five
seconds of training.

Without any further training, both controllers achieved 100% on three- and
four-row banks, including 128 independently saved and reloaded physical banks
at each size. Shuffling generic query features reduced row accuracy to
49.4–53.3%; corrupting values reduced visual success to 45.7–48.9%. Reward
shuffling and resetting the inherited conditional policy each collapsed row
accuracy to 50%, showing that both new verified feedback and the old learned
skill are necessary. The original conditional task, selective disk,
unequal-strength volatility, binary mapping, and four-rule behavior all remain
retained.

That four-way frontier is now closed on its first robust curriculum rung.
Every query contains four behaviorally distinct values, any one of the four
rows can be correct, and physical row order is independently permuted. The
controller explores its generic continuous retrieval action, retains only
regions that earn real verifier reward, and reuses one verified batch for
intensive internal optimization. Parent rehearsal constrains deployed
retrieval behavior rather than freezing obsolete numeric activations.

Three independent runs reached 100% in every target regime from one batch:
512 unique verifier bits and 512 new logical contexts. Training took
1.47–2.72 seconds despite 1,000 internal replay updates. All runs retained
100% parent continuous and conditional retrieval plus binary-mapping and
four-rule gates. They also reached 100% across 128 independently saved and
reloaded disk banks. No fixed scalar exceeded 25%; feature shuffling fell to
24.8–25.4%, value corruption fell to 0%, and shuffled reward learned only one
class at 25%.

That zero-shot boundary frontier is now closed. The fixed-envelope parent
collapsed to 0% when its crossing points moved in an unfamiliar direction.
The cause was observational aliasing: its four statistics described only the
best two rows. A zero-output 113-parameter residual now also observes four
sorted cosine values and their four usage values. The inherited controller is
frozen and insertion is exactly behavior-preserving.

Training shifts were sampled continuously from `[-0.09, 0.12]`; evaluation
used disjoint bands `[-0.099, -0.095]` and `[0.13, 0.16]`. Two independent
runs reached 100% in all four classes on both unseen bands and stable mastery
after 1,536 verifier bits, using 4,096 bits total. Both retained parent
continuous/conditional retrieval at 98.9–100% and every older behavioral gate.
All 512 shifted physical bank evaluations were correct after exact disk
reload. Feature shuffling fell to 23.3–25.0%, value corruption to 0%, shuffled
reward to 0–25%, and the exact four-feature ablation to 50%.

That independent-shape frontier is now closed. A zero-effect 421-parameter
relational proposer computes four generic regions where candidate memory rows
exchange rank. The learner executes those four proposals and uses only their
scalar verifier outcomes to train both the candidate selector and the final
continuous action. This breaks the former closed-gate credit-assignment loop
without exposing a target row, private boundary, or correct action.

Training crossing and slope deformations were bounded by `±0.07` and `±0.12`.
Two disjoint held-out shape families used crossing magnitudes
`[0.075, 0.085]` and slope-ratio magnitudes `[0.13, 0.15]`. Two independent
runs reached 100% in every class on both families, retained every older gate,
and completed 512 physical disk-bank audits with exact reloads. Feature
shuffling fell to 23.2–26.8%, value corruption and shuffled rewards fell to
0%, and a matched selector-credit ablation left the hard middle class at 0%.
The conservative replicated stable-learning threshold was 8,192 verifier
bits; the best seed stabilized after 512.

That natural-equivalence frontier is now closed for the binary hidden-rule
family. A discarded probe first established that independently acquired
same-rule memory values carry a decodable relation (99.79% linear and 100%
with a 32-unit pair scorer on held-out appearances). The deployed controller
then gained a zero-effect 12,354-parameter shared relation scorer. It compares
a fresh feedback-derived memory value with four independently stored values
and chooses one of the existing generic rank intervals. Training uses only
the four scalar outcomes earned by actually retrieving the candidate values.

Two independent 1,024-verifier-bit runs reached 100% when one, two, or three
stored rows were behaviorally equivalent, while the inherited policy remained
at 46.9–50.6%. All 256 physical disk banks behaved correctly after exact
reload. Probe shuffling fell to 49.2–52.0%, stored-relation shuffling to
52.7–53.5%, retrieved-value corruption to 35.2–35.4%, and matched reward
shuffling to 43.6%. An exact-duplicate-only curriculum stopped at 86.9%, so
the gradual bridge to independently acquired equivalents is causal at this
budget. A valid counterfactual replay held every candidate bank tensor and
pixel stream fixed, reversed only the verifier rule, and caused the fresh
latent and selected physical row to flip in 100% of cases while behavior
remained 100% correct. Every older retention gate passed.

That capacity-limited consolidation frontier is now closed for streams of two
hidden binary behaviors. Only a scalar scale and bias were trained from
verifier outcomes; rule bits, equivalence labels, and merge/store targets
remained private. Two independent 64-bit runs compressed 16 natural
controller-created memories to two rows and reached 99.46–99.51% held-out
behavior, with both distinct skills retained in 98.93–99.02% of streams.
Every physical bank reloaded exactly. The reduction is 8× in logical rows and
3.09× in serialized bytes because fixed metadata dominates tiny files.

The 64-bit result passed all inherited retention and counterfactual gates.
Inverting the learned relation reduced two-skill retention below 0.9%, and
shuffling the verifier outcomes reduced behavior to 50% on both seeds. A
32-bit race passed one seed but failed the other, so 64 bits is the smallest
replicated frontier rather than a selected lucky run.

The next frontier is compounding utility: test whether the clean,
capacity-limited bank reduces the verifier experience required to acquire a
genuinely new primitive while retaining both old skills.

The first zero-shot transfer arm exposed a useful correction to that plan:
merging every equivalent memory to one prototype over-compressed future-useful
variation. One representative per behavior retained roughly 97.2–97.5% when
bar-shaped memories were queried through a never-trained disconnected
dot-pair geometry, while the uncompressed bank remained near 99.5%.

A diversity-preserving policy now keeps two relation-equivalent
representatives per behavior. On two independent 4,096-stream audits it
reached 98.36% and 98.57% on dot pairs with zero new training outcomes,
retained 100% on bars and 99.69–99.72% on unseen diamonds, and preserved both
skills in every bank. The four-row bank is 4× smaller logically than the
16-row source and 2.41× smaller in serialized bytes. Naively keeping the first
four rows reached only 91.99–92.13%, and zeroed memory fell to chance.

All 2,048 physical banks reloaded exactly. Counterfactual reruns kept RGB and
bank tensors fixed, reached 98.46–98.54% in both rule worlds, and flipped the
selected row in 98.07–98.36% of cases. No parameters changed and every
inherited gate passed.

The current frontier is adaptive diversity budgeting: learn when an
equivalence class needs one, two, or more representatives from verifier
history and resource pressure, rather than fixing that allowance by hand.

The read-compute half of that frontier now passes. A 32,097-parameter
action-conditioned critic reads the fresh latent and the first representative
from each learned class, then predicts whether consulting the remaining
representatives will improve verified success. It receives only the two
attempted read budgets' scalar outcomes.

At the replicated 16,392-verifier-bit frontier, two independent controllers
reached 99.57% held-out accuracy versus 99.56% and 99.62% for always reading
all six rows. They averaged only 2.092 and 2.094 comparisons instead of about
5.997—a 65.1% reduction—and beat the full-read latency-aware verified utility
in both seeds. A 8,196-bit rung passed one seed and failed the other, so it was
not promoted.

Shuffled critic features fall near the shallow baseline, zeroed memory falls
to chance, and shuffled verifier training fails every capability-specific
gate. All 1,024 physical banks reload exactly, reversed-rule behavior remains
99.53–99.58%, inherited skills pass, and every old tensor is bit-identical.

The remaining frontier is adaptive physical storage: use accumulated marginal
read value to prune representatives that are not worth their disk/RAM cost,
while retaining uncertain diversity until evidence justifies deletion.

The new unified-controller line now has its first retained compounding
milestone. A single 298,252-parameter controller with one vision encoder,
recurrent state, generic differentiable workspace, latent intention, and
replaceable actuator adapter learns hidden visual-action functions from its
own attempted opaque actions and scalar outcomes.

Prior visual grounding changed a matched 600-step four-rule task from a stable
75% shortcut to 99.85–99.90% on two independent seeds. The next rung inferred
an identity-versus-flipped mapping after one support outcome; inherited
training reached 100%, while matched fresh stayed at 49.26%. Balanced rehearsal
then preserved both the one-support skill and the broader two-support
four-function skill. The selected checkpoint passed disjoint 2,048-lifetime
normal, private-rule reversal, prediction-flip, blank-vision,
shuffled-feedback, and active-state-reset audits:

- one-support bijection: 99.98% normal, 99.95% reversed;
- retained four-function task: 100% normal and reversed;
- paired counterfactual flips: 99.93% and 100%.

This is evidence of fast within-lifetime binding, positive forward transfer,
and behavioral retention in one controller.

The same controller now also performs content-addressed latent recall across
active-state resets. A 600-update capacity-two rung reached 96.53% blind
recall; 150-update bridges at capacities 8 and 16 produced zero-shot transfer
to capacities 16 and 32. A later five-second rung used only 20 new-memory
updates at capacity 40 and reached 90.00% blind recall, then transferred
zero-shot to capacity 48 at 88.28% and capacity 56 at 87.33%. An independent
five-second acquisition replicated the result, and the two checkpoints crossed
the old capacity-64 frontier at 85.57% and 86.33%. Empty, shuffled, and
corrupted memories collapse toward chance; disk save/load reproduces hard
retrieval; the earlier one-support and four-rule skills remain retained. The
frozen retrieval frontier is now capacity 72; both parents fail capacity 80.

A subsequent selective-memory atom learned from verified success minus a
generic write cost. On blind data it wrote on 61.16% of first encounters but
only 5.10% of redundant repeats, averaging 0.663 writes per context while
retaining 99.90% query accuracy. Removing writes, shuffling admissions,
corrupting values, or hiding the prior memory read causally degraded the
appropriate behavior.

The first physical integration audit exposed a boundary: intentionally absent
default rows retrieve unrelated neighbors in a shared disk bank. Scalar
rejection gates restored 87.99–88.96% disk accuracy but missed duplicate or
false-accept gates. A discarded diagnostic localized the limitation: the four
generic memory statistics supported only 83.01% held-out classification with
a five-parameter linear gate, while an eight-unit nonlinear gate reached
88.18%. No diagnostic weights entered the agent.

A fresh 49-parameter gate inside the same controller was then trained only
from verified query success minus a generic read cost. In 160 updates, 81,920
unique contexts, and 9.71 seconds it reached 91.55% held-out accuracy, accepted
89.67% of useful reads, rejected enough absent reads to hold false acceptance
to 17.33%, and retained both prior behavioral gates.

Two independent physical disk audits then passed every pre-registered gate:

- first save/reload accuracy: 91.50% and 92.19%;
- accuracy after a repeated encounter: 91.02% and 91.41%;
- duplicate rows per context: 17.68% and 17.29%;
- empty-memory controls: 50.20% and 50.00%;
- wrong-value disk corruption controls: 70.41% and 70.70%.

This admits the first unified learned RAM/VRAM-to-disk loop: the controller
creates sparse opaque rows, reloads them after active-state erasure, learns
whether a retrieved row deserves use, and suppresses redundant repeat writes.
The selected checkpoint is
`artifacts/checkpoints/unified_selective_disk_adaptive_seed5962.pt`, SHA-256
`91822064436fae1d4f799e41c79d9369dacb8aeeee20b711df1c1b6af037fbc4`.

The next gradual atom bounded each disk bank at four rows. A 57-parameter head
inside the same controller learned whether to skip or which physical row to
replace, using only generic row metadata and later verified success. After 40
updates and 7.01 seconds it reached 96.90% held-out accuracy versus 84.35%
random, 85.11% fixed-slot, and 80.91% skip controls. Two physical disk audits
replicated at 96.97% and 96.29% with exactly 2,048 rows before and after and
zero capacity growth. Shuffling age-to-slot correspondence reduced accuracy to
81.35% and 82.37%. The earlier sparse disk loop still passed at 91.21%.

The promoted 298,358-parameter checkpoint is
`artifacts/checkpoints/unified_memory_replacement_seed6101.pt`, SHA-256
`0178b15228e3d75a445abdb2376be1291a078f8b47236444fbd1824fab3d3b76`.
That first policy transferred zero-shot to capacity 5, but not causally to
capacity 6. A gradual bridge then used only 20 capacity-6 updates interleaved
with 20 capacity-5 rehearsal updates. It matched the capacity-6 oracle at
96.39% with 100% correct evictions and retained every prior gate. Two physical
replications reached 96.55% and 96.71%. The sharpened rule subsequently
transferred with zero weight updates through capacities 7 and 8 and reached a
replicated physical capacity-9 frontier at 94.57% and 94.62%, with
99.61–100% correct evictions and zero growth.

The fixed-utility parent is
`artifacts/checkpoints/unified_memory_frequency_recency_capacity6_seed6607.pt`,
SHA-256
`1346da994de4ba20864c5f1bc1da12684fc13d8dcda480a76cfc6f713da0181c`.

The next utility rung kept capacity fixed at six but made future usefulness
depend jointly on noisy recency and access frequency. Ordinary
content-addressed reads now increment persistent access counters, which survive
disk save/reload and reset on replacement. A zero-initialized one-parameter
residual let the proven recency policy compose this new generic statistic
without changing its inherited path.

Two reward-only 20-update runs passed in 3.23 seconds of training each. They
reached 95.32% and 95.10% held-out future accuracy, 87.30% and 86.13% correct
evictions, and retained recency, binary mapping, and four-rule gates. The
learner consumed 51,200 unique verifier bits per run with no replay or utility
labels; only the one new coefficient changed.

Two physical disk audits then reached 96.81% and 96.29% on access histories
generated by actual retrievals. They made 92.97% and 93.36% correct evictions,
captured 76.3% and 91.7% of the visible-oracle gap above the strongest
single-feature control, preserved all 512 audited histories exactly through
save/reload, kept 3,072 total rows bounded, and never grew capacity. Shuffling
age reduced accuracy by 4.56–4.75 points; shuffling frequency reduced it by
6.77–7.75 points.

The next rung removed the fixed utility mixture. In one uninterrupted stream,
the relative value of recency and access frequency changed from 65:35 to
35:65, returned to 65:35, and ended at 50:50. No phase identity, boundary
signal, optimizer reset, utility label, correct eviction label, or replay was
available to the learner. A symmetric two-candidate horse race changed only
the controller's existing one-parameter utility residual according to which
candidate produced more verified future success.

Two independent 64-update runs passed all online, retention, and causal gates
in 28.66 and 28.89 seconds:

| Seed | Recency target | Frequency target | Recency-return target | Equal-return target |
|---:|---:|---:|---:|---:|
| 6809 | 90.67% | 86.43% | 91.16% | 90.53% |
| 6810 | 90.82% | 87.16% | 91.31% | 89.99% |

The frequency-dominant phase improved held-out accuracy from 93.87% to 95.62%
and from 93.49% to 95.37% over an unadapted copy. Shuffling which candidate
received each verified outcome made the adaptation fail: the frequency target
fell to 57.71%. The selected checkpoint then passed a 1,024-bank physical disk
audit at 96.94%, within 0.13 points of the visible oracle. Shuffling age or
frequency reduced it to 92.74% and 88.66%; all 6,144 rows and access histories
survived save/reload exactly and capacity never grew.

The one-parameter online parent is
`artifacts/checkpoints/unified_memory_online_utility_seed6810.pt`, SHA-256
`c3e837c6512a30c11b1c861b79242296b76cfa0cd9fe62aa414d3e5b2aa10750`.
This establishes rapid verifier-driven adaptation of one generic controller
coefficient, not yet a learned general-purpose internal meta-optimizer.

The next gradual rung added one genuinely new task-agnostic statistic:
verified outcome reliability for each memory row. Physical rows now keep
success and failure counts attributed through ordinary content-addressed use.
The counters survive disk save/reload and reset when a row is replaced. Older
memory schemas load with zero counts.

A redundant write-strength coefficient was first rejected: write strength was
already visible to the inherited controller, so it added only 2.93 target
points. Reliability passed the representation gate by over 30 points and then
learned online. Two independent 48-update runs used a three-candidate
move/stay horse race, 196,608 verifier bits, no replay, and about 29.4 seconds:

| Seed | Old equal | Reliability dominant | Old return | All equal |
|---:|---:|---:|---:|---:|
| 6932 | 89.75% | 78.22% | 88.48% | 87.45% |
| 6938 | 88.67% | 88.43% | 84.72% | 83.35% |

Frozen target rates were only 57.62–58.40% in the reliability phase and
63.48–64.60% under all-equal utility. The exact reward-shuffled control failed
the multi-phase gates and ended all-equal at 64.31%. Both intact runs retained
binary mapping and four-rule behavior; only the two-coefficient generic
residual changed.

The selected controller passed an independent 1,024-bank physical audit at
96.21%, versus 96.35% for the visible oracle. All 6,144 rows and all
access/success/failure histories survived save/reload, with zero capacity
growth. Shuffling age, frequency, or reliability changed correct evictions by
50.29, 60.55, and 30.18 points and reduced actual accuracy by 3.11, 6.75, and
2.56 points.

The tensor-trained parent checkpoint is
`artifacts/checkpoints/unified_memory_multifeature_reliability_seed6932.pt`,
SHA-256
`bb5cd158c08f4b92061aca7bfae0751d4e18408e8e37f53cac13dffaed8ac9f4`.
It has 298,360 parameters.

The adaptation loop itself now also runs through bounded physical disk
memories. A parity preflight made disk serialization sovereign and kept the
old tensor arena only as a shadow audit. Two undersized 32-bank pilots were
rejected before scaling. At the proven 128-bank scale, seeds 7012 and 7015
both passed the four utility phases, retention, persistence, parameter-scope,
and physical/tensor parity gates in about 136 seconds. Seed 7015 achieved
85.74%, 77.25%, 86.72%, and 82.67% across old-equal,
reliability-dominant, old-return, and all-equal phases. All 6,144 physical
histories persisted; all 48 physical choices were tensor-equivalent. The
matched reward-shuffled control failed every adaptation phase and saved no
checkpoint.

The current checkpoint is
`artifacts/checkpoints/unified_memory_physical_online_seed7012.pt`, SHA-256
`2c6e61b5e2689d46dfc43dd5cfc9c5b234736d217aae28f6221501bd5ddeea70`.
The independent replica is
`artifacts/checkpoints/unified_memory_physical_online_seed7015.pt`, SHA-256
`7ae96b44ec6bed0db8eb7f9b78640fe40b621875195303e3e3c604f357bb441d`.

Unseen elongated diamonds and disconnected dot-pair stimuli also transfer
zero-shot at 94.95–98.14%, tightening the evidence that visual identity is
relational rather than tied to the original rectangles.

Long-lived physical banks that accumulate experience across multiple updates,
consolidation, deletion/merging, unbounded memory, and cross-modality transfer
remain open.
See `experiments/unified_cognitive_controller/README.md`.

The two-decision identify-then-act task requires the agent to:

1. emit an opaque probe action;
2. observe its visible consequence;
3. infer the hidden actuator mapping;
4. observe a target;
5. emit the correct opaque action.

The current fresh predictive learner reached on seed 211:

- 100% held-out accuracy at 64 unique verifier bits;
- 100% accuracy and 100% prediction flips under valid protocol rerenders;
- 100% accuracy and 100% prediction flips under target reversal;
- chance performance when the probe consequence is removed.

An incremental 8→16→32→64-bit learner reached 93.36% with 256 cumulative
optimizer updates. A 32-bit arm with 512 updates failed at 52.73%, so extra
replay does not substitute for the missing unique outcomes.

A subsequent exact three-seed map corrected the robustness claim. At 64 bits,
normal accuracy was 55.47%, 99.61%, and 81.64% for seeds 151, 211, and 307;
only seed 211 passed every causal and anti-fluke gate. Thus 64 bits is the
current single-seed capability frontier, not a robust sample threshold.

Earlier fixed-target weights caused negative transfer to the full task.
Inherited weights are therefore retained only when they improve the next
held-out learning curve.

See:

- `experiments/forward_transfer_attention/SAMPLE_EFFICIENCY_LEDGER.md`
- `experiments/forward_transfer_attention/MICRO_INTERCEPT_DESIGN.md`
- `experiments/forward_transfer_attention/README.md`

## Repository map

| Path | Contents |
|---|---|
| `experiments/unified_cognitive_controller/` | Single-controller few-shot binding, retention, and persistent-memory interface |
| `experiments/forward_transfer_attention/` | Main sample-efficiency, transfer, memory, binding, and causal-audit research |
| `experiments/syllogimous_neural_computer/` | Learned external-memory neural computer |
| `experiments/syllogimous_latent_agent/` | Latent real-time agent and sensory models |
| `experiments/syllogimous_bitter_lesson/` | Emergent reasoning experiments without symbolic solution machinery |
| `experiments/syllogimous_realtime/` | Real-time deterministic syllogism environment and Elisa sources |
| `experiments/sensory_codec/` | Sparse sensory stream experiments |
| `artifacts/checkpoints/` | Curated current checkpoints that are small enough for Git |
| `artifacts/manifests/` | Checksums for curated and excluded historical artifacts |
| `session_records/` | Compact historical reports and continuation notes |

## Setup

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CUDA-capable PyTorch is recommended for training. CPU and Apple unified-memory
backends are useful for tests and tiny diagnostics.

## Verify

```bash
python -m pytest experiments/forward_transfer_attention -q
./scripts/verify_curated_artifacts.sh
```

The narrow current-task regression suite is:

```bash
python -m pytest \
  experiments/forward_transfer_attention/test_identify_then_act.py -q
```

## Reproduce the current 64-bit task

```bash
python -m experiments.forward_transfer_attention.train_identify_then_act \
  --report experiments/forward_transfer_attention/reports/reproduction.json \
  --checkpoint-out artifacts/checkpoints/reproduction.pt \
  --device cuda \
  --seed 211 \
  --curriculum-rung random_probe \
  --intention-width 64 \
  --pretrain-lifetimes 128 \
  --pretrain-steps 40 \
  --policy-lifetimes 64 \
  --test-lifetimes 256 \
  --fit-updates 256 \
  --batch-size 32
```

## Next experiment

Two different 32-bit searches are now complete. The feature-interface winner
fell from 69.53% blind accuracy to 55.47% on its replication seed. A subsequent
learning-mechanism population compared a frozen core, zero-initialized residual
adapters, and conservative action/predictor/recurrent adaptation. Its rank-16
adapter reached 66.41% blind accuracy on seed 211 but also fell to 55.47% on
seed 307, with invalid causal reversal behavior. No checkpoint was promoted.

The cheap "more readout capacity or optimizer freedom" branch is closed at 32
unique outcomes. A subsequent eight-clone reward-free predictive-objective
screen also failed: contrastive refinement reached only 58.59% blind accuracy,
and the unrefined core had the best final selection score. Extra auxiliary
prediction losses therefore do not earn a longer run.

The variance decomposition is complete. Across a nine-horse race at 64 bits,
predictive-core initialization changed the causal floor by 74.22 percentage
points, versus 7.03 points for readout initialization and 5.86 points for
readout replay sampling. All frozen cores passed exact retention checks.

The next sub-minute population should therefore race predictive-core
initializations under identical experience and optimizers, using successive
halving at 32, 48, and 64 outcomes. A winner must then reproduce on a disjoint
lifetime stream and pass old-capability retention before promotion.

That race is now complete. Core seed 263 passed every causal and anti-fluke
gate at 48 and 64 outcomes on a disjoint policy stream with a different
downstream initialization. Its replicated causal floors were 98.05% and
97.27%, respectively. Core seed 211 did not reproduce a stable pass.

The new frontier is therefore a **population-selected, replicated 48-bit
learner**, with search compute accounted separately. Seed 263 is admitted to
the prior-primitive retention/compatibility suite; no general-agent checkpoint
is promoted until that suite passes.

The selected core is now materialized as an immutable 2.9 MB candidate with
SHA-256
`d027b80a631f61c3a9769b60a079494e0a669e1211d3324a13e5ad7b65a1006d`.
Exact reloads reproduce metric-for-metric. With exact complemented negative
controls it passes every gate at 48 and 64 outcomes. A tempting 40-bit point
reaches 95.31% accuracy but fails the missing-evidence uncertainty gate and is
honestly rejected.

Compatibility testing preserves fixed-probe mastery at 16 outcomes and
fixed-target mastery at 48 outcomes, with the predictive core bit-identical
throughout behavioral learning. This establishes a reproducible 25% reduction
from the previous 64-outcome frontier without observed forgetting inside the
identify-then-act family.

The first compounding ladder is also complete. With the immutable core frozen,
novel target-side and observed-effect-side questions each require 8 outcomes,
and their effect-target composition requires 24. The composition replicated on
two disjoint streams while matched-fresh stayed at chance through 64 outcomes.
The gain localizes to the learned vision encoder.

A gradual appearance bridge then changed the palette, object geometry, and
finally both. Stable composition mastery remained 24 outcomes on every rung;
the combined shift replicated with 100% normal/counterfactual accuracy and
100% causal flips. A third-stream retention audit preserved the earlier ladder
at 8/8/16 outcomes. This is verified surface generalization and earlier ability
reuse, not yet broad amodal transfer: spatial relation and event structure are
still shared.

The next bridge replaces position with color identity. It uncovered selective
negative transfer—position-trained vision accelerated observed-effect color
but suppressed target color—so the system retained the useful branch and reset
the harmful one. After acquiring both color primitives from attempted answers
and scalar outcomes, a new relation head reached stable causal mastery from 16
new outcomes on both the selected and blind streams. The identical unacquired
architecture, and either primitive alone, failed through 64 outcomes: a
replicated transfer-ratio lower bound of 4×.

The blind audit reached 100% normal accuracy, 100% accuracy and flips under
both protocol and target rerenders, chance with either fact missing, and 0%
under exact complement controls. Stratified shuffled-label controls produced
no causal pass. The earlier position ladder remained 8/8/24 with bit-identical
cores.

The curated 5.5 MB milestone is
`artifacts/checkpoints/color_primitive_compounder_bits16_seed1901.pt`.

See
`experiments/forward_transfer_attention/ROBUST_SAMPLE_EFFICIENCY_STRATEGY.md`
for the population-search decision and pre-registered diagnostic.

The longer-term optimization is a gradient-trained population with
successive-halving compute allocation. Fitness is held-out learning AULC,
stable bits-to-threshold, retention, latency, and positive transfer to the next
primitive—not old-task accuracy.

## Latest ancestry frontier

Exact-zero skill gates protected retention but made deeper ancestry invisible.
Separating reads from writes restored transfer at ancestry 3 → 4: readable
latent content produced a +0.0242 pooled advantage (48W/22L, p = 2.5e-3), while
a zero-content capacity control did not.

At 4 → 5, reading still improves absolute new-skill learning substantially:
+0.0944 for the four-skill parent and +0.0785 for the five-skill parent against
matched no-read controls. The open problem is accumulation—a second readable
ancestor adds no further depth advantage. Compressing the combined read does
not fix it. A sub-minute recent-only pilot preserved +0.0807 over no read and
was +0.0191 above reading both ancestors, but did not yet reverse the
deep-versus-shallow gap, so no longer run was promoted.

The next tiny diagnostic compares one immediate ancestor with one older
ancestor. A causal difference would justify learned task-agnostic latent
selection; a flat result would rule routing out before more compute is spent.
See `session_records/strategy_accounting_2026-07-28/README.md`.

## Latest verified frontier: causal magnitude compounding

The 369,926-parameter unified controller now learns a genuinely new
larger/smaller visual relation while retaining its earlier repertoire. Two
independent causal audits scored 92.05% and 91.96%; deleting the second object
fell to about 60%, and disabling the inherited same/different representation
cost 10.70 and 10.34 percentage points. All old relation appearances and
unrelated cognitive skills retained their gates.

The acquisition used no semantic labels: only pixels, opaque attempted
actions, and scalar outcomes. It required 131,072 new-task lifetimes and is
already compiled to one controller pass per event; additional thought reduced
accuracy. The checkpoint is stored on Hugging Face as
`checkpoints/unified_pair_magnitude_compound_seed21475.pt`.

The next gradual rung is magnitude across morphed, diamond, and disconnected
dot-pair appearances, with sample efficiency measured before optional thought.
See `session_records/pair_magnitude_compounding_2026-07-29/README.md`.

## Latest breakthrough: learned skill advances the next unseen frontier

The magnitude controller now learns a gradual contour change from only 256 new
lifetimes while preserving every older capability. Three of three seeds pass.
On a fresh 16,384-lifetime audit, the promoted controller scores 91.36%;
deleting one object falls to 60.52%, and disabling inherited latent reads costs
12.16 percentage points.

This learned rung produces a verified compounding gain. On identical unseen
17.1875% contour events, the parent scores 88.57% and fails mastery; the child
scores 90.68% and passes without any training on that rung. It also zero-shot
masters 18.75%. The exact reset-memory control remains near chance.

The controller still requires only one pass per event. Its next gradual
boundary is 20.3125% bars→diamonds morph.

See
`session_records/pair_magnitude_gradual_bridge_2026-07-29/README.md`.

## Latest breakthrough: internal consolidation substitutes for experience

The same 388,191-parameter controller now extends its magnitude concept from
15.625% to 20.3125% bars→diamonds morph without adding a new adapter. It uses
one balanced packet of 128 new lifetimes (768 verifier bits) and 128 rehearsal
lifetimes, then performs 16 private optimizer passes over that fixed evidence.
Three of three seeds pass acquisition, causal reversal, and complete retention
gates.

The organization of experience is causal. A matched 512-fresh-lifetime arm
failed at 89.90% despite using the same 16 optimizer updates. Resetting the
inherited magnitude skill reached 89.08%, one pass over the selected packet
failed counterfactual mastery, and shuffling its verifier outcomes reached
89.48%.

A fresh 32,768-lifetime audit passed at 90.22%; deleting the second object
reduced accuracy to 60.61%, and disabling inherited reads cost 11.71 points.
The child also masters two unseen morph levels through 20.7031%, while the
parent fails. It already runs at one controller pass per event—extra thought
hurts.

The result shows a fixed-size learned representation getting more capability
from the same verifier evidence by internal consolidation. The next exact
frontier is 20.8984375%.

See
`session_records/pair_magnitude_experience_consolidation_2026-07-29/README.md`.

## Latest breakthrough: the next skill uses half the consolidation compute

The fixed 388,191-parameter controller acquired the exact next 20.8984375%
contour from 128 new lifetimes while preserving its entire repertoire. A
prefix-controlled experiment held the experience packet and gate-leak schedule
fixed: four, six, and seven optimizer passes failed the complete gate, while
eight, twelve, and sixteen passed. Eight therefore halves acquisition work
relative to the previous 16-pass recipe without consuming more verifier
evidence.

The eight-pass schedule replicated on three of three seeds at 90.01–90.24%.
Resetting inherited magnitude knowledge reached 88.10%; shuffling new outcomes
reached 89.62%. A fresh 32,768-lifetime audit passed at 90.21%; deleting object
two fell to 60.64%, inherited-read ablation cost 12.09 points, and every older
skill retained.

The separate forward-transfer gain gate did not pass: +0.188 points versus a
pre-registered +0.200 requirement. Although the child mastered two unseen
contours, no new compounding-transfer claim is promoted. The next frontier is
learning a task-agnostic stopping rule rather than using the fixed eight-pass
budget.

See
`session_records/pair_magnitude_half_compute_2026-07-29/README.md`.

## Latest breakthrough: accumulated skill reduces new experience

The fixed 388,191-parameter magnitude controller now stabilizes its first
genuinely unmastered 21.484375% contour from 96 new lifetimes—25% fewer than
the preceding 128-lifetime acquisition rung. It does this by spending twelve
private consolidation passes, preserving the rule that accuracy and retention
come first, external experience second, and private compute third.

The untouched parent failed three of three preflight streams. The selected
recipe passed three of three acquisition runs and a complete 32,768-lifetime
audit at 90.45%. On eight additional paired streams the parent mastered 0/8
and the child 8/8; mean accuracy increased by 0.4677 percentage points and
every stream improved.

The gain is causal. Resetting inherited magnitude knowledge reached 87.95%,
and shuffling the new scalar outcomes reached 89.39%. Both failed. No
parameter was added, and every older cognitive skill retained its gate.

See
`session_records/pair_magnitude_experience_compounding_2026-07-29/README.md`.

## Latest breakthrough: sample efficiency compounds again

The same 388,191-parameter controller stabilized the next harder 22.65625%
magnitude contour from only 44 new lifetimes / 264 verifier bits. The preceding
frontier required 96 lifetimes, so accumulated skill cut new experience by a
further 54.2% while retaining every older capability.

The threshold was searched rather than guessed: 32 and 40 lifetimes failed,
42 passed only 1/3 complete gates, and 44 passed 3/3. Resetting inherited
magnitude knowledge fell to 87.62%; shuffling the new outcomes fell to 88.66%.
A 32,768-lifetime causal audit passed at 90.26%.

Across eight matched streams the untouched parent mastered 2/8 and the child
8/8. Every child stream improved. The acquisition sequence now requires
128 → 96 → 44 new lifetimes on progressively harder contours: two consecutive
verified reductions in external experience.

See
`session_records/pair_magnitude_repeated_compounding_2026-07-29/README.md`.

## Latest breakthrough: reward-grounded cross-operation reuse

The numerosity controller now measurably reuses its learned “choose larger”
intention to acquire a different operation, “choose smaller.” The learner sees
only rendered pixels, its uniformly sampled opaque action, and that action's
scalar outcome.

At 1,024 new lifetimes / 6,144 verifier bits, two fresh seeds reached 57.42%
and 56.98% on 8,192 held-out lifetimes. Matched shuffled-outcome controls
reached 38.48% and 38.21%; matched controls with inherited intention removed
reached 38.53% and 36.29%. The mean truthful advantage is therefore +18.86
points over shuffled outcomes and +19.79 points over no inherited intention.
All three prior skills remained above their inherited gates.

The key mechanism is an attempted-action policy gradient under exact uniform
logging. It uses observed success and failure through a task-agnostic chance
baseline without constructing the unattempted answer. Unlike the earlier BCE
rule, shuffled outcomes no longer drift performance toward chance and masquerade
as learning.

This is cross-operation transfer, not mastery: 57.20% mean accuracy has not
reached a stable competence threshold. The next frontier is to turn the
verified reuse signal into reliable inverse-operation mastery with fewer
lifetimes.

See
`session_records/cross_operation_policy_gradient_2026-07-30/README.md`.

## Retracted result: fixed-operation streamed cue

The controller scored 83.96% and 83.65% on the fixed “choose smaller” task,
but a stronger control has now retracted the interpretation that it read the
operation cue. Keeping the prestimulus timestep while blanking only its cue
pixels retained 81.93% accuracy. The old missing-cue control had also removed
the timestep and therefore confounded visual content with recurrent timing.

The bottleneck was a confounded visual stream. Drawing the operation cue over
the count frame reduced the frozen parent's reusable comparison from 90.46% to
63.15%. The new protocol presents a cue-only visual frame immediately before
each clean count frame. It still exposes only sensory pixels, opaque actions,
and scalar outcomes, but cleanly factors the requested operation from its
arguments.

A compatibility migration also moves the legacy post-intention action residual
back into the amodal intention through the learned actuator's right inverse.
It changed zero of 36,864 audited old-skill actions.

The outcome and inheritance controls remain evidence that the adapter learned
from reward and reused the parent, but they do not establish conditional
operation reading. The corrected experiment below randomizes larger versus
smaller on every event.

See
`session_records/stream_separated_operation_2026-07-30/README.md`.

## Latest breakthrough: causal conditional-operation learning

`visible_pair_numerosity_operation` requests larger or smaller independently
on every event. Every six-event lifetime contains three of each operation, so a
fixed inverse-operation adapter is exactly chance. A counterfactual preserves
all count pixels, flips only the public operation symbol, and complements every
verifier answer.

Two independent 256-update runs reached **81.08%** and **79.68%** held-out
accuracy from 2,048 unique lifetimes / 12,288 scalar verifier outcomes.
Timing-matched blank cues scored **50.18%** and **49.82%**. Relation remained
at 99.20–99.34%, magnitude at 90.42–91.29%, and inherited numerosity at
88.23–88.25%.

A history-free audit resets the controller for every event. It reached 70.26%
and 67.27% under true cue reversal, with 46.25% and 40.11% prediction flips.
The paired blank-cue audit is exactly 50% with zero flips. Shuffled verifier
outcomes scored 49.87%, and removing inherited latent/intention content with
the same parameter count scored 49.78%.

The new task-agnostic interface multiplies a learned recurrent latent by the
inherited amodal intention. At a matched 128-update seed it improved 71.60%
concatenation-only learning to 75.76%. Extending training from 256 to 512
updates improved only 81.08% to 82.60%, so the project stopped buying duration.

This is the first causal reward-grounded selection between opposite operations
on otherwise identical scenes. It is not promoted mastery; the research
checkpoint is explicitly marked unpromoted. The frontier is ≥90% conditional
operation accuracy, especially on independent history-free events, without
lowering inherited numerosity.

See
`session_records/conditional_operation_2026-07-30/README.md`.

## Latest breakthrough: one-event sensory RAM

The controller now carries a generic `latest_event` latent in temporary RAM.
Every sensory step overwrites it; a later skill may read it on the next step.
This is not an operation-specific register and receives no frame type, cue
flag, task ID, or verifier metadata.

At the same 128-update / 1,024-lifetime budget, two replicas reach **84.75%**
and **84.33%** sequential conditional-operation accuracy. Their history-free
counterfactual scores are 78.15% and 73.52%, while timing-matched blank cues
remain at chance. All inherited relation, magnitude, and numerosity skills
remain above 90%.

A matched-capacity control zeroes only the snapshot content. Sequential
accuracy falls from 84.75% to 72.18%; history-free accuracy falls from 78.15%
to 55.54%; causal cue flips fall from 66.95% to 15.27%. A matched-seed
shuffled-outcome run reaches 50.03%.

The selected 256-update candidate reaches **84.82% sequential and 84.58%
history-free**, with 82.25% prediction flips under cue-only reversal. It
retains relation at 99.17%, magnitude at 91.25%, and numerosity at 90.13%.
This closes the recurrent-history distribution gap without adding another
controller pass.

The checkpoint remains explicitly unpromoted because neither mode has crossed
90%. Further duration did not improve the sequential ceiling, so the next
frontier is the shared residual error rather than more recurrent context.

See
`session_records/event_snapshot_operation_2026-07-30/README.md`.

## Latest breakthrough: explicit pairwise latent operation binding

The remaining conditional-operation error was decomposed before another
repair. The inherited numerosity relation is only about 91% accurate, but even
when that relation is correct the elementwise operation binder still loses
roughly ten points. Balanced exploration, residual regularization, longer
prerequisite refinement, full-budget appearance curricula, and additional
duration did not close that gap.

The new generic interface projects the immediately preceding sensory event and
the inherited amodal intention to eight learned dimensions each, then exposes
all 64 outer-product terms to the appended zero-output slot. It supplies no
operation ID, count, answer, task label, or verifier-private state and adds no
controller step.

At 256 updates, two prospective seeds reach **86.05%** and **85.41%**
history-free counterfactual accuracy. Seed 25301 improves over the previous
event-snapshot candidate from 84.58% to 86.05%, while cue-reversal prediction
flips rise from 82.25% to 86.47%. Seed 25311 crosses 85% in both cue directions
after its previous 128-update elementwise run reached only 73.52%
history-free.

A matched module with only outer-product content zeroed falls to 61.95%
history-free and 29.53% flips. Shuffled outcomes reach 52.04%, blank cues are
exactly 50% paired with zero flips, and relation, magnitude, and numerosity
retention all remain above 90%.

This is an explicitly unpromoted architectural breakthrough, not mastery:
sequential accuracy remains 84–85%, below the 90% gate. The error audit now
localizes the next frontier to the asymmetric action-zero boundary and the
91% inherited numerosity ceiling—not missing temporal context or insufficient
pairwise binding.

See
`session_records/outer_product_operation_2026-07-30/README.md`.

## Latest frontier: `next item` and protected plasticity

The controller now masters the first `next item` anchor across all three query
positions (98.33% overall, 97.46% causal conflicts) while retaining
`previous item` at 99.22%. The second anchor was localized to a new problem:
binding the relation to a genuinely independent third memory item.

A target-aligned bridge learned rapidly, rising from 58.12% to 84.33% in
3,072 target outcomes, but overwrote the previous relation. Consolidation
restored the old skill and erased the new one. Frozen adapters preserved old
skills but did not learn. Per-parameter usage/volatility protection produced a
measured stability-plasticity curve but no setting both learned and retained.

The next high-ROI experiment is direction-aware gradient protection on the
successful bridge: redirect only new gradients that conflict with verified
rehearsal gradients. See
`session_records/procedural_shape_next_operation_2026-07-30/README.md`.
