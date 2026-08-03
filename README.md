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

The clean-room Brain Workshop benchmark is the current learning frontier. With
a no-label reconstruction frontend, a generic one-step RAM snapshot, and
reward-only controller updates, the same controller learns visual 1-back
comparisons over 2, 4, and 8 positions. The mechanism also transfers to an
audio-only stream: two independent eight-symbol audio runs reached 67.32% and
67.31% held-out accuracy, while history reset returned to 57.17% and 57.19%
and temporal shuffling to 56.34% and 56.92%. These are causal cross-modality
results, not semantic-label ceilings. The next frontier is simultaneous
vision+audio fusion through the unchanged amodal bus; the full record is in
`session_records/brainworkshop_reward_probe_2026-08-01/README.md`.

The first dual-stream follow-up localized the barrier rather than passing it:
the visual bit reached 100.0%, while audio and joint exact action stayed at
55.3%. A frontend-scale audit found the visual event RMS about ten times the
audio RMS. Optional generic normalization, serial-event transport, and
per-bit credit were tested in tiny runs without a causal promotion. The
evaluator now distinguishes exact joint accuracy from partial factorized
reward, and the decoder warm-start mapping was corrected before interpreting
the result. See
`session_records/brainworkshop_dual_stream_followup_2026-08-01/README.md`.
The follow-up then tried independent RAM relation adapters per stream: vision
still reached 100.0%, audio remained 53.6–56.4%, and joint exact accuracy
remained 53.6–56.4%. A controller-frozen continuation had healthy adapter
gradients but no improvement, localizing the remaining barrier to the
controller-facing source-preserving interface rather than disproving frozen
external learning.

The controller-freeze audit also passed: starting from a visual controller,
the recurrent weights and audio encoder stayed frozen while a zero-initialized
generic RAM-side relation adapter, bus, and decoder learned audio n-back from
scalar reward. Two 256-update replicas reached 67.08% and 67.07%; history
reset returned to 57.19%/57.18% and temporal shuffling to 57.50%/57.10%.
This demonstrates new skill acquisition in the external memory/computation
path without updating the central controller.

The next source-preserving bridge closes a stronger gradual-learning gate. Two
independent reward-only runs froze the controller, both encoders, and the
mastered visual output subspace while training only generic audio RAM bridges
and audio output rows. Fresh 512-lifetime audits retained **100% vision** and
reached **60.2–62.3% exact joint** (audio equal to joint), with history-reset
and temporal-shuffle controls at **33.8–36.3%**. The controller hash was
bit-identical across all 29 tensors; zeroing both learned audio bridges
returned audio to 56.2%. After correcting the reward path to train the new
RAM-to-intention bridge, the strongest 64-update run reached **64.1–65.5%
exact joint** on two fresh audits while retaining **100% vision**; reset and
shuffle returned to **34.6–34.7%**. This is the first replicated frozen-core acquisition
of a second simultaneous stream without forgetting the first. The result is
still a protected two-stream curriculum, not unrestricted N-stream learning;
the detailed record is in
`session_records/brainworkshop_dual_stream_followup_2026-08-01/`.

The next protected-stream rung is now qualified on a three-stream token
variant. Starting from a two-stream parent (vision 100.0%, audio 62.6%), a
generic zero-impact input-confidence gate let a new opaque text/token stream
learn from scalar reward without perturbing inherited paths. A neutral new-bit
initialization plus entropy-preserving exploration now reproduces acquisition
on **three independent seeds**: after 384 updates the text stream reaches
**93.9–94.3%**, while vision remains 100.0% and audio 62.4%. History reset and
cross-episode text swaps return the new bit to **56.2–56.5%**. The inherited
controller, vision encoder, and audio encoder remain bit-identical. This is a
protected three-stream proof of concept, not unrestricted N-stream or natural
language transfer: the token frontend is an opaque protocol encoder and a
learned stopping rule is still open. Full accounting and artifacts are in
`session_records/brainworkshop_three_stream_2026-08-01/README.md`.

The same record now contains the first direct compounding result. A generic
depth-2 RAM bridge let the frozen controller adapt the inherited text primitive
from 1-back to 2-back. At a matched 256-update budget, three inherited seeds
reached **83.8%, 84.3%, and 87.6% eligible accuracy** (mean **85.2%**), while
matched fresh 2-back learners reached **50.6%, 50.4%, and 50.0%** (mean
**50.3%**). Reset-memory and cross-stream controls stayed near 50%. The
evaluator excludes the forced warm-up prefix for n-back metrics, and the
controller remains frozen. This is verified reuse of an old temporal skill to
learn a harder one faster, not only eventual task mastery. The next gradual
rung is 3-back with another generic RAM snapshot.

That 3-back rung is now implemented and audited with a protected skill ladder.
A depth-3 RAM bridge receives verifier-only rehearsal for both earlier
difficulties, while the controller remains frozen. Across three seeds and 256
new-task updates, 3-back reached **85.55%, 85.74%, and 86.23% eligible
accuracy** (mean **85.84%**). Independent retention audits on the same final
checkpoints retained 1-back at **90.90–91.20%** (mean **91.05%**) and 2-back at
**89.10–89.58%** (mean **89.36%**). History-reset and cross-stream controls
remained at chance for every rung. A single rehearsal level preserved 2-back
but let 1-back fall to 84.56%, so multi-level rehearsal is now the promoted
mechanism rather than an unexamined assumption.

This closes the current protected-plasticity milestone: the external RAM path
can acquire a harder temporal relation while retaining the complete earlier
ladder. It does **not** yet prove that 3-back itself is learned faster than a
1-back-only parent: a matched 1-back parent also reached 85.94% after 256
3-back updates. The next frontier is therefore a bits-to-threshold race at
equal starting state, followed by a learned stop/continue policy; final
accuracy alone would hide the sample-efficiency question.

The corrected common-initialization compounding audit now answers the broader
question. At the same 256-update 3-back budget, three mastered 1-back parents
reached **85.94%, 77.48%, and 84.04%** (mean **82.49%**), while three runs from
the same genuinely unadapted controller stayed at **49.39–49.53%** (mean
**49.48%**). Reset and cross-stream controls remained at chance. Thus prior
experience makes a novel, harder temporal primitive learnable with the same
amount of experience. Direct 2-back inheritance did not yet accelerate 3-back
when its bridge was simply expanded, so the next frontier is a monotonic
skill-stack/threshold race that preserves this 1→3 transfer while making
2→3 transfer beneficial rather than merely safe.

A non-destructive relation-only stacked bridge was tested as the first targeted
repair. It froze the mastered 2-back bridge and gave 3-back a zero-initialized
branch seeing only the current event plus the newest relation. Three seeds
reached **86.50%, 85.92%, and 85.66%** (mean **86.03%**) while retaining
1-back at **90.84%** and 2-back at **88.98%**. Reset/cross controls stayed at
chance, but the paired gain over the existing 85.84% baseline was only **0.19
points**, so this branch is not promoted. The result narrows the frontier:
representation isolation alone is insufficient; the missing ingredient is a
better way to route or reuse mastered computations during a harder task.

A learned generic router over that stacked bridge was also piloted. It began
with the inherited branch active and was trained only through the 3-back plus
1/2-back verifier mixture. Seed 47405 reached **86.07%**, with reset and
cross-stream controls at chance—valid plumbing, but no improvement over the
unrouted 85.84–86.50% range. It is therefore not promoted. The remaining
frontier is a more expressive skill-selection mechanism or curriculum, not
another scalar gate.

## Latest compounding rung: 1-back to 4-back

The generic RAM bridge was extended by one more opaque snapshot and audited at
4-back. At an equal 256-update budget, three mastered 1-back parents reached
**84.67%, 75.27%, and 55.35%** (mean **71.76%**), while the same genuinely
unadapted controller stayed at **49.98%, 50.02%, and 49.98%** (mean **49.99%**).
History-reset and cross-stream-shuffle controls remained near chance in every
run. Thus prior experience still makes a substantially harder temporal
primitive learnable; the mean gain over the common unadapted controller is
**21.77 percentage points**. The lower third-seed result is why this is
promoted as a replicated compounding effect, not as a stable 4-back mastery
threshold.

The first protected 4-back pilot mixed 4-back updates with verifier-only
rehearsal of 1-, 2-, and 3-back. It reached **81.37% 4-back**, retained
**89.61% 1-back**, **71.61% 2-back**, and **73.93% 3-back**, with reset controls
near 50%. The 1-back parent baseline was **92.44%**, so the remaining frontier
is a retention-aware 4-back curriculum that keeps the complete earlier ladder
within the existing two-point gate. In other words, the new capability is
real, but zero-loss protected 4-back is not yet demonstrated.

The matched no-rehearsal 4-back runs retained only **65.79–73.06% 1-back** in
the three audits, making the causal value of rehearsal visible rather than
assumed.

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
N=2 remains 98.51–98.55%. A promoted hidden-64 head trained for 256
self-supervised updates scales the same router through N=11: 85.34–86.14%
at N=11, with 49.6% no-agreement controls and gains above 35 points. The fixed
85% gate is not met at N=12 (83.19–83.88%), so N=11 is the current verified
cardinality frontier. Cross-modality relevance and more than eleven streams
remain open. See
`session_records/amodal_input_noise_2026-08-01/README.md` for the rejected
noise-adaptation evidence,
`session_records/amodal_n3_confidence_2026-08-01/README.md` for the interface
gate, and
`session_records/amodal_learned_confidence_2026-08-01/README.md` for the
learned-confidence audit, and
`session_records/amodal_pair_agreement_2026-08-01/README.md` for the learned
relevance breakthrough and
`session_records/amodal_pair_agreement_cardinality_2026-08-01/README.md` for
the gradual N=2→N=11 audit and N=12 boundary.

The complete N-to-M runtime wrapper is now behaviorally qualified on that
promoted complementary skill. Two independently registered external frontends
feed one frozen controller and one decoder bus; the 4,096-lifetime audit
reaches **96.57%/91.13%/95.56%** on bars/diamonds/dot-pairs, with a 512-lifetime
replica at **97.03%/90.27%/95.66%**. Individual streams and shuffled partners
remain near chance, contradictory partners cause the expected flips, and every
wrapper action logit matches the prior explicit bus path exactly (maximum drift
`0.0`). This qualifies the runtime boundary for an existing skill, not natural
audio/language transfer or cold-start cross-modal learning. Evidence is in
`session_records/amodal_runtime_composition_2026-08-03/`.

The next alignment diagnostic qualified a narrower neural-IR learning step. A
second encoder with a deliberately reversed latent basis learned an adapter
from paired unlabeled sensory consistency while the controller, original
encoder, input bus, and decoder were frozen. Two identity seeds and one
random-init seed transferred composition across bars, diamonds, and dot-pairs;
shuffled-partner and contradictory controls passed, and N=1 retention stayed
intact. Matched reward-only arms failed the causal gates and remain negative
controls. This is not yet natural audio/text or cold-start transfer; the
complete record is in
`session_records/amodal_latent_alignment_2026-08-03/`.

The latent adapter was independently saved and replayed at 2,048 lifetimes.
The next raw-modality test also passed: a synthetic audio waveform frontend
aligned to the frozen vision neural-IR basis using paired unlabeled consistency
only. The controller, vision encoder, input bus, and decoder were unchanged;
an independent replay reached **95.21%/88.96%/94.60%** fused on
bars/diamonds/dot-pairs, with shuffled partners near chance and contradictory
flips at **84.85%/69.50%/89.93%**. This is a qualified synthetic
cross-modality alignment result, not yet natural audio, asynchronous input, or
cold-start transfer. See
`session_records/amodal_latent_alignment_2026-08-03/`.

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

## Current frontier: replicated gated continuation for protected 4-back learning

The Brain Workshop ladder now has a replicated, verifier-audited
``learn → check → continue`` loop. Starting from the inherited 3-back parent,
three 4-back runs with private 1/2/3-back rehearsal reached **80.42%**,
**66.99%**, and **81.52%** eligible accuracy on seeds 47408, 47409, and 47405
at 256 updates (76.31% mean). Reset and time-shuffle controls stayed near
chance. The weaker seed was continued for only 64 additional updates after it
crossed the acquisition gate; it reached **77.00%** at 320 updates, for a
79.65% mean across the three gated final checkpoints.

Frozen-checkpoint audits retained 1-back at **94.29%**, **94.61%**, and
**94.56%** (parent baselines 93.96%, 94.32%, and 94.24%), all within the
two-point retention gate. Each 256-update stage consumed 65,536 target-stream
verifier bits; the continued seed consumed 81,920 bits total before reaching
77.00%. The new ``--rehearsal-weights`` option permits verifier-side per-rung
weighting, but the evidence does not support fixed weights as universally
superior to the uniform baseline. The promoted idea is the gated continuation
decision, driven by held-out progress and retention—not unconditional extra
training. The reusable audit is
`experiments/unified_cognitive_controller/audit_nback_continuation.py`.
The complete reports and negative-control provenance are in
`session_records/brainworkshop_three_stream_2026-08-02/README.md`.

This is a protected-plasticity/sample-efficiency breakthrough, not yet a
general learned stopping policy: the continuation rule is still a small
verifier-side controller. A first 8-update 5-back compatibility probe now
executes with depth-five RAM but remains at 46.88% eligible accuracy, so the
gate correctly stopped before a costly run. The next frontier is calibrating
the stopping decision from early learning progress, then revisiting 5-back
only when that signal justifies the longer budget while preserving the same
retention and causal gates. The training telemetry now exposes
warm-up-excluded `batch_eligible_accuracy`, so that calibration is based on
the actual target-bearing trials rather than an optimistic mixed score.

## Latest breakthrough: protected fifth-back learning and reward fine-tuning

The depth-five interface is now more than executable. A verifier-label
diagnostic reached **93.03%** eligible 5-back held-out accuracy after 8,192
unique lifetimes (256 updates), with reset **50.00%** and time-shuffle **49.74%**.
This establishes the representational ceiling while keeping the diagnostic
labels out of inference.

The new `--freeze-inherited-history` ablation freezes the inherited controller,
decoder, and old RAM columns, exposing only the appended fifth-history input
columns to training. That protected adapter reached **60.55%** eligible
5-back accuracy at the same 256-update budget; reset and time-shuffle controls
were **49.35%** and **51.43%**. A zero-learning-rate retention audit kept
1-back at **94.08%** (reset **50.08%**). Thus the extra skill can be added through
an additive RAM-side path without overwriting the old path, although the
protected representation needs more efficient discovery than the ceiling
probe.

The strongest sample-efficiency result is the next eight verifier-reward
updates: protected 5-back rose from **60.55%** to **78.91%** while 1-back stayed
at **93.53%**. Reset and time-shuffle remained **49.35%** and **52.34%**. This is
honestly a **supervised-bootstrapped reward continuation**—not cold-start
reward-only discovery—but it demonstrates the desired compounding pattern:
discover a reusable new memory path, improve it from sparse verifier reward,
and preserve the earlier skill. Reports are in
`session_records/brainworkshop_three_stream_2026-08-02/`; the next frontier is
closing the sample-efficiency gap to the 93% ceiling and then testing transfer
to a genuinely different cognitive primitive.

The next fresh-seed eight-update continuation reached **80.86%** on a larger
3,072-episode target-bearing evaluation; reset and time-shuffle controls were
**50.10%** and **50.78%**. A zero-update 1-back audit remained **93.14%**. The
gain over 78.91% was below the +5-point continuation gate, so the run stopped
instead of spending more compute on a weak slope. This confirms a stable
protected reward-learning curve, while the cold-start reward-only question and
the gap to the 93% supervised ceiling remain open.

## Latest breakthrough: protected sixth-back compounding

The sixth temporal rung is now supported by the generic RAM bridge. An
unprotected supervised ceiling run reached **80.86%** 6-back, but destroyed
1-back retention (61.16%), so it was rejected. The protected extension trains
only the appended sixth-history columns from the inherited 5-back checkpoint.
It first reached **57.42%** 6-back with 1-back at **93.28%**; the 5-back audit
exposed a task-conditional interference cost, falling to 72.66% when the new
history features were active.

Eight verifier-reward updates with 5-back rehearsal then raised 6-back to
**72.27%** on the check evaluation and **71.24%** on a larger evaluation with
2,048 target-bearing trials. Reset and time-shuffle controls were **50.00%**
and **49.95%**.
The earlier skills remained intact: 5-back **80.86%**, 1-back **92.75%**.
This is the first protected 1→5→6 compounding result. It is a
**supervised-bootstrapped reward continuation**, not a cold-start reward-only
claim. The remaining frontier is a task-conditional, task-agnostic memory gate
that closes the sixth-history interference margin, then 7-back or transfer to a
new cognitive primitive.

A fresh eight-update continuation with 5-back rehearsal weight **0.25** then
raised 6-back to **77.34%** on a larger 2,048-target-trial evaluation; reset
and time-shuffle were **50.00%** and **49.66%**. 5-back and 1-back stayed at
**80.86%** and **92.22%**. This improves the protected sample-efficiency curve
without opening the inherited parameters. The next frontier is to close the
remaining sixth-history interference and approach the 93% ceiling, then test a
new primitive or cold-start reward learning.

Two short follow-up forks were rejected rather than promoted. Repeating the
same 0.25-weight continuation from the 77.34% checkpoint and a reward-trained
router both settled near **77.15%**. A zero-initialized stacked
relation/router branch degraded 6-back from **76.37%** to **65.04%** at 128
updates; the same branch trained from the 5-back parent stayed at **49.61%**.
This rules out "add another router" as the next move, despite healthy
gradients. The promoted checkpoint survived a larger no-update audit at
**77.16%** over **32,768 target-bearing trials**, with reset **50.00%** and
time-shuffle **49.21%**. The next experiment must make the RAM write
conditional on requested history depth (or otherwise isolate sixth-history
features from 5-back), while preserving the existing retention and causal
gates.

## Latest breakthrough: protected seventh-back compounding

The generic RAM bridge now extends to seven opaque snapshots. A protected
256-update diagnostic reached **54.30%** eligible 7-back from a 47.66% parent.
An eight-update reward continuation with 6-back rehearsal weight 0.25 rose to
58.59%, but reduced 6-back retention to 71.44% and was rejected. Raising only
the verifier-side rehearsal weight to **1.0** fixed the interference: 7-back
rose from **57.81%** to **68.75%** in eight reward updates.

A larger no-update audit measured **68.36%** over **8,192 target-bearing
trials**, with reset **50.00%** and time-shuffle **55.18%**. Independent
retention audits passed: 6-back **76.32%**, 5-back **80.14%**, and 1-back
**91.99%**, each over 8,192 trials with reset/time-shuffle controls near
chance. This is a protected 1→5→6→7 compounding result, still honestly
labeled supervised-bootstrapped reward continuation rather than cold-start
reward-only discovery. The next frontier is 8-back or transfer to a different
cognitive primitive.

## Protected eighth-back representational rung

The generic RAM bridge now accepts eight opaque snapshots. An eight-update
reward compatibility probe stayed at chance, so the matched 256-update dense
diagnostic was the justified escalation. It reached **60.55%** eligible
8-back from a **46.88%** parent, with reset **50.00%** and time-shuffle
**44.92%**. An eight-update reward continuation did not improve the diagnostic
point and was stopped by the progress gate.

The eighth-history path nevertheless preserved the full ladder on independent
8,192-trial audits: 7-back **68.55%**, 6-back **76.07%**, 5-back **80.11%**,
and 1-back **91.95%**, with reset/time-shuffle controls near chance. This is a
protected eighth-back representational breakthrough, not a reward-only
mastery claim. The next high-ROI question is how to turn the decodable
eighth-back relation into reward-fine-tuned behavior without sacrificing the
ladder.

## Relational-reader integration boundary (2026-08-02)

The next experiments tested that boundary with a fixed, retention-safe parent
(`brainworkshop_three_stream_nback8_protected_supervised_seed47820_256.pt`).
All runs used `--target-modalities text`, factorized opaque output, eligible-only
loss (the first eight warm-up trials contribute no gradient), and independent
time-shuffle/reset controls. This matters because the earlier mixed-modality
score could improve while the target-bearing n-back relation stayed at chance.

The direct recurrent relational gate produced a real but sub-threshold held-out
signal: 16 updates reached **57.42%** eligible accuracy (+0.98 points), 64
updates **59.18%** (+2.73), and 128 updates **59.96%** (+3.52). The 128-update
time-shuffle and reset controls were **48.05%** and **50.00%**. The gate's
gradients were alive, but the pre-registered continuation bar is +5 points;
none of these runs qualified for a longer capability claim.

Three targeted attempts to close the gap were rejected. A disposable
verifier-side auxiliary head on the relation residual reached **59.57%** at 64
updates (+3.13), so forcing the residual to predict the target did not solve
the reader problem. A zero-initialized additive opaque output adapter reached
**57.81%** at 16 updates (+1.37); freezing the inherited decoder and training
only that adapter plus the gate reached **57.23%** (+0.78). Both had
time-shuffle near chance. These are useful negative controls: the relation is
present, but a simple residual-to-output path does not make it behaviorally
usable.

Optimization sweeps exposed a shortcut rather than a solution. At 16 updates,
learning rate **1e-3** reached **61.13%** (+4.69) with a clean 48.83%
time-shuffle control, but the matched 64-update run fell to **57.42%**. At
**3e-3**, the 64-update run fell to **54.49%** while its mixed training score
rose, a clear overfit/shortcut warning. AdamW weight decay **1e-4** did not
improve the 16-update result (57.42%). The eligible-only mask, LR changes,
auxiliary head, and output adapter are retained as reproducible diagnostics,
not promoted capability.

This closes the current relation-reader fork. The project has demonstrated a
protected, causal, decodable eighth-back representation and a robust
1→5→6→7 compounding ladder, but not yet a retention-safe 8-back behavioral
breakthrough. The next design must address the representation-to-action
credit-assignment boundary (for example, a task-agnostic reader trained on a
larger diverse lifetime cache or a recurrent snapshot binder with an explicit
state-preservation objective) before spending another long run. Any candidate
must beat the +5 eligible gate and pass reset, time-shuffle, cross-stream, and
full-ladder retention audits.

One final 16-update control appended the frozen controller's opaque intention
state to the relation-gate input. It reached **57.81%** eligible accuracy
(+1.37) with a **47.85%** time-shuffle control. The controller state is useful
information in principle, but simply exposing it does not close the reader
gap, so this branch is also rejected for longer training.

## Latest memory-bank hardening (2026-08-03)

The bounded hot/cold skill bank now records a SHA-256 for every opaque disk
artifact and verifies it before promotion. Legacy manifests remain loadable
and are upgraded on their next save. Promotion also exposes optional generic
confidence and top-row-margin thresholds: when an address is ambiguous or
weak, the bank can abstain instead of silently activating an unrelated skill.
Tampered-artifact and ambiguous-address controls are covered by regression
tests. This hardens the long-term-memory boundary; it does not yet claim that
the controller has learned a multi-skill selector or solved the protected
eighth-back reader.

## Reward-only address-routing diagnostic (2026-08-03)

A small selector and audit harness now tests whether a frozen controller can
learn to route among opaque skill-bank rows from only attempted-row identities
and scalar verifier outcomes. The first three-row run was deliberately not
accepted: normal routing was **100.00%**, but the reward-shuffled control was
also **100.00%**, matching the fixed cosine baseline (**100.00%**). Candidate
permutation was **100.00%** and the controller remained bit-identical. This is
useful negative evidence, not a capability result: the controller-produced
query/key geometry already exposes the correct row, so the reward-only claim is
not identifiable in this setup. The next diagnostic must remove that static
shortcut (for example with episode-level address scrambling) before a learned
selector can be promoted.

## Reward-only routing with opaque addresses (2026-08-03)

The shortcut was then removed by replacing controller-aligned candidate keys
with fixed random opaque addresses. The controller remained frozen, and the
selector saw only controller queries, candidate keys, the attempted row, and
the scalar outcome. A three-row run reached **100.00%** after 32,768 verifier
bits, while reward shuffling and cosine similarity both stayed at **33.33%**;
candidate permutation stayed at **100.00%**. A second seed required a larger
budget—**65,536 bits** and 16 training queries per skill—to reach the same
100/33/100/33 pattern. At that budget, a four-row run reached **100.00%** with
the shuffled and cosine controls at **25.00%**. This is the first clean
reward-dependent routing diagnostic, not yet a production bank change: the
seed-variance and sample-efficiency curve must be measured across more skills
and then integrated with behavioral and retention audits.

## Reward-routed real skill-bank integration (2026-08-03)

The selector was then exercised through the actual hot/cold bank rather than
only tensors. Real span-nine and span-ten successor-slot artifacts were saved
behind random opaque addresses, the bank was reloaded, and 64 held-out queries
were promoted through the reward-trained selector. Routing was **64/64**;
reload rows were exact; and the rehydrated behavior matched the direct child
at **91.45%** for span nine and **86.64%** for span ten. Wrong-skill controls
fell to **86.59%** and **79.96%**, respectively. Reward-shuffled and cosine
controls were both **50.00%**, candidate permutation was **100.00%**, and the
controller stayed frozen. This validates the plumbing and causal artifact
selection, but promotion remains opt-in until multi-seed retention and online
task-shift audits are added.

## Online selector retention: replay distillation (2026-08-03)

The first continual-routing experiment exposed the expected stability/plasticity
failure: after skill 9 was mastered, new-skill-only updates learned skill 10
at **100%** while old skill 9 fell to **0%**. A small task-agnostic replay
distillation term preserved the old selector outputs while scalar outcomes
trained the new skill. Two seeds retained skill 9 at **100%** and learned skill
10 at **100.00%** and **95.31%**; shuffled rewards left the new skill at
**0%**, while candidate permutation remained exact. This is a replicated
router-level anti-forgetting result, not yet a production claim: the next gate
is to apply the same online update to real bank artifacts and audit behavioral
retention after multiple task shifts.

An independent seed reproduced the routing gates (16/16 held-out queries per
skill, 100% normal, 50% shuffled/cosine, 100% permutation) and direct-vs-routed
behavior equivalence at 93.49% for span nine and 87.66% for span ten. The
result is now replicated at the plumbing level; the remaining promotion gate
is whether online task shifts and repeated acquisition preserve the older
artifact behavior under a learned selector.

## Online retention through disk-backed replacement (2026-08-03)

The distilled update was exercised through a real bank lifecycle: an old
span-nine artifact and an unused opaque address were saved, the placeholder
was replaced by span ten, and the bank was reloaded before promotion. The
naive new-only arm learned span ten at **100%** but erased span nine (**0%**).
The distilled arm retained span nine at **100%** and learned span ten at
**100%**, with shuffled rewards leaving the new route at **0%**. Both rows
selected correctly after replacement; direct and routed behavior matched at
**92.45%** and **88.05%**, respectively. This is a diagnostic milestone for
memory-side continual learning, not yet a general production policy: broader
task shifts and longer retention windows remain to be audited.

## Replay rebuild across three opaque skill rows (2026-08-03)

The next routing audit compared two ways to add a third skill family while the
controller stayed frozen: incremental updates with output-distilled replay,
and rebuilding the small selector from accumulated opaque query/attempt/outcome
replay. The selector never received span identities or correct-row labels;
the verifier retained those privately. At 1,024 updates per arm (65,536
verifier bits), both the incremental and replay-rebuild arms reached
**100% on all three held-out skill families** after the final shift. The
rebuild arm also mastered the first two rows before the third row was added.

The independent-random-outcome rebuild null scored `[0%, 100%, 0%]` across
the three rows (33.3% aggregate), while the candidate-row permutation control
remained **100%**. The controller digest was unchanged. A smaller 512-update
pilot was seed-sensitive and failed to master the middle row, so this is a
bounded compute/data result rather than evidence that rebuilding is always
better than online distillation. It does show that an ever-growing external
replay bank can support selector reconstruction without changing controller
weights or exposing semantic labels.

The full report is
`session_records/sequence_working_memory_2026-08-02/skill_bank_router_rebuild_seed93401.json`.
The next gate is repeated multi-seed three-row behavior through the real
disk-backed bank, followed by a longer sequence of replacements and retention
audits; this diagnostic alone does not promote learned routing to the default
cosine resolver.

## Successor-slot extension probe: span eleven (2026-08-03)

To close the gap between the abstract three-row routing diagnostic and a real
three-artifact bank, a new utility now appends one zero-output successor slot
to the cumulative span-ten controller and trains only that slot. Insertion is
bit-identical before training, so the experiment cannot hide a regression in
the inherited controller.

The first short rungs did not yet justify promotion. A 16-update MPS run
showed a **+1.85-point** span-eleven movement, but the zeroed-slot control
returned exactly to baseline and the older span-nine/span-ten streams stayed
within **−1.56/+0.31 points**. A 32-update replica showed only **+0.85
points**. Reusing each tiny batch eight times produced a smaller **+0.50-point**
movement while violating the two-point retention gate (−2.56 and −1.48
points). Output distillation from the frozen parent preserved old behavior,
but at the tested 8–16-update budgets it produced no measurable new-slot gain.

These are useful bounded negatives: the slot interface and causal zeroing
control work, but span eleven has not yet learned strongly enough to create a
real third cumulative artifact. The extension remains an unpromoted tool and
all reports are preserved as `span11_slot_extension_*.json` in the sequence
session record. The next efficient fork is a retention-constrained frontier
curriculum (fresh target batches, no repeated-batch shortcut) before spending
enough verifier experience to build a production three-artifact bank.

The follow-up activity probe explains the failure: the unregularized new slot
opened broadly on inherited and target streams. A task-agnostic old-stream
residual penalty reduced that leakage and kept retention within the gate, but
two short penalty runs still produced only **+0.36 points** with no causal
zeroed-slot separation. This branch is therefore paused at a correctly
instrumented negative rather than being scaled into an expensive blind run.

The immediate-prior read variant was also tested: it preserved insertion and
retention, but 16 and 32 fresh updates produced only **+0.85/+1.92 points**
with the zeroed-slot controls still at baseline. A prior-slot interface is
therefore mechanically available, but it has not yet converted into measured
sample-efficient span-eleven learning.

A gradual target curriculum (zero distractors/fixed positions, then a small
position-and-distractor ramp) preserved retention but produced only **+0.07
points** at 10,240 fresh verifier bits. The next-span branch is now paused;
the evidence points to a deeper credit-assignment or representation bottleneck,
not justification for simply extending the same run.

## Decisive successor-input probe (2026-08-03)

Before changing the controller again, a frozen diagnostic extracted the raw
input to the appended slot's first linear layer on lifetime-disjoint span-11
episodes. Throwaway probes decoded the correct action at **84.66% linear** and
**87.71% with a small MLP**; an independent random-label null stayed at
**50.57%**, with a second prior-slot variant at **87.43% / 48.15%**. The
controller and slot weights were unchanged and the diagnostic heads were
discarded.

This localizes the current failure: the relevant next-action information is
already present at the new slot's input. The short reward-only runs are failing
to discover/use it, rather than proving a missing visual representation. The
next high-ROI branch is therefore credit assignment—dense use of the same
verified outcome, action-conditioned critics, or a carefully isolated
auxiliary diagnostic—while preserving the two-point retention and shuffled,
blank, reset, and reversal gates. The full reports are
`span11_slot_input_probe_*.json` in the sequence session record.

A temporary action-conditioned success critic (trained only from the attempted
action's scalar outcome, then discarded) was tested at weights 0.5 and 5.0.
Both preserved old skills, but produced only **+0.99/+0.71 points** with no
zeroed-slot causal separation. This is a bounded negative for the simplest
critic auxiliary; the representation probe says the next useful change must
improve how that information drives the slot/action update, not add another
encoder. The two unpromoted critic checkpoints are retained under
`artifacts/checkpoints/span11_slot_extension_critic*.pt` for inspection.

## Successor replay-credit sweep (2026-08-03)

The next diagnostic branch moved successor-slot learning onto a frozen latent
replay buffer containing only controller-visible features, opaque attempted
actions, and scalar outcomes. The collector was corrected to include inherited
workspace, usage, event-snapshot, and age reads automatically; a regression
test now checks the event-snapshot path. A detached action-conditioned critic,
an optional binary-complement bandit loss, and a nonlinear successor gate were
tested as small, explicitly unpromoted controls.

The best safe arm (seed 93712: hidden gate, binary outcome loss, binary critic
loss, detached critic bridge, and persisted old-span replay) produced a
**0.89-point causal span-11 gain**, with span-9/span-10 retention changes of
**−1.04/−0.43 points**. Its shuffled-outcome control collapsed and the
zeroed-slot replay returned to the parent, so the signal is real but below the
pre-registered **5-point causal promotion bar**. More data, on-policy replay,
higher learning rates, and 32× reuse did not improve that bound; hard span 11
also regressed when distractors were restored after an easier no-distractor
arm. The complete sweep is recorded in
`session_records/sequence_working_memory_2026-08-02/span11_replay_credit_assignment_2026-08-03.md`.

This closes the current “add another critic/gate knob” fork. The earlier input
probe still shows **84.66% linear / 87.71% MLP** action decoding at the slot
input, so the remaining gap is reward-to-output credit and task difficulty,
not missing sensory information. The next high-ROI move is a smaller
intermediate primitive or an explicit per-output curriculum, not a longer
blind span-11 run.

## Adjacent complement primitive (2026-08-03)

Before spending more compute on the difficult span-eleven successor, the
controller was tested on a smaller adjacent primitive. A third, visibly
distinct operation cue required the complementary binary action (`1 -
sequence`). The new successor slot was trained from controller-visible latent
features, attempted opaque actions, and scalar verifier outcomes only; no
correct action or semantic rule was placed in the replay buffer.

The longer truthful arm (seed `93748`, 256 fresh lifetimes, 128 epochs, two
distractors) reached **59.77%** on an independent 1,024-lifetime audit, versus
**50.64%** for its parent and **50.64%** when every parameter in the appended
slot was zeroed. This is a **9.12-point causal gain**, above the registered
five-point promotion bar. The operation-cue-blank control returned to 45.16%
and the complete-memory reset to 50.00%. Span-nine and span-ten retention
changed by only −0.01 and −0.04 points, within the two-point gate. A matched
outcome-shuffled child scored 47.56%, so the gain does not survive destroying
the scalar reward correspondence.

The result replicated with the same recipe: seed `93750` reached **58.12%**
with a **7.51-point** causal gain over its zeroed-slot control, while its
outcome-shuffled replica scored 47.47%. Span-nine/span-ten retention changed by
−1.23/−0.86 points. This is now a replicated but still partial acquisition
result, not mastery: 58–60% is not a deployment threshold. The candidate and
replica checkpoints are archived under
`artifacts/checkpoints/complement_slot_*.pt`; the independent audits are
`session_records/sequence_working_memory_2026-08-02/complement_slot_audit_seed294800.json`
and `complement_slot_audit_seed294950_with_shuffled.json`, with the full
interpretation in
`session_records/sequence_working_memory_2026-08-02/complement_slot_2026-08-03.md`.
The next efficient gate is protected continuation with fresh logical
lifetimes: measure accuracy gained per verifier bit without exceeding the
retention gate. Span eleven remains paused until that curve is measured.

### Complement data/retention frontier

The follow-up curve shows why the retention gate remains essential. Increasing
fresh complement lifetimes raised independent accuracy from 59–60% at 256 to
61.51% at 512 and 66–69% at 1,024. At 1,024, however, the new slot's
context-selective plasticity was seed-sensitive: ordinary penalties, replay
weighting, and provenance supervision each produced both passing and failing
retention arms. A 512+512 old-task rehearsal arm preserved span nine/span ten
exactly but reduced new-task accuracy to 62%, which is a poor sample-efficiency
tradeoff.

The best single diagnostic arm (seed `93763`, strong generic provenance gate)
reached **69.19%** with an **18.67-point** causal gain and old-skill changes of
−1.88/−1.81 points. Two matched seeds (`93764`, `93765`) failed retention by
large margins, so no 1,024 checkpoint is promoted. The complete curve,
including rejected arms, is recorded in
`session_records/sequence_working_memory_2026-08-02/complement_data_scaling_2026-08-03.md`.
The next frontier is a robust, context-selective plasticity constraint—not
2,048 more target lifetimes. Follow-up controls (extra rehearsal, gate-only
calibration, and a smaller slot) are archived in
`session_records/sequence_working_memory_2026-08-02/complement_retention_controls_2026-08-03.md`;
none produced a robust promotion.

The first task-agnostic context fork is now wired but remains negative. A
successor slot can read the inherited controller's action vector before its
own residual is applied; this is intended as a generic parent-confidence
signal for the plasticity gate. With raw action logits at 512 target
lifetimes (seed `93770`), independent complement accuracy was **51.94%** versus
50.48% with the slot zeroed, only **+1.46 points** causally. It therefore
misses the +5-point bar and is not promoted. The code and audit are retained
for the normalized-probability follow-up, documented in
`session_records/sequence_working_memory_2026-08-02/parent_action_context_2026-08-03.md`.
The matched probability-simplex follow-up (seed `93771`) reached **53.29%**
with only **+2.35 points** of causal gain, while span-nine/span-ten retention
fell by **6.76/12.53 points**. Both representations therefore fail the causal
and retention bars. Stop expanding this feature list and return to a genuinely
selective gate or an explicit promotion/rejection population.

### Complement margin-loss frontier

Replacing the complement learner's ordinary binary cross-entropy with a
constant-gradient binary margin objective improved the 512-lifetime arm while
preserving the audits. Two independent seeds reached **63.52%/61.59%** with
**+13.04/+10.88-point** causal gains and span-nine/span-ten changes within
−1.55/−1.54 points. A matched shuffled-outcome control returned to **50.09%**.
This is the current partial, retention-safe acquisition checkpoint—not
mastery. Escalating to 1,024 lifetimes reached 66.7–68.1%, but three of four
margin/seed arms violated retention; one safe arm is diagnostic only. Protected
continuation of the learned slot added at most 0.76 points, so repeated
experience has not yet produced a strong second compounding gain. The complete
curve and rejected controls are recorded in
`session_records/sequence_working_memory_2026-08-02/complement_margin_frontier_2026-08-03.md`.

### Complement population-selection frontier

To reduce seed-sensitive interference, three complement-slot clones were
raced on the same target stream (`--data-seed`) and selected by a private
causal/accuracy/retention screen. The selected child then faced a larger
untouched audit, reset-memory control, and matched outcome-shuffled control.
The 1,024-lifetime race replicated twice on disjoint streams: the promoted
children reached **66.81%/+16.28 points** and **67.64%/+16.90 points** causal
complement performance, with span-nine/span-ten retention drops bounded by
−1.97/−1.61 and −1.58/−1.22 points. Full reset controls were exactly 50.00%,
and shuffled controls returned to 51.68% and 50.50%.

This is a robust population-promotion result, not yet a sample-efficiency
win: three clones cost about three training exposures. A 2,048-lifetime
replicate improved complement accuracy to 69.88% and causal gain to +19.05
points, but violated the span-nine retention gate (−2.67 points), so it was
rejected. The private reset check is advisory at 256 samples because it is too
noisy; full promotion still requires the reset band. Reports and the exact
selection logic are in
`session_records/sequence_working_memory_2026-08-02/population_races_2026-08-03/`.

### Continuation-selection and entropy-gate audit (2026-08-03)

The next compounding test was deliberately small: continue the promoted
complement slot with 256 fresh lifetimes and rehearse spans nine and ten.
Three clones all appeared about 2.0–2.34 points better on private 256-lifetime
audits, but the selected child fell from **67.68% to 62.04%** on the larger
independent audit; checkpoint averaging was worse at **61.00%**. A gentler
512-lifetime continuation preserved old skills but changed complement by only
−0.19 points. This exposes private-screen selection noise and is not a
compounding claim.

Appending a fourth slot with 256 lifetimes also fell (67.24% → 64.84%). A new
task-agnostic normalized parent-action entropy scalar was implemented as a
checkpoint-compatible gate input, but its first append arm fell from 67.35%
to 55.96%. These are archived negative controls, not promoted models. The
replay trainer now rejects latent-buffer feature-width mismatches explicitly.
Full reports are in
`session_records/sequence_working_memory_2026-08-02/continuation_and_gate_frontier_2026-08-03/`.

### Verified fourth-slot complement compounding (2026-08-03)

Starting from the promoted three-slot parent, a new zero-impact fourth slot
trained on 1,536 fresh complement lifetimes with span-nine/span-ten rehearsal
and stronger generic residual/gate/logit penalties. The promoted candidate
reached **71.90% vs 66.58%** on its first full audit (**+5.31 points**), with
reset 49.96%, blank 49.98%, shuffled control 54.48%, and old-span changes
of +0.04/−0.01 points. Two additional independent audit seeds gave
**+5.95/+5.80 points**, so the checkpoint itself is robust rather than a
single audit-seed fluke. The checkpoint is
`artifacts/checkpoints/complement_population_fourth_slot_seed93871.pt`.

The same recipe from a second training seed produced +2.97 points and missed
the promotion bar, so seed-to-seed acquisition variance remains. This is a
verified single-checkpoint compounding milestone—not a claim that every seed
will succeed—and the complete reports are in
`session_records/sequence_working_memory_2026-08-02/fourth_slot_compounding_2026-08-03/`.

The next 1,536-lifetime fifth-slot trial saturated at only **+0.46 points**
(72.74% vs 72.29% parent), and the fourth-slot child showed no measurable
zero-shot improvement on span 11 (both 73.44%). This closes the same-task
slot-expansion branch for now: the next high-ROI move is a new primitive or a
better span-eleven credit path, not more complement slots.

The immediate span-eleven append was also rejected under the retention gates.
With 256 target lifetimes, span nine/ten fell −6.63/−4.78 points and span 11
fell 1.59 points. With 1,024 targets, span 11 rose 2.15 points, but span
nine/ten fell −3.53/−4.96 points. Blank and full-reset controls remained near
chance. This closes the current append recipe rather than proving that span 11
is impossible: the next high-ROI move is explicit task-agnostic credit
assignment or a smaller intermediate primitive, with old-skill retention
remaining a hard gate. The complete reports are in
`session_records/sequence_working_memory_2026-08-02/span11_from_fourth_frontier_2026-08-03/`.

### Successor-slot routing smoke (2026-08-03)

A matched sub-minute test checked whether an easier-prefix query curriculum
could make the span-eleven successor slot selective. With 64 fresh target
lifetimes and 64-lifetime span-nine/span-ten rehearsal, the no-curriculum arm
had only **+1.21 points** of causal gain over its zeroed-slot control. The
staged prefix arm had **0.00 points**. Neither approached the +5-point
promotion bar, so it was not scaled.

A separate feature audit found the new slot's gate opening broadly on old
mixed spans (roughly 0.76–0.79) instead of remaining closed; the target stream
opened it more strongly, but the overlap explains the retention damage. Parent
action entropy was near zero and event-age ranges overlapped, so those simple
context signals did not provide a selector. The immediate frontier is generic
slot selectivity / credit routing, not more target data or a longer blind
span-eleven run. The full smoke reports and hashes are in
`session_records/sequence_working_memory_2026-08-02/suffix_routing_smoke_2026-08-03/`.

### Protected-rehearsal mask correction (2026-08-03)

The reward-buffer trainer had a retention-control bug: freshly collected
span-nine/span-ten rehearsal rows were marked as fresh, so replay/gate
penalties never reached them. The opt-in `--protect-rehearsal` mask now marks
non-target rehearsal rows as protected while leaving the historical default
unchanged. Its unit test covers target, rehearsal, and persisted-row layout.

The corrected 64-lifetime smoke produced **+2.13 points** of causal gain with
span-nine/span-ten changes of −1.56/−0.59 points. A 256-lifetime escalation
produced only +1.53 points; the matched outcome-shuffled control produced
−7.48 points, so the small real signal is reward-dependent but below the +5
promotion bar. A strong gate penalty preserved old behavior but produced zero
new learning. This fixes the experiment's interpretation; it does not solve
span eleven. The full reports and controls are in
`session_records/sequence_working_memory_2026-08-02/protected_rehearsal_routing_2026-08-03/`.

## Latest breakthrough: span-three working-memory compounding (2026-08-02)

The sequence branch now demonstrates the desired learn-to-learn effect on a
harder primitive. Starting from the promoted robust two-item checkpoint,
two matched 128-update span-three runs reached **93.55%** and **93.42%**
held-out accuracy after 24,576 fresh verifier bits. A fresh controller at the
same budget reached only **74.98%** and showed an operation-blind shortcut
(0.20% valid operation-reversal flips). The inherited runs crossed the stable
90% gate at 21,504 and 18,432 bits; the fresh arm did not cross it, giving a
conservative lower-bound sample-efficiency gain of **1.23x**.

The first replicas revealed that span-three training could damage the older
span-two skill (80.23% and 90.87% retention). The smallest targeted repair was
to alternate span-two and span-three episodes, rather than rehearsing only a
less-distracting span-three stream. After 64 updates (32 per span), the
promoted checkpoint scored **100.00%** on span two and **95.75%** on span
three on independent 8,192-episode audits. Blank-sequence and complete-memory
reset controls remained at chance, valid reversal produced 66.67% flips, and
position blends remained at 95.74%. A shuffled-outcome span-three control was
50.00% with zero flips.

This is a verified protected 2→3 working-memory compounding result, not a
claim of generic variable-capacity memory or cold-start reward-only discovery.
The new `--rehearse-span2` option makes the retention repair reproducible.
Full reports and the promoted checkpoint are in
`session_records/sequence_working_memory_2026-08-02/README.md`. The next
high-ROI experiment is a one-axis span-four escalation with explicit
span-two/span-three rehearsal and the same adversarial gates; more duration on
span three is not justified.

### Replicated three-skill cold-bank routing (2026-08-03)

The reward-trained external-bank selector now routes among three real skill
artifacts (span nine, ten, and diagnostic-only eleven) with 100% held-out
routing in two independent 1,024-update runs. Reward-shuffled and cosine-only
controls remained at the 1/3 baseline, candidate permutation was 100%, bank
reload was exact, and routed behavior exactly matched direct artifact
rehydration. The 256/512-update failures are bounded budget negatives, not an
architecture rejection. This advances the cold-bank architecture, but does not
yet prove cold-start skill acquisition, long-horizon retention, or span-eleven
mastery. See
`session_records/sequence_working_memory_2026-08-02/three_skill_real_bank_routing_2026-08-03/`.
