# Unified cognitive controller

> **Architecture status:** this experiment remains vision-only, not the final
> modality boundary. The normative target is
> `N encoders -> amodal event bus -> one controller/memory -> intention bus ->
> M decoders`, specified in
> [`../../docs/AMODAL_N_TO_M_ARCHITECTURE.md`](../../docs/AMODAL_N_TO_M_ARCHITECTURE.md).
> The legacy class retains its bundled API, while `ExtractedAmodalRuntime` now
> owns the vision encoder, controller, and decoder separately and reproduces the
> current checkpoint exactly. Multiple output backends consume one intention
> simultaneously, and a generic set bus now composes synchronous N=1/N=2 visual
> events. Delayed/asynchronous and cross-modality composition remain unproven.
> Historical results below must not be described as audited multimodal transfer.

## Extracted neural-IR migration rung (2026-08-01)

The first four migration gates are implemented without retraining:

- `AmodalEvent` and `IntentEvent` have versioned opaque transport schemas;
- `ExtractedAmodalRuntime` owns three disjoint modules: external vision
  frontend, controller core, and external action decoder;
- `UnifiedCognitiveController.step_event()` accepts an already encoded event;
- legacy checkpoints convert into three independently loadable state dicts and
  reconstruct the original key layout tensor-for-tensor;
- the original `step(frame)` remains as a compatibility wrapper.

The latest five-capability checkpoint was replayed through both paths over 64
held-out five-trial lifetimes. Actions, rewards, logits, hidden state, workspace,
and all memory outputs were exactly equal; maximum absolute logit difference was
`0.0`. All 66 checkpoint tensors reconstructed exactly. The wider controller
suite also passed 301 tests, and the new extraction suite passed eight tests
covering plain, deferred-action, action-to-intention, relation, and successor-
slot paths.

The extraction exposed one honest piece of migration debt. Older checkpoints
can add a learned two-action residual after the 24-dimensional intention. To
preserve behavior exactly, migration-v1 appends those two opaque coordinates to
`IntentEvent`, and only the external decoder interprets them. This is a
compatibility suffix, not an amodal capability claim.

That active debt has now been removed from the promoted successor. The learned
two-action residual was mapped through the frozen decoder's minimum-norm right
inverse and folded into a new 24-dimensional intention residual. This was a
closed-form checkpoint transformation: zero examples, zero verifier bits, zero
optimizer updates, and no semantic supervision.

At the 64-lifetime smell test the parent and candidate reports were exactly
identical, including the same small-sample four-rule miss. At 512 lifetimes both
passed all five gates with identical reports. Across four paired 512-lifetime
rollouts, all 12,288 actions were unchanged and maximum logit drift was
`5.72e-6`. The full 4,096-lifetime audit passed binary mapping, four-rule,
relation, persistent memory, and span-two working memory, including their
reversal, corruption, reset, and missing-evidence controls. The candidate's
compatibility suffix is structurally zero.

Promoted artifact:

- `artifacts/checkpoints/unified_repertoire_span2_amodal_intention_seed122005.pt`
- SHA-256 `9eea7ab479cb8450737f040b76495cc5ec737e970cdc165af2446873e530cd6c`
- evidence in `session_records/amodal_runtime_migration_2026-08-01/`

The next architecture rung is variable-cardinality visual input. It begins with
N=1 as an exact control and advances to redundant and complementary N=2
evidence.

## Independently learned output fan-out (2026-08-01)

`AmodalOutputBus` now accepts a runtime-variable mapping of backends. A new
two-command decoder learned a deliberately reversed opaque protocol while the
controller and inherited decoder stayed frozen. Its only learning signal was
whether its own attempted command succeeded.

Three span-two calibration seeds crossed the static threshold after 64
verifier bits and passed the complete five-capability closed loop at 512
lifetimes. The promoted decoder passed at 4,096 lifetimes. A matched
reward-shuffled learner failed; shuffled and zero intentions stayed at chance.
The simultaneous two-decoder audit preserved inherited logits bit-for-bit.

Promoted artifact:

- `artifacts/checkpoints/opaque_protocol_decoder_span2_seed133001.pt`
- SHA-256 `0258822d056a0bc5cf430a3035d81f84ede477eb0dbdd7fc9365d6be66bb03a7`
- evidence in `session_records/amodal_output_fanout_2026-08-01/`

Honest boundary: the historical recurrent API still consumes canonical action
IDs. Alternate physical protocols require a thin command-to-canonical-action
lowering when they drive that legacy loop. Asynchronous input is the next gate.

## Complementary N=2 input composition (2026-08-01)

`AmodalEventCollection` and `AmodalInputBus` now support runtime-variable event
cardinality without resizing the controller. N=1 and identical duplicates are
structurally bit-exact, including after the learned residual changes.

A 4,817-parameter permutation-invariant set residual learned to combine two
separately encoded partial views of a relation scene. It received only its own
attempted action and scalar outcome; the frozen controller and adapters received
no task, modality, identity, relation, or answer labels. Three seeds crossed
the stable 85% gate after 768–1,344 verifier bits.

At 4,096 held-out lifetimes the promoted bus scored 96.46% on bars versus
55.84% and 45.02% for either stream alone. Shuffling the partner returned to
51.77%; contradictory evidence caused 86.67% prediction flips. Without further
training it scored 90.96% on diamonds and 95.63% on dot pairs. Strict
cross-renderer transfer did not replicate across every seed, so only bars
composition is a replicated claim.

Promoted artifact:

- `artifacts/checkpoints/amodal_input_bus_complementary_seed145001.pt`
- SHA-256 `4ae96f60b99107834c27840b8841e8b2ba20c10e6565c220b5607fb9c80d3c71`
- evidence in `session_records/amodal_input_composition_2026-08-01/`

Timestamp-preserving transport alignment is now qualified: at 4,096 lifetimes,
out-of-order and bounded-jitter delivery reproduced synchronous actions exactly
at 96.36%, while mismatched timestamps remained separate windows. This is
transport plumbing, not a learned wait policy. Two outcome-only bus adaptation
pilots retained clean behavior but worsened a held-out pixel-erasure curve, so
the frozen bus remains the promoted artifact. The next boundary is replicated
cross-renderer transfer, corruption-aware noisy/missing streams, and learned
latency-versus-wait control; see
`session_records/amodal_input_noise_2026-08-01/README.md`.

Confidence routing closes one additional cardinality gate. With the promoted
frozen bus, two streams score 96.58% on 4,096 held-out lifetimes; an opaque
third stream scores 58.42% at confidence 1.0 but 96.40% at confidence 0.01.
This is an interface-level result: the frontend supplied generic confidence,
and a learned confidence estimator is still required before claiming fully
learned noisy/missing-stream handling. Evidence is in
`session_records/amodal_n3_confidence_2026-08-01/`.

That learned-confidence boundary is now partially closed for corruption. A
small head trained from clean/corrupted latent consistency, with full and
partial views in its training distribution, improves an 80%-erased third
stream by 5.24–6.61 points across two seeds while retaining N=2 at
98.77–99.04%. It is not yet a relevance estimator for a valid but irrelevant
stream; evidence is in
`session_records/amodal_learned_confidence_2026-08-01/`.

The next relevance boundary is now also qualified. A pair-agreement head,
trained only on same-frame complementary-view positives and independent-frame
negatives, routes a valid irrelevant third stream without changing the
controller or bus. Across two seeds, N=3 rises from 57.45% to 89.25–89.59%
while N=2 remains 98.51–98.55%. Cross-modality relevance and N>3 remain open;
see `session_records/amodal_pair_agreement_2026-08-01/`.

Using strongest-pair confidence with the fixed 0.8 threshold and a promoted
hidden-64/256-update self-supervised head extends the same router to N=11.
Across two independent audits, N=3 reaches 96.15–96.40%, N=4 95.76–96.13%,
N=5 94.96–95.52%, N=6 94.04–94.42%, N=7 92.77–93.17%, N=8 91.23–91.73%,
N=9 89.43–89.96%, N=10 87.45–88.10%, and N=11 85.34–86.14%; no-agreement
controls remain near chance, and N=2 remains 96.19–96.47%. N=12 reaches
83.19–83.88%, below the pre-registered 85% gate. See
`session_records/amodal_pair_agreement_cardinality_2026-08-01/`.

Convert a legacy checkpoint with:

```sh
python -m experiments.unified_cognitive_controller.convert_amodal_checkpoint \
  artifacts/checkpoints/unified_repertoire_span2_strict_seed122005.pt \
  /tmp/unified_repertoire_span2_extracted.pt
```

Canonicalize the inherited action residual into intention space with:

```sh
python -m \
  experiments.unified_cognitive_controller.canonicalize_intention_checkpoint \
  artifacts/checkpoints/unified_repertoire_span2_strict_seed122005.pt \
  /tmp/unified_repertoire_span2_amodal_intention.pt
```

This isolated experiment begins the transition from separately trained
primitive modules to one controller that learns within its recurrent state and
external memory.

## Boundary

The learner receives only:

- rendered RGB frames;
- its own previous opaque action;
- the scalar outcome of that attempted action;
- its own latent RAM/VRAM workspace;
- later, controller-created rows retrieved from disk memory.

It never receives stimulus identities, hidden rule bits, correct actions,
unattempted-action labels, semantic task IDs, or symbolic solution steps.

## First rung

The smallest rung chooses one private correct opaque action for each lifetime.
Trial zero is necessarily chance; after one attempted action and outcome, the
answer is fully identifiable. The controller must retain that evidence across
later query trials without a weight update or further learner-visible feedback.
Visible stimuli vary but are irrelevant at this rung, preventing the first
experiment from conflating active-state learning with visual rule binding.

The second rung chooses a private bijection between two visible stimuli and two
opaque actions. The first pilot learned a predictable second-trial shortcut but
stayed near chance on later randomized trials. It was correctly rejected and
motivated the smaller constant-action atom. The corrected generator samples
every query identity independently, eliminating that positional shortcut.

Before binding the two atoms, a visible-identity rung trains the same controller
to ground the two rendered identities in opaque actions using only attempted
actions and outcomes. Acceptance uses true pixel-level stimulus reversal,
required prediction flips, and blank-vision collapse. The discarded diagnostic
probe established that identity is already perfectly decodable from the parent
vision features; this rung tests whether behavioral reward can route it.

Naive sequential visual training erased the constant-action memory atom. A
combined parent is therefore admitted only when balanced rehearsal preserves
the old blind causal gate while the new pixel-grounding gate passes:

```bash
python -m experiments.unified_cognitive_controller.train \
  --device cuda \
  --task visible_identity \
  --rehearsal-task constant_action \
  --rehearsal-every 2 \
  --retention-task constant_action \
  --checkpoint-in artifacts/checkpoints/unified_constant_action_v1_seed2501.pt \
  --steps 600 \
  --batch-size 256 \
  --trials 6 \
  --feedback-trials 1 \
  --report experiments/unified_cognitive_controller/reports/combined_atoms.json \
  --checkpoint-out artifacts/checkpoints/unified_combined_atoms_v1.pt
```

The separate constant and visible tasks are ambiguous when mixed without a task
identifier: one demonstration can be consistent with both while they demand
different next actions. The unified composition rung therefore samples all
four binary functions—constant zero, constant one, identity, and flipped. Two
feedback-bearing support events cover both visible identities, making the
private function uniquely inferable. Later queries carry no feedback. This is
human-solvable without a task label and directly tests memory × perception.

The same **currently internal** vision encoder, recurrent controller, generic
workspace operations, latent intention, and actuator submodule process every
trial. The
long-term memory interface is serializable but deliberately inactive until
RAM-based few-shot behavior passes.

Acceptance requires:

- held-out post-feedback accuracy of at least 85%;
- the same under a true private-rule reversal with identical pixels;
- at least 80% paired prediction flips;
- at least a 15-point advantage over blank vision;
- near-chance zero-shot behavior;
- material degradation when workspace is reset;
- material degradation when outcome history is shuffled.

## Audited milestone

The gradual curriculum produced a repeatable forward-transfer result:

| 600-step four-rule initialization | Seed 2502 | Seed 2503 |
|---|---:|---:|
| mastered visual primitive | 99.90% | 99.85% |
| fresh controller | 75.05% | 74.85% |

The fresh controllers learned a feedback-only shortcut: constant rules can be
solved perfectly and identity-dependent rules guessed, giving a 75% ceiling.
Blank vision reproduced that ceiling. Primitive-initialized controllers learned
the full composition instead. On a 1,024-lifetime blind audit the selected
parent reached 99.85% normal and 99.90% reversed-rule query accuracy, 99.76%
paired query flips, 48.78% with shuffled outcomes, 55.66% with complete active
state resets, and 74.46% with blank vision.

A disposable supervised probe—not used by the agent—found visual identity
remained 100% decodable after compound training, versus 50.68% with shuffled
labels. This localized an apparent old-protocol regression to incompatible
behavioral prefixes rather than erased perception.

The next gradual rung restricts the function family to identity versus flipped,
so one attempted action and one scalar outcome uniquely identify the mapping.
Starting from the four-rule parent reached 100% query accuracy after 600 steps;
the matched fresh controller remained at 49.26%. The exact checkpoint repeated
at 100% on 1,024 blind lifetimes with 100% rule-counterfactual flips and chance
under shuffled feedback, active-state reset, or blank vision.

Training that one-support rung alone erased the broader four-rule behavior.
Balanced rehearsal fixed this without a task ID: bijection lifetimes retained
one support outcome while four-rule rehearsal lifetimes retained two. The
admitted checkpoint passed both 2,048-lifetime blind audits:

| Capability | Normal | Reversed | Paired flips |
|---|---:|---:|---:|
| one-support bijection | 99.98% | 99.95% | 99.93% |
| retained two-support four-rule | 100% | 100% | 100% |

The learner saw only RGB, its attempted opaque action, scalar outcome, and
latent active state. It received no task ID, rule bit, correct-action label,
unattempted-action label, or within-lifetime weight update. The promoted
checkpoint is
`artifacts/checkpoints/unified_compound_rehearsal_seed2505.pt`, SHA-256
`8e346fd2863f595fd0ee96b5f4a8353cae48e96e7739ad0febd11345a60f9099`.

This demonstrates fast within-lifetime binding, repeatable positive transfer,
and rehearsal-based behavioral retention inside one compact controller. It
does not yet demonstrate transfer to a new modality or broad reasoning.

## Cross-family same/different repertoire

The controller now also supports a simultaneous two-object relation rather
than only single-glyph mappings and context compositions. `pair_relation`
renders two identities at different positions and makes the correct opaque
action depend on same versus different. The relation changes independently on
every event, preventing reward history from substituting for visual evidence.

Three promoted runs reached 99.02–99.56% on held-out bars while retaining
binary mapping, visible context, and visible context XOR. Pixel-level relation
reversal, blank vision, missing-second-object, and retention audits passed.
Unseen contour transfer remains open: diamonds are ~26%, and disconnected dot
pairs are 68–71%. The next rung adds contour diversity gradually before
delayed matching.

The next bridge now holds the same relation across bars and diamonds. A
zero-output nonlinear correction to the slot gate preserves exact behavior at
insertion. Acquisition broadens the relation residual; consolidation freezes
that content and self-distils its verified diamond behavior while the refiner
learns to shut on unrelated skills. The promoted checkpoint reaches 99.61%
bars and 97.52% diamonds with all three earlier behavioral gates intact. A
matched reset relation slot reaches 97.69% diamonds but only 72.03% bars,
causally establishing reuse. Dot pairs remain at 70.61%.

## Zero-shot appearance transfer

The retained controller was audited without weight updates on two appearances
that never occurred in training:

| Unseen geometry | One-support mapping | Four-rule composition |
|---|---:|---:|
| elongated diamonds | 98.34% | not required |
| disconnected dot pairs | 94.95% | 98.14% |

The dot-pair renderer changes both contour and topology while preserving only
the tall-versus-wide relation. On the one-support arm, reversed rules reached
95.24%, paired predictions flipped 90.20%, and blank, shuffled-feedback, and
active-state-reset controls stayed at chance. The four-rule arm reached 98.43%
under reversal and 96.57% paired flips. No new verifier outcomes or optimizer
updates were used.

## Persistent-memory milestone

Recurring-context sessions give each visually identifiable world a private
binary mapping. A support session supplies ordinary attempted-action feedback.
The controller emits a context key before feedback and a latent value after
feedback. Its active state is then erased; a later session must retrieve the
row by content similarity and act through the same controller.

A frozen-interface probe localized the initial state:

- the post-feedback value carried the private rule at 100% diagnostic
  decodability;
- the pre-feedback key had a real but imperfect context signal;
- the post-feedback key collapsed because outcome processing changed it;
- feeding the correct value through the dormant read path improved behavior
  only slightly.

Training therefore paired the pre-feedback key with the post-feedback value.
No context ID, rule label, correct action, or retrieval target entered the
learner. Query loss used only the attempted action and its scalar outcome.
Two ordinary rehearsal streams preserved the earlier capabilities.

The gradual ladder produced:

| Trained memory capacity | Updates | Same-capacity recall | Harder capacity, zero-shot |
|---:|---:|---:|---:|
| 2 | 600 | 96.53% blind | 8: 87.48% |
| 8 | 150 | 93.65% | 16: 90.09% |
| 16 | 150 | 91.21% | 32: 87.48% |
| 40 | 40 total / 20 memory | 90.00% / 89.83% blind | 48: 88.28%; 56: 87.33%; 64: 85.57% / 86.33%; 72: 85.81% / 85.17% |

Every admitted checkpoint passed private-rule reversal, paired prediction
flips, empty memory, shuffled rows, corrupted latents, disk save/load
equivalence, one-support retention, and four-rule retention. The capacity-16
parent reached only 82.60% at capacity 48 because retrieval fell to 65.49%;
forcing the correct row restored 100% behavior, localizing the frontier to
keys rather than stored values or the reader. Soft reads at temperature 10
were badly mismatched to hard top-1 deployment. Raising the training
temperature to 50 produced a replicated capacity-40 gain after only 20 memory
updates and transferred zero-shot through capacity 56. Two independently
trained five-second runs also crossed the old capacity-64 frontier at 85.57%
and 86.33%, versus the prior parent's rejected 81.69% recall and 62.44%
retrieval. Both parents passed capacity 72 narrowly, then failed capacity 80
at 83.88% and 84.25%. A direct 40-update capacity-80 continuation regressed to
80.08%, so capacity 72 is the current frozen retrieval frontier.

The selected persistent-retrieval parent is
`artifacts/checkpoints/unified_persistent_capacity40_temp50_seed4801.pt`,
SHA-256
`3adc437e87e3ec65f02aeb22fe56bb16d0d48b43543d9858762eb2f27e2b3d9d`.

## Selective-memory milestone

The first admission atom adds a read-before-write cycle. An empty-slot
encounter and an already-occupied repeat are ordinary sensory episodes; the
controller receives no novelty, repetition, rule, or context label. The
memory-write gate is rewarded only by later verified query success minus a
generic write cost.

The first coupled pilot learned the easy always-write policy and was rejected.
A gradual occupied-memory curriculum separated the two credit-assignment
problems. After 80 updates, the controller discovered a more compact strategy
than the pre-registered "always write the first encounter" expectation:
memory absence represents the controller's default mapping, while a row is
written for exceptions. On a 4,096-context blind audit:

- first-encounter writes: 61.16%;
- redundant-repeat writes: 5.10%;
- total writes: 0.663 per context;
- query accuracy after either encounter: 99.90%;
- no-write accuracy: 49.93%;
- shuffled-admission accuracy: 79.71%;
- corrupted-value accuracy: 55.18%;
- hiding the prior read restored repeat writes to 61.16%;
- the two earlier behavioral capabilities remained at 100%.

The current checkpoint is
`artifacts/checkpoints/unified_selective_memory_atom_seed5402.pt`, SHA-256
`3fa82275e37ba5de686d4ec9966c1345e15b46e89938a8cf9bc0e0da94b15c30`.

## Adaptive disk-integration milestone

The learned selector was then connected to actual eight-context disk banks:
commit admitted rows, serialize files, erase active state, reload, retrieve,
revisit the contexts, commit any repeat writes, serialize again, and query.
The ungated reader failed because an intentionally absent default row still
retrieved the nearest unrelated row: tensor accuracy was 100%, first disk
reload accuracy fell to 56.84%, and duplicate growth rose from 5.10% to 47.07%.

A one-scalar no-match gate learned from verified outcomes raised held-out
tensor-bank accuracy to 88.94%. On two physical 1,024-context disk audits it
preserved 87.99–88.96% accuracy, but duplicate growth reached 20.02% and
21.39%, missing the pre-registered 20% gate. The failure localized a confidence
bug: ranking mixed cosine similarity with a write-strength prior. A
backward-compatible raw-cosine confidence mode fixed exact-repeat confidence,
but two fresh scalar pilots still exceeded the 25% absent-row false-accept
gate at 29.13% and 27.00%. Both were rejected and no checkpoint was saved.

A discarded capacity probe then tested four task-agnostic retrieval
statistics: cosine match, top-two ranked margin, selected-row strength, and
bank occupancy. The five-parameter linear probe reproduced the failure
(83.01% held-out classification, 33.26% absent false accepts). An eight-unit
nonlinear probe reached 88.18% and 18.84% respectively. Its private labels and
weights were discarded; only the evidence that the interface supported the
decision was retained.

A fresh 49-parameter nonlinear gate was added inside the existing controller
and trained exclusively from verified query success minus a generic read
cost. No match, presence, context, rule, or action labels entered this learner.
The admitted run used 160 optimizer updates, 81,920 unique logical contexts,
no replay, and 9.71 seconds of GPU wall time. Its held-out tensor-bank audit
reached:

- 91.55% gated accuracy versus 54.74% ungated and 50.02% empty;
- 89.67% acceptance when the context had stored a row;
- 17.33% false acceptance when its row was absent;
- unchanged one-support and four-rule retention gates.

The gate itself added 0.139 ms for a batch of 4,096 decisions on the RTX PRO
6000 (1,000 timed iterations after warm-up); this is batched throughput, not a
claim about end-to-end serial action latency.

The exact checkpoint then passed two independent physical 1,024-context disk
audits:

| Audit seed | First reload | Repeat reload | Duplicate rows/context | Empty | Wrong-value corruption |
|---:|---:|---:|---:|---:|---:|
| 6001 | 91.50% | 91.02% | 17.68% | 50.20% | 70.41% |
| 6002 | 92.19% | 91.41% | 17.29% | 50.00% | 70.70% |

Each audit created eight-context banks, serialized them, erased active state,
reloaded them, retrieved through the learned gate, revisited the contexts,
committed any new writes, serialized again, and queried. The corruption arm
preserved keys and admission statistics but rotated values between contexts,
showing that correct stored content—not merely a context match or gate
activation—caused the gain.

The promoted checkpoint is
`artifacts/checkpoints/unified_selective_disk_adaptive_seed5962.pt`, SHA-256
`91822064436fae1d4f799e41c79d9369dacb8aeeee20b711df1c1b6af037fbc4`.

## Bounded replacement milestone

The next atom deliberately filled a four-row bank, presented one incoming
controller-created row, and required a choice among skip or four physical
eviction slots. Physical slot order was randomized. The future sensory query
stream revisited the candidate and the three still-useful existing contexts;
the oldest logical row was no longer useful. The policy saw only normalized
row age, write strength, candidate-key similarity, candidate strength, and a
skip flag. It received only later verified correctness minus a generic
replacement cost—never context IDs, utility labels, correct eviction targets,
or semantic memory contents.

A fresh 57-parameter head inside the controller passed after 40 updates:

- 96.90% held-out accuracy versus 97.73% oracle;
- 93.55% correct eviction selection;
- 84.35% random, 85.11% fixed-slot, and 80.91% skip controls;
- 81.79% when age features were shuffled across physical slots;
- 51,200 generated support contexts, 20,480 future query outcomes,
  71,680 total verifier bits, no replay, and 7.01 seconds total wall time;
- both older behavioral gates remained intact.

The replacement gate scored 1,024 banks with five options each in 0.141 ms
batched on the RTX PRO 6000 (1,000 iterations after warm-up). This is
throughput, not end-to-end serial action latency.

The exact checkpoint then passed two physical disk audits:

| Seed | Normal disk | Age-corrupted | Correct eviction | Rows before/after | Capacity growth |
|---:|---:|---:|---:|---:|---:|
| 6201 | 96.97% | 81.35% | 92.97% | 2,048 / 2,048 | 0 |
| 6202 | 96.29% | 82.37% | 91.02% | 2,048 / 2,048 | 0 |

Each audit populated real bounded files, applied the learned choice, serialized
and reloaded every bank, and queried through the inherited adaptive read gate.
A fresh capacity-8 sparse-memory retention audit also passed at 91.21% first
reload, 91.02% repeat reload, and 15.23% duplicate growth.

The first capacity-4 policy was additionally applied zero-shot to capacity 8.
It reached 93.99%, but correct eviction fell to 71.48% and age corruption
reduced accuracy by only 3.91 points. That arm failed its causal gates. The
curriculum therefore returned to the next single-step rung.

Capacity 5 passed twice without training at 95.63% and 95.51%. Capacity 6 was
a genuine near miss: 94.99% and 84.96% correct evictions, but insufficient
causal separation. The bridge used 20 capacity-6 updates alternating with 20
capacity-5 rehearsal updates. It consumed 94,720 verifier bits, no replay, and
8.69 seconds total wall time. The resulting policy reached:

- capacity 6: 96.39%, exactly matching its oracle, with 100% correct evictions;
- rehearsed capacity 5: every admission gate passed;
- unchanged one-support and four-rule behavioral gates.

Two physical capacity-6 audits reached 96.55% and 96.71%, with 100% correct
evictions, zero capacity growth, and age-corrupted accuracies of 86.04% and
86.78%. Capacity-5 retention reached 96.37%; the inherited sparse disk loop
remained at 90.33% first reload, 90.14% repeat reload, and 17.38% duplicate
growth.

The sharpened capacity-6 policy then generalized without weight updates:

| Capacity | Physical accuracy | Replication | Correct evictions | Growth |
|---:|---:|---:|---:|---:|
| 7 | 95.70% | 96.23% | 100% / 100% | 0 |
| 8 | 94.85% | 95.75% | 100% / 100% | 0 |
| 9 | 94.57% | 94.62% | 99.61% / 100% | 0 |

For capacity-aware causal auditing, age corruption must change correct
eviction selection by at least 50 points and erase at least 40% of one query
slot (`0.4 / capacity`). This prevents an impossible fixed absolute gap as
each bank grows while retaining a direct causal-action requirement. At
capacity 9, corrupting age-to-slot correspondence drove correct evictions to
0% on both seeds and accuracy to 89.13% and 88.80%.

The original replacement checkpoint remains
`artifacts/checkpoints/unified_memory_replacement_seed6101.pt`, SHA-256
`0178b15228e3d75a445abdb2376be1291a078f8b47236444fbd1824fab3d3b76`.
The fixed-utility parent is
`artifacts/checkpoints/unified_memory_frequency_recency_capacity6_seed6607.pt`,
SHA-256
`1346da994de4ba20864c5f1bc1da12684fc13d8dcda480a76cfc6f713da0181c`.

### Noisy frequency-plus-recency utility

The next rung held capacity at six and changed the utility distribution rather
than scaling the same oldest-first task. Each row gained a persistent
`access_count`. Ordinary content-addressed retrievals can increment it; the
count survives save/reload, copies into active memory, grows with storage, and
resets when a row is replaced. Old memory files load with zero counts.

The learner saw centered log access frequency beside the five existing generic
replacement features. It never saw realized future utility, future query
identity, or the correct eviction. Training still used only future verified
success minus the generic replacement cost.

The first useful failure localization was optimization-level:

- the inherited recency head assigned about 91% probability to one action;
- merely widening its first layer changed a frequency weight but did not change
  decisions;
- softened exploration exposed alternatives, but the cold exponential reward
  baseline reinforced below-average samples during tiny runs;
- batch-centered verified advantage corrected the sign;
- a direct zero-initialized residual avoided routing the new statistic through
  the saturated inherited MLP;
- centering the statistic prevented all real rows from shifting against the
  non-row skip option.

The final adapter adds one trainable parameter. The controller has 298,359
parameters total. Two independent reward-only runs used 20 updates each:

| Seed | Training time | Unique verifier bits | Held-out | Correct eviction | Age shuffle | Frequency shuffle |
|---:|---:|---:|---:|---:|---:|---:|
| 6607 | 3.23 s | 51,200 | 95.32% | 87.30% | 89.57% | 89.96% |
| 6608 | 3.23 s | 51,200 | 95.10% | 86.13% | 89.18% | 89.67% |

Both passed the recency-retention gate and the inherited binary and four-rule
behavioral audits. Only `memory_replacement_extra_gate.weight` changed. There
was no replay.

Physical audits then generated history through ordinary content-addressed
retrievals, serialized and reloaded every bank before making the replacement
decision, replaced at most one row, serialized and reloaded again, and issued
the future queries:

| Train / audit seed | Learned | Visible oracle | Strongest single | Correct eviction | Age shuffled | Frequency shuffled | Rows | Growth |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6607 / 6701 | 96.81% | 97.40% | 94.92% | 92.97% | 92.25% | 90.04% | 1,536 / 1,536 | 0 |
| 6608 / 6702 | 96.29% | 96.48% | 94.14% | 93.36% | 91.54% | 88.54% | 1,536 / 1,536 | 0 |

The learned policies captured 76.3% and 91.7% of the available composition gap
above the strongest single-feature control. All 512 realized access histories
survived disk save/reload exactly. Only 208 matched the requested counters
exactly because ordinary content addressing sometimes routed a repeated query
to a competing row. The verifier recomputed utility and future demand from the
history that the physical memory actually experienced; it did not overwrite
the counters to make the test easier.

Claim boundary for the fixed-mixture rung: this demonstrates
controller-created, content-addressed latent storage and recall across
active-state resets, with actual disk serialization.
It also demonstrates learned write/skip and read/no-read decisions operating
together within the storage-efficiency gate, plus learned bounded replacement
that composes recency and retrieval frequency under noisy utility.

### Online utility adaptation

The next atom made the utility distribution piecewise stationary while keeping
capacity at six. One continuous stream used four phases:

1. 65% recency / 35% frequency;
2. 35% recency / 65% frequency;
3. return to 65% recency / 35% frequency;
4. return to the inherited 50% / 50% mixture.

The learner received no phase identity or boundary signal, and its optimizer
was never reset. It saw only sensory-derived generic memory features, attempted
replacement actions, and later verified future success. It did not see the
utility weights, correct eviction, future-query identity, or semantic labels.

Several sub-minute failures localized the learning mechanism before the
successful run:

- a high-temperature policy gradient optimized a stochastic objective whose
  local gradient pointed opposite to greedy held-out behavior;
- a paired greedy baseline and a pairwise preference update still reinforced
  the currently selected action instead of exploring the useful coefficient;
- exact coefficient sweeps showed the benchmark was learnable and identified
  different optima for the two utility phases.

The successful mechanism is a generic symmetric perturbation horse race. On
each fresh bank, two temporary copies of the same coefficient are evaluated at
`w + 3` and `w - 3` using actual greedy actions and future verified outcomes.
The surviving controller moves 1.5 units toward the better copy. Only
`memory_replacement_extra_gate.weight` changes; there is still one controller,
not a task-specific expert or phase-conditioned mode.

Two independent runs replicated:

| Seed | Seconds | Recency target | Frequency target | Recency return | Equal return |
|---:|---:|---:|---:|---:|---:|
| 6809 | 28.66 | 90.67% | 86.43% | 91.16% | 90.53% |
| 6810 | 28.89 | 90.82% | 87.16% | 91.31% | 89.99% |

Each used 64 updates, 114,688 generated training contexts, 212,992 unique
verifier bits counting both candidates, and zero replay. Every phase stayed
within three accuracy points of its visible oracle. Each shifted phase beat
the frozen parent and added at least four points of correct-target eviction.
The coefficient moved toward recency, then frequency, then recency again.
Binary mapping and four-rule retention gates passed, and only the one residual
coefficient changed.

The reward-alignment control used the identical algorithm and budget but
randomly swapped which candidate received each paired verified outcome. It
failed as required: its frequency-dominant target rate was 57.71%, below its
frozen parent's 69.53%, and it moved the coefficient in the wrong direction.
No checkpoint was saved.

The selected seed-6810 checkpoint then passed a larger physical disk audit:

- learned accuracy 96.94%, visible oracle 96.81%, full oracle 97.41%;
- age-shuffled 92.74%, frequency-shuffled 88.66%;
- 6,144 physical rows before and after, zero capacity growth;
- all 1,024 access histories survived save/reload exactly;
- no weights changed during the audit.

Selected checkpoint:
`artifacts/checkpoints/unified_memory_online_utility_seed6810.pt`, SHA-256
`c3e837c6512a30c11b1c861b79242296b76cfa0cd9fe62aa414d3e5b2aa10750`.
Independent replica:
`artifacts/checkpoints/unified_memory_online_utility_seed6809.pt`, SHA-256
`d25d26c4d34ff86e50474b5ff38c630a2d92b782dea10d4782b01a363bb64a81`.

Honest boundary: the adaptation stream used tensorized banks, while the final
adapted policy was audited against actual serialized disk memory. This proves
fast label-free adaptation of one generic utility coefficient, not online
learning directly inside an unbounded physical-disk stream and not a learned
general meta-optimizer. Consolidation, deletion/merging, higher-dimensional
utility adaptation, modality transfer, and broad reasoning remain open.

### Outcome reliability and two-dimensional online utility

The next gradual feature was chosen by a representation gate before training.
A proposed extra write-strength coefficient was rejected because write
strength already existed in the inherited five-feature replacement path; at
the first learnable mixture it added only 2.93 target points. The next feature
was row reliability: the fraction of verified successful versus failed uses,
with a one-success/one-failure prior. This statistic is task-agnostic and
derived only from attempted use plus scalar verifier outcomes.

Physical memory schema v3 adds persistent success and failure counts. Outcomes
are attributed to whichever row ordinary content addressing selected. Counts
copy into active memory, survive save/reload, grow with storage, and reset on
replacement. V1 and V2 files load with zero outcome counts.

The controller grew from 298,359 to 298,360 parameters. Its existing frequency
coefficient was copied exactly and one reliability coefficient was
zero-initialized. A generic Rademacher horse race compared `w + d`, `w`, and
`w - d` on the same fresh banks. Including the center candidate was necessary:
the earlier forced plus/minus race drifted even when the current policy was
already best.

The uninterrupted phases were old equal recency/frequency utility, 40%
reliability-dominant utility, old return, and all three features equal. No
phase signal, labels, replay, or optimizer reset were exposed.

| Seed | Seconds | Verifier bits | Old equal | Reliability dominant | Old return | All equal |
|---:|---:|---:|---:|---:|---:|---:|
| 6932 | 29.37 | 196,608 | 89.75% | 78.22% | 88.48% | 87.45% |
| 6938 | 29.34 | 196,608 | 88.67% | 88.43% | 84.72% | 83.35% |

The frozen parent's reliability-dominant target rates were 57.62% and 58.40%;
all-equal rates were 63.48% and 64.60%. Ablating only the learned reliability
coefficient reduced seed 6932 to 55.27% and 63.67%. Both inherited behavioral
gates passed, and only `memory_replacement_extra_gate.weight` changed.

The exact reward-shuffled control failed the registered phase gate. It damaged
old-equal performance to 79.15%, returned to the old mixture at only 75.93%,
and finished all-equal at 64.31%. No checkpoint was saved.

The selected seed-6932 controller passed a 1,024-bank physical disk audit:

- learned 96.21%, visible oracle 96.35%, full oracle 97.18%;
- 6,144 rows and 1,024 complete histories survived save/reload; zero growth;
- learned correct eviction 89.65%;
- age shuffle: 39.36% correct, 93.10% behavioral accuracy;
- frequency shuffle: 29.10% correct, 89.45% behavioral accuracy;
- reliability shuffle: 59.47% correct, 93.65% behavioral accuracy.

The first fixed four-point corruption gate was explicitly rejected after two
audits showed it was mis-scaled for three features and six future query slots.
A fresh seed used a capacity-aware registered gate: each corruption had to
change at least 20% of correct evictions and erase at least 15% of one query
slot (`0.15 / capacity`) from behavior. All three passed.

Selected checkpoint:
`artifacts/checkpoints/unified_memory_multifeature_reliability_seed6932.pt`,
SHA-256
`bb5cd158c08f4b92061aca7bfae0751d4e18408e8e37f53cac13dffaed8ac9f4`.
Independent replica:
`artifacts/checkpoints/unified_memory_multifeature_reliability_seed6938.pt`,
SHA-256
`0342a8266bde7bc5a0f79004792ce29668f758904aa954755b7bf7130993730d`.

Honest boundary: training still uses fast tensorized histories, followed by a
physical audit of the adapted controller. The project has not yet demonstrated
the adaptation loop itself running over an evolving physical disk stream.
Reliability is generic verifier history, but the controller does not yet learn
which new statistics to invent.

### Physical online adaptation

The next experiment made physical disk memory sovereign during the horse race.
Every candidate was evaluated by copying, saving, reloading, and querying
bounded physical memory banks. Tensor rewards were retained only as a shadow
parity audit and never selected the update.

A 32-bank parity preflight passed exactly in 1.34 seconds. Two 32-bank training
pilots were rejected: their candidate estimates were too noisy to move the
coefficients reliably. Scaling only that evidence-backed bottleneck to 128
banks reproduced the tensor experiment:

| Seed | Seconds | Old equal | Reliability dominant | Old return | All equal |
|---:|---:|---:|---:|---:|---:|
| 7012 | 136.33 | 91.06% | 82.13% | 88.23% | 82.91% |
| 7015 | 136.69 | 85.74% | 77.25% | 86.72% | 82.67% |

Each run used 48 updates, 196,608 unique verifier bits, zero replay, 6,144
persisted physical histories, and 18,432 bank-candidate evaluations. All old
behavioral gates passed and only the two-dimensional replacement residual
changed. Physical and tensor choices were equivalent within `1e-6` on all 48
updates; the largest raw reward difference was `5.96e-8`. The tolerance is
tie-aware because an earlier replica exposed an exact physical tie separated
by one floating-point unit in the tensor shadow.

The matched reward-shuffled control learned the reliability coefficient in the
wrong direction, failed all four adaptation gates, and saved no checkpoint.
This rules out mere parameter motion, disk traffic, or the phase schedule as
the explanation.

Selected checkpoint:
`artifacts/checkpoints/unified_memory_physical_online_seed7012.pt`, SHA-256
`2c6e61b5e2689d46dfc43dd5cfc9c5b234736d217aae28f6221501bd5ddeea70`.
Independent replica:
`artifacts/checkpoints/unified_memory_physical_online_seed7015.pt`, SHA-256
`7ae96b44ec6bed0db8eb7f9b78640fe40b621875195303e3e3c604f357bb441d`.

Honest boundary: each update still starts from newly generated bounded banks.
The next atom is a small set of banks that persists and accumulates reads,
outcomes, and replacements across multiple updates and utility switches.

### Persistent physical stream

That next atom now passes. The same eight bounded physical banks were kept
alive for six sequential decisions spanning old-equal, reliability-dominant,
and old-return utility phases. Each winning decision mutated the existing
bank; ordinary content-addressed reads and binary verifier outcomes updated its
history; and the bank was saved and reloaded before the next decision.

Two independent CPU replicas completed in 2.14 and 1.92 seconds. In both:

- all 144 candidate bank copies and all 56 stream-state save/reloads per
  replica remained exactly bounded at six rows;
- every access/outcome transition matched the previous persisted totals,
  minus deliberately replaced-row history, plus the six new verified queries;
- physical rewards matched the tensor shadow within `1e-6`, and all candidate
  choices were tensor-equivalent within `1e-6`;
- replacements, access counts, success counts, and failure counts survived
  serialization; and
- a causal control that rolled the physical access/outcome histories across
  rows changed at least one replacement decision in each replica.

Reports:
`reports/persistent_physical_stream_seed7022_banks8.json` and
`reports/persistent_physical_stream_seed7023_banks8.json`.

This is a persistence and causal-plumbing result, not yet a learning-speed
result. The honest frontier is to adapt the utility residual from rewards while
these same banks remain alive, then compare switch recovery and verifier bits
to the fresh-bank baseline. Only after that passes should bank lifetimes or
learned statistics be enlarged.

### Reward adaptation over persistent physical banks

The next atom passes in a 24.3-second CPU run. Sixteen bounded physical banks
remained alive for nine decisions across old-equal, reliability-dominant, and
old-return utility phases. At every decision, three symmetric utility-residual
candidates competed on verified future outcomes from physical save/reload
episodes. Only the winning residual was retained, and its action mutated the
same persistent banks.

During the reliability switch, the adapted controller selected the verifier's
target row on 50.0% of banks versus 14.6% for the frozen controller on the same
states. Mean verified future reward rose from 91.67% to 94.79%. When the old
utility returned, adapted reward remained within 0.35 points of the frozen
control. Binary-mapping and four-rule retention gates passed, only the
two-weight utility residual changed, all 736 physical save/reloads remained
bounded and exact, and the tensor implementation remained a parity-only shadow
within `1e-6`.

The causal reward-shuffle control failed as intended. Reliability target
selection fell to 6.25% and verified reward to 88.89%, below its frozen
control. Therefore persistence or random perturbation alone does not explain
the useful update; correctly aligned physical verifier reward is necessary.

Reports:
`reports/persistent_physical_adaptation_seed7032_banks16.json` and
`reports/persistent_physical_adaptation_shuffled_seed7033_banks16.json`.

This demonstrates fast reward-driven utility adaptation while state truly
persists. It does not yet demonstrate a compounding learning-speed advantage
over a fresh-bank learner at matched verifier bits. The next experiment should
run a matched persistent-versus-fresh race from the same initial residual and
report area under the verified-reward curve per verifier bit, switch-recovery
bits, retention, and wall time.

### Matched persistent-versus-fresh efficiency

That comparison now passes on two paired seeds. Each arm began from identical
controller weights, used identical perturbations and task seeds, made nine
updates, and consumed 2,592 candidate verifier bits. The comparison normalized
each learned policy against a frozen policy evaluated on the same physical
states, avoiding the misleading fact that newly generated banks are easier in
absolute terms.

Across seeds 7032 and 7034:

- persistent normalized verified-reward AUC averaged `0.10417`;
- fresh-bank normalized verified-reward AUC averaged `0.03125`;
- persistence therefore produced 3.33 times the reward gain at the same
  candidate-verifier budget;
- persistent target-selection advantage AUC averaged `+1.03125`;
- fresh target-selection advantage AUC averaged `-0.28125`; and
- persistence beat fresh memory on both reward and target-selection advantage
  in both independent replicas.

Binary and four-rule retention passed in every arm. The previously recorded
reward-shuffle control remained rejected. Report:
`reports/persistent_vs_fresh_efficiency_seeds7032_7034.json`.

This is the first evidence that retaining physical experience compounds
short-horizon utility-learning efficiency. Its boundary is important: both
arms learned the same two-feature utility family. It does not yet show that one
learned cognitive primitive accelerates acquisition of a different novel
primitive. The next rung is a gradual two-task transfer atom: learn one latent
utility relation, switch to a related but held-out relation, and compare
learning curves with intact, empty, shuffled, and fresh long-term memory.

### Gradual transfer and the global-residual limit

The causal memory half of that rung passes. On two paired seeds, an intact
physical-history view beat a shuffled view on both normalized verified-reward
AUC and target-selection AUC. Intact memory crossed a ten-point
target-advantage threshold in the first 288 verifier bits in both replicas;
the shuffled-history arm never crossed it. Emptying only the controller-visible
history also removed the advantage in the supporting seed. All old-skill
retention gates passed.

The learned-weight reuse half did not establish robust compounding. Resetting
the two-weight utility residual before the held-out phase hurt one seed and tied
the other. Two deliberately gradual curricula then localized why:

- reliability `0.2 -> 0.4` produced identical warm and cold target curves
  because the source was too easy to teach the residual anything reusable;
- reliability `0.3 -> 0.4` taught the source relation, but the warm residual
  transferred negatively and underperformed a cold reset on the target.

Therefore accumulated physical experience is causally useful, while a single
global residual is an interference bottleneck. More rounds on that mechanism
are low ROI. The next architecture atom should keep utility strategies as
context-indexed fast state in RAM and consolidated long-term memory, with the
same controller retrieving and composing them. Compare it against the global
residual at equal verifier bits, and require positive transfer plus retention.

Canonical audit:
`reports/gradual_transfer_audit_seeds7032_7034.json`.

### Four-slot latent strategy RAM

The next architecture atom is implemented and mechanically passes. A bounded
four-slot RAM bank stores ten-dimensional physical-context summaries plus a
three-dimensional recent verifier-response signature as latent keys, and the
two generic utility coefficients as values. Retrieval is content-addressed;
verified candidate improvement updates success/failure statistics; low-utility
records are replaced at capacity; and every update survives exact
serialization. No task name, utility weight, semantic label, or verifier target
enters a key.

The first retrieval protocol failed diagnostically: physical-only keys had
cosine similarities near `0.99` across all phases and merely reproduced the
global residual. Adding recent centered candidate outcomes made contexts
distinguishable. A post-probe protocol then allowed the retrieved strategy to
compete as a fourth candidate, with its additional verifier bits charged.

That produced one genuine positive pilot on seed 7042: target normalized
reward-AUC improved from `-0.01042` for the global residual to `+0.02083` for
strategy RAM, even after accounting for 3,360 versus 2,592 candidate verifier
bits. However, seed 7043 tied in raw reward and was less efficient after the
extra probe. Shuffling strategy keys did not reduce target reward. Therefore
the capability gate is correctly rejected.

This is a useful bounded negative:

- strategy storage, retrieval, replacement, and persistence work;
- a retrieved strategy can rescue a failing trajectory;
- the present context representation does not address strategies causally or
  reliably enough for replicated sample-efficiency gains; and
- increasing capacity, rounds, or model size is not justified.

Canonical audit:
`reports/strategy_memory_audit_seeds7042_7043.json`.

The frontier is now a learned context encoder trained only through verified
strategy usefulness. It should predict which stored strategy will improve
future reward, using physical history and recent outcomes, while shuffled-key
and cold-memory controls remain mandatory. The encoder must first beat the
current fixed key in sub-minute probes before any memory bank is enlarged.

### Compute allocation and the weight-transfer boundary

A passive inverse-propensity advantage learner made the external-memory read
decision useful from 720 attempted-action verifier bits and replicated with a
105-parameter head. It captured roughly 60% of the oracle compute-allocation
gap; reward, feature, missing-evidence, and evidence-shuffle controls failed.
The inherited production read gate remained more accurate, so its weights were
preserved.

The same blueprint was then tested on a distinct optional operation: answer now
or take one more recurrent thought step. Generic controller statistics
contained real decision signal, but neither direct whole-head transfer nor
hidden-trunk-only transfer beat a matched reset learner under the
pre-registered 720-bit gate. The strongest near-miss reached 64.1% choice
accuracy, but near-misses are not promoted. Private balancing also required
screening 86,016 lifetimes per run.

This closes immediate operation-specific weight transfer. The next compounding
atom is a shared compute-value representation learned across multiple optional
operations with small operation-specific outputs, evaluated by stable
bits-to-threshold on a third held-out operation. The verifier remains
sovereign; only attempted actions and scalar outcomes are learner-visible.

The first gradual bridge now passes and replicates. Moving the learned
read-advantage allocator from memory capacity three to capacity four required
zero new bits for stable mastery on two fresh powered seeds; matched reset
learners required 120 verified outcomes. Final inherited choice accuracy was
72.1% and 71.1%, with 54.0% and 50.7% oracle-gap capture. Reward, feature, and
missing-evidence controls removed the benefit, and old skills were retained.

This is narrow environmental transfer, not yet cross-operation abstraction.
It establishes the correct curriculum scale: alter one compute-context axis at
a time, demand replicated bits-to-threshold improvement, then gradually bridge
toward a new physical operation.

Sequential consolidation now strengthens this result. Reusing the original
capacity-three allocator directly at capacity five was slower than reset
(`240` versus `120` stable bits). After learning capacity four and saving that
updated 105-parameter allocator, capacity-five mastery required zero new bits
on two target streams. The older capacity-three lineage required 120 and 240
bits; reset required 120 on one stream and never crossed within 720 on the
other. All causal and retention gates passed.

This is the current compounding frontier: intermediate verified experience
must actually update and persist in the lineage. The next controlled distance
increase should hold capacity fixed and shift one generic evidence
distribution, then compare the consolidated lineage against both its earlier
ancestor and reset.

That next axis now also passes. At capacity five, a private zero-training
frontier search selected read cost 0.24: high enough to make indiscriminate
extra computation harmful. On two fresh target streams, the
capacity-five-consolidated lineage was already stably mastered at zero new
bits. Its capacity-four ancestor required 120 bits on both streams; reset
required 360 and 120. Final inherited accuracy was 81.1% and 81.7%, with 58.5%
and 59.3% oracle-gap capture. Evidence interventions and retention audits
passed.

The updated cost-sensitive checkpoint is persisted for the next rung. The
frontier is now a gradual physical-operation bridge—external read to re-query
before recurrent thought—while keeping stable bits-to-threshold, ancestor,
reset, evidence, and retention controls unchanged.

The first second-ranked re-query bridge localized the operation boundary.
Direct whole-head transfer inverted the decision and performed catastrophically.
Resetting only the one-neuron operation output allowed the inherited trunk to
master re-query in 120 bits, while a fully reset model did not master within
720. However, the cost-consolidated trunk tied its capacity-five ancestor at
120 bits, so no additional compounding gain was established.

The new frontier is a shared compute-value trunk with operation-specific
outputs and explicit normalized compute cost as a generic input. This preserves
reusable evidence processing while preventing action-sign semantics and cost
policy from being entangled in the same output neuron.

The first explicit-cost version did not pass. Variable-cost read experience
created measurable but weak cost sensitivity, while cost-aware,
cost-shuffled, ancestor, and reset target learners all required 600 re-query
bits—substantially worse than the existing 120-bit inherited-trunk result.
More source data is not justified without a stronger mechanistic signal.

The frontier therefore returns to the successful gradual-curriculum pattern:
persist the learned re-query trunk, train it through nearby re-query
difficulty shifts, and only then test transfer toward recurrent thought.

That frontier is now resolved more strongly. A 64-hidden-unit action-value
challenger learns `Q(ordinary read)` and `Q(re-query)` directly from the scalar
outcome of the operation actually attempted. Active disagreement allocation
spends verifier bits where incumbent and challenger differ. The incumbent
remains immutable; a frozen proposal requires a positive lower-95% bound on
2,400 disjoint shadow outcomes before deployment.

Across nine consecutive streams, the gap learner was safely confirmed and
promoted while the mastered incumbent never changed. The final two seeds also
passed durable integration: opaque parent and action-value child skills were
atomically committed to a content-addressed disk store, reloaded bit-exactly,
retained audited utility, detected child corruption, and preserved parent
retrieval plus earlier controller capabilities.

The learning milestone is therefore no longer “can a candidate improve without
forgetting?” It can. The next compounding frontier is sequential retrieval:
load the confirmed child for a related new primitive and measure whether it
reduces bits-to-confirmation versus the parent and reset while every ancestor
remains intact.

That sequential-retrieval frontier has now advanced through three verified
generations. The two-action champion was wrapped as an option to learn a third
action, that hierarchy was wrapped to learn a fourth, and the complete
four-action hierarchy was then wrapped to learn a fifth. The fifth-action
relation was highly decodable but ordinary single-action reward regression was
unstable. Paired population exploration solved the credit problem without
hand labels: temporary clones attempted the old hierarchy and new action on
the same context, and the router learned their verified scalar advantage.

On two frozen replications, the composed hierarchy reached a stable +2-point
utility gain in 3,360 and 3,600 verifier bits; flat five-action reset required
15,600 and 5,400 bits. Transfer ratios were therefore 4.64× and 1.50×. Both
routers passed eight-stream held-out improvement, feature-shuffle, reversal,
independent randomized confirmation, old-skill retention, exact reload,
corruption detection, and parent-lineage survival. The canonical record is
`session_records/fifth_option_composition_2026-07-26/README.md`.

The new frontier is automatic population allocation: predict when the expected
value of counterfactual clone experience justifies its extra verifier cost,
while periodically revisiting apparently unpromising branches so
phase-transition skills are not starved.

The same mechanism has now compounded once more. The complete verified
five-action hierarchy was retrieved as one option and extended with a sixth
action. Two frozen replications reached stable mastery in 5,280 and 5,760
verifier bits, versus 10,800 and 6,480 for flat six-action reset. Both passed
independent eight-stream, randomized-confirmation, shuffle, reversal,
retention, persistence, corruption, and lineage audits.

The six-action experiment revealed a new resource boundary: 512 optimizer
updates over 7,680 unique contexts found the skill transiently but did not
retain it; 2,048 updates over the exact same experience produced stable
mastery. The frontier is therefore adaptive experience processing—learn how
much replay or internal thought each experience merits, charging processing
alongside verifier bits and latency.

That adaptive-processing frontier now passes. A generic controller monitors
loss over its own observed replay memory and treats 16 updates as a maximum,
stopping once loss reaches 0.14. On two prospective matched seeds it preserved
the fixed learner’s exact stable mastery thresholds (5,760 and 6,000 verifier
bits) while reducing optimizer updates by 39.6% and 47.3%. Wall time fell by
roughly 41% and 52%.

Both adaptive checkpoints retained the six-action gain under the full
eight-stream, randomized-confirmation, shuffle, reversal, old-skill retention,
persistence, corruption, and lineage audit ladder. The next frontier is to
learn the stopping criterion across task generations rather than fixing one
generic loss target.

That learned-stopping fork has now been localized. Generic replay state from
four five-action streams predicted eight-update loss reduction on held-out
six-action streams with `0.91–0.92` correlation and `23–24%` lower MAE than a
mean predictor; shuffled targets stayed near zero. The corresponding behavioral
policy did not replicate inside its frozen gate. An ensemble mean saved
`12.9–18.0%` of updates and preserved capability, but one seed required
`11.11%` more verifier bits than fixed replay, just outside the `10%` limit.
No stopper checkpoint was promoted.

The frontier is now objective alignment rather than replay-state
representation: predict long-horizon verified behavioral learning progress and
retention per unit compute, rather than local replay-loss reduction. See
`session_records/learned_replay_stopper_2026-07-27/README.md`.

The behavioral-target experiment sharpened that conclusion. Rich learner-state
features predicted held-out future behavioral improvement with `0.31`
correlation (`0.34–0.35` with a longer target), and correctly ranked a harmful
bottom quartile on fixed trajectories. Prospective pruning nevertheless failed:
omitting a block changes all later learner states, so one-step counterfactual
rank is not long-run causal value.

Matched full-trajectory budgets then produced the next sample-efficiency
breakthrough. Eight replay updates lost to sixteen on all eight streams.
Twenty-four updates beat sixteen on all eight streams, improved final utility
by `+0.0034` to `+0.0294`, reduced verifier bits to stable mastery by `19–44%`
whenever both solved, and rescued one stream that sixteen never mastered. More
internal compute can therefore buy substantially less external experience.
The completed causal ladder established a 48-update sweet spot for this task
family: it beat 40 updates on verifier efficiency in 9/12 fresh streams and
on final utility in 9/12. At 56 updates the experience cost still fell in 9/12
streams, but final utility regressed in 8/12: more thought became overthinking.
A 48-update checkpoint passed independent confirmation, causal shuffle and
reversal, old-skill retention, persistence, and corruption audits. The current
frontier is a causal budget allocator around this verified operating region;
it must train on whole-trajectory outcomes, not local replay-loss targets. See
`session_records/causal_replay_budget_2026-07-27/README.md`.

Run the sub-minute GPU experiment:

```bash
python -m experiments.unified_cognitive_controller.train \
  --device cuda \
  --task constant_action \
  --steps 600 \
  --batch-size 256 \
  --trials 6 \
  --report experiments/unified_cognitive_controller/reports/hidden_rule_smoke.json \
  --checkpoint-out artifacts/checkpoints/unified_hidden_rule_v1.pt
```

Only a passing run may save a checkpoint.

Blind-audit a saved parent without changing its weights:

```bash
python -m experiments.unified_cognitive_controller.audit_checkpoint \
  --device cuda \
  --task constant_action \
  --checkpoint artifacts/checkpoints/unified_constant_action_v1_seed2501.pt \
  --seed 2707 \
  --report experiments/unified_cognitive_controller/reports/constant_action_blind_seed2707.json
```

## Natural equivalence between independently acquired memories

The four-target shape-transfer parent still relied on a generator-created
opposite latent. The next rung replaces it with values produced by independent
visual support lifetimes. A fresh feedback-derived value is compared against
four stored values; any stored value that earns the same verifier behavior is
valid.

A discarded probe established that the relation is present in the frozen
latents. The deployed repair is a shared 32-unit pair scorer plus a
zero-initialized opening (12,354 parameters). A straight-through hard row
choice avoids averaging across disconnected successful retrieval intervals.
Only scalar candidate outcomes train it.

Two seeds reached 100% on held-out banks containing one, two, or three
equivalent rows after 1,024 verifier bits. The parent stayed near 50%.
Reward-shuffled training reached 43.55%; exact-duplicate-only training reached
86.91%. Probe, stored-relation, and retrieved-value interventions all caused
large collapses. Across both accepted seeds, 256 physical disk banks reloaded
exactly and behaved at 100%, and every older retention gate passed.

A valid counterfactual replay kept pixels and every candidate-bank tensor
fixed while reversing the target verifier rule. The fresh latent and selected
physical row flipped in 100% of cases, with 100% behavior in both worlds.

Full record:
`session_records/natural_memory_equivalence_2026-07-29/README.md`.

## Learned equivalence drives online consolidation

The learned relation now controls a real capacity-two memory bank. A stream of
16 independently rendered experiences contains two hidden binary behaviors.
New values merge with relation-equivalent rows, fill a free row when novel,
or replace the least-used row when full. Only a scalar calibration scale and
bias train from observed verifier outcomes; semantic rule/equivalence labels
and correct memory actions stay hidden.

Two independent 64-verifier-bit runs reached 99.46–99.51% held-out visual
accuracy and retained both behaviors in 98.93–99.02% of 1,024 streams. A
32-bit race passed only one of two seeds and was not promoted. Shuffled
verifier outcomes produced 50% behavior on both matched seeds, and relation
inversion reduced distinct-skill retention below 0.9%. All 256 promoted
physical banks reloaded exactly, inherited behavior remained intact, and the
counterfactual selection flipped in 100% of cases.

The physical claim is an 8× logical-row reduction (16 to 2), not an 8× file
reduction: serialized bytes fell to 32.41% because metadata is fixed.

Full record:
`session_records/equivalence_consolidation_2026-07-29/README.md`.

## Preserve diversity inside learned equivalence classes

The 8× consolidation result revealed a future-distribution cost: one
representative per hidden behavior reached only 97.17–97.47% when memories
acquired from bars were queried through never-trained disconnected dot-pair
objects. The uncompressed bank remained near 99.5%, localizing the gap to
discarded representative variation.

The online policy now permits a bounded diversity reserve inside each class
discovered by the learned relation. With two representatives per behavior,
two independent zero-update audits reached 98.36% and 98.57% on dot pairs,
100% on bars, and 99.69–99.72% on unseen diamonds. This uses four of sixteen
logical rows. A matched first-four control reached only 91.99–92.13%.

Across 2,048 physical banks every reload was exact. Zeroed values fell to
chance; counterfactual rule reversal retained 98.46–98.54% accuracy and
flipped the selected row in 98.07–98.36% of cases. No model tensor changed,
so the result is zero-shot reuse of the learned memory relation rather than
additional training.

Full record:
`session_records/diversity_preserving_consolidation_2026-07-29/README.md`.

## Predict when extra representatives are worth reading

An ordinary success predictor collapsed to the shallow action because shallow
reading already succeeds on roughly 99% of events. A marginal-success
diagnostic instead asked whether deep reading corrects a shallow failure. The
latent query plus first representative from each learned class made that event
0.994 AUC decodable.

A 32,097-parameter critic now predicts this marginal benefit from those
latents. It trains only from executed shallow/deep scalar outcomes. At the
replicated 16,392-bit frontier, held-out adaptive accuracy is 99.57% on both
seeds while mean comparisons fall from about 5.997 to 2.092–2.094. The
accuracy-first utility exceeds always-deep in both replicas. A 8,196-bit rung
was seed-sensitive and rejected.

The critic deep-reads only 1.34–1.49% of events, with the highest rate on
disconnected dot pairs. Feature shuffling removes the advantage, memory
zeroing falls to chance, verifier shuffling fails, all physical reloads are
exact, and inherited behavior is bit-identical. The population search used
147,456 separately accounted verifier outcomes shared across twelve critic
initializations.

Full record:
`session_records/adaptive_representative_read_2026-07-29/README.md`.

## Let verified use determine physical memory survival

The adaptive read critic established when extra representatives are useful, but
all six physical rows still survived forever. The next rung attaches a generic
protection signal to memory experience. An extra row earns protection only
when a requested deep read succeeds and the corresponding shallow read fails.
Rule bits, appearance names, semantic clusters, and correct pruning actions
remain verifier-private.

Protecting an exact third row did not generalize and was rejected. Protecting
the bank's diversity reserve after any extra row causally rescues an error did.
Across three 4,096-bank graduation replicas, physical memory falls from about
5.996 to 4.34–4.37 rows while accuracy stays at 99.53–99.56%, within 0.063
percentage points of the full store. A matched history shuffle performs worse
on every seed.

The controller is unchanged. Compaction physically removes rows using the
existing disk-backed memory boundary. All 384 audited compact stores reload
their latent and history tensors exactly; logical rows fall roughly 28% and
serialized bytes roughly 15%. Reversed-task behavior remains intact and
zeroed memory falls to chance.

This is a concrete bridge from the previously learned volatility principle to
adaptive *capacity*: frequently proven diversity becomes stable, while unused
reserve remains disposable.

Full record:
`session_records/adaptive_physical_pruning_2026-07-29/README.md`.

## Lossless cold archive with a self-thinning hot working set

Permanent pruning saves disk but cannot recover a discarded variant when an old
difficulty returns. The next rung separates preservation from attention:
six-row consolidated memories remain losslessly on disk while a roughly
four-row working set is materialized in RAM/VRAM. A cold row is promoted only
after it causally rescues a failed hot attempt; its generic protection trace
decays during quiet intervals.

A six-way verifier-side race selected decay `0.90`: faster decay failed
reactivation gates, while slower `0.97` retained too much stale context. The
selection stream and all counterfactual search compute are accounted separately
from three untouched 4,096-bank graduation seeds.

During an easy interlude, the adaptive hot set falls to 4.07–4.08 rows versus
4.24–4.26 for permanent cumulative protection and about 5.995 on cold disk,
while accuracy remains 99.97–100%. When the hard dot-pair distribution returns,
first-attempt accuracy beats fixed-core and matched shuffled-evidence
controls on every seed. Its paired advantage over fixed core grows from
0.027–0.052 percentage points in the first four returning rounds to
0.116–0.140 points in the final four.

The reusable `TieredLatentMemory` persists cold tensors, learned representative
ranks, protection and threshold. All 384 physical archives reload exactly,
promote after a rescue, thaw after quiet intervals, and retain every cold row.
Only 0.68–0.69% of hot events execute a cold retry. Corrupting cold values falls
to chance, and every controller tensor stays bit-identical.

Full record:
`session_records/adaptive_hot_cold_memory_2026-07-29/README.md`.

## Acquire, compile, compound a third appearance

The pair-relation bridge now rehearses every previously mastered appearance,
not only the first bars form.  This closes a false-promotion hole in which dots
could reach 95% while diamonds silently fell to 82.54%.  Relation replay is
also excluded from the unrelated-skill locality cost: the relation slot must
remain active on all of its learned renderings.

Eight permissive acquisition updates followed by 48 refiner-only consolidation
updates produced 99.96% bars, 97.83% diamonds, and 96.44% disconnected dot
pairs on an independent 8,192-lifetime audit.  Every appearance passed blank,
missing-object, valid pixel-counterfactual, and prediction-flip controls.  The
run used 1,792 new lifetimes and 37,632 total verifier outcomes.

At that same new-experience budget, the two-contour ancestor beat the earlier
bars-only ancestor by +9.66, +6.29, and +7.42 percentage points on paired
seeds.  A reset-slot control could not retain the combined repertoire.  An
optional-compute audit found that zero extra recurrent thoughts already
masters all three appearances, so the relation is deployed at one controller
pass per sensory event.

The original 56-update endpoint was seed-sensitive.  A later fixed
64-acquisition plus 32-consolidation schedule crossed the ignition valley on
three of three fresh seeds without changing the architecture.  Mean held-out
bars, diamonds, and dot-pair accuracy was 99.61%, 96.62%, and 97.81%, with
every unrelated retention score above 90%.

A proposed additive gate extension and a matched whole-slot version also
worked, but the old architecture passed the same seeds and budget.  Added
capacity and population selection are therefore not credited for the gain.
The promoted seed-9672 checkpoint passed a fresh 8,192-lifetime causal audit
at 99.65%, 97.85%, and 97.78%, and already operates at zero optional thoughts.
The next curriculum axis is a new relation on the familiar appearances.

Full record:
`session_records/pair_relation_robust_compound_2026-07-29/README.md`.

## Compound a genuinely new visual relation

The robust three-appearance same/different controller acquired a new
larger/smaller relation on familiar bars. The corrected renderer samples
adjacent pairs from five overlapping absolute sizes, bounding any one-object
shortcut at 62.5%. An earlier two-size renderer was rejected after its
one-object ablation remained at 75.5%.

The replay loop now cycles the full earlier relation repertoire at unchanged
experience cost. A 512-update probe repaired dot-pair retention in 35 seconds;
a 2,048-update rung preserved every relation appearance; only then did four
4,096-update population members run. Three passed internal mastery and
retention. Seed 21475 alone passed every independent compounding gate and was
promoted.

Two fresh 8,192/16,384-lifetime audits scored 92.05%/91.96%. Removing the
second object reduced accuracy to 60.53%/60.40%; disabling the inherited
same/different read reduced it by 10.70/10.34 points. Every bars, diamonds,
dot-pair, binary-mapping, visible-context, and XOR retention gate passed.

The skill already runs at the physical minimum of one controller pass per
event. Optional thought monotonically reduced accuracy from 91.92% at zero
extra steps to 87.22% at eight. Experience efficiency must therefore be
optimized first and compute second, with accuracy and retention as hard gates.

Magnitude has not transferred to unseen diamond or dot-pair contours. The next
frontier is a gradual magnitude appearance bridge with a reset-read control.

Full record:
`session_records/pair_magnitude_compounding_2026-07-29/README.md`.

## Gradual magnitude learning compounds forward

The first magnitude appearance bridge now closes the compounding loop. A
renderer audit first caught and fixed a false morph in which thresholding made
every nonzero blend the same union contour. With a real continuous
bars→diamonds axis, the parent masters 14.0625% morph and first fails at
15.625%.

Editing the mastered magnitude slot damaged bars and was rejected. The
successful controller freezes every mastered tensor and appends one zero-output
64-unit successor that reads the immediately preceding magnitude slot. Eight
updates on 256 new lifetimes, with 224 replay lifetimes, pass every gate on
three of three seeds. Target accuracy is 90.37–91.22%; a matched reset
immediate-magnitude control remains at 52.60%.

The promoted checkpoint passes a fresh 16,384-lifetime audit at 91.36%.
Deleting object two falls to 60.52%; disabling inherited reads falls to
79.20%. Bars magnitude, all three relation appearances, binary mapping,
visible context, and XOR remain mastered.

Most importantly, a paired parent/child frontier audit shows real forward
transfer. At the next unseen 17.1875% morph, the parent scores 88.57% and
fails; the child scores 90.68% and passes with zero new training. The child
also masters 18.75%, and its advantage grows to +3.51 points by 25%.

Zero optional thought remains optimal. The next exact rung is 20.3125%, the
first point beyond the child’s current causal mastery.

Full record:
`session_records/pair_magnitude_gradual_bridge_2026-07-29/README.md`.

## Consolidate experience before advancing

The next boundary exposed a fixed-size plasticity failure rather than a need
for more data. Refining the latest magnitude slot on 512 fresh lifetimes
reached only 89.29–89.65%, and sometimes damaged mastered contours. A gradient
probe found the reason: frozen-teacher rehearsal has essentially zero gradient
before the first student update, so preservation reacts only after the shared
slot has moved.

The successful schedule generated one balanced packet of 128 new target
lifetimes and 128 rehearsal lifetimes, then made 16 internal optimizer passes
over that fixed packet. This consumed 768 new verifier bits and 1,536 total
unique bits. The 388,191-parameter architecture did not grow. Three of three
seeds passed complete acquisition, causality, and repertoire gates at
20.3125%, with target accuracy of 90.02–90.47%.

Matched controls failed: resetting inherited magnitude reached 89.08%; one
pass over the same packet failed counterfactual mastery; 512 fresh lifetimes
at the same 16 updates reached 89.90%; shuffling new outcomes reached 89.48%.
Thus both inherited knowledge and repeated consolidation of aligned experience
are necessary.

The selected checkpoint passed a fresh 32,768-lifetime audit at 90.22%.
Deleting object two fell to 60.61%; disabling inherited reads cost 11.71
points; every old skill remained mastered. It also generalized without
training to 20.5078% and 20.7031%, where its parent failed. Zero optional
thought remains optimal.

This establishes that private computation can substitute for additional
verifier experience in a fixed-size learned concept, not yet a general
stable-bits threshold across task families. The next exact contour failure is
20.8984375%.

Full record:
`session_records/pair_magnitude_experience_consolidation_2026-07-29/README.md`.

## Halve consolidation compute on the next exact contour

The fixed-size magnitude controller advanced to the first non-robust
20.8984375% contour using the same 128 new lifetimes and no new parameters.
Fixing gate-leak annealing to 16 updates made acquisition prefixes directly
comparable. Four, six, and seven passes failed; eight, twelve, and sixteen
passed. The first robust prefix therefore cuts optimizer work from 16 to 8
passes and lifetime exposures from 4,608 to 2,304.

The eight-pass recipe passed three of three fresh seeds. Reset and
verifier-outcome-shuffled controls failed at 88.10% and 89.62%. On 32,768
independent lifetimes the selected checkpoint reached 90.21%, lost 29.57
points when object two was removed, lost 12.09 points when inherited reads
were disabled, and retained four magnitude contours plus every older skill.
It remains optimal at zero optional thoughts.

A separate forward-transfer gate was deliberately not promoted: the child
mastered two unseen contours but improved the registered next rung by +0.188
points, just below the fixed +0.200 requirement. The next mechanism should
learn when to stop consolidation from task-agnostic learner-visible signals.

Full record:
`session_records/pair_magnitude_half_compute_2026-07-29/README.md`.

## Accumulated skill reduces next-frontier experience

The next experiment first rejected two false shortcuts. A learned stopping
probe made unsafe held-out decisions even after receiving sensory-latent and
attempted-outcome consistency features, so fixed consolidation remained. A
21.09375% training arm was also discarded when matched audits showed the
parent already mastered that contour.

At the genuine 21.484375% frontier, the untouched parent failed three of three
preflights. Sixty-four new lifetimes with eight passes failed 0/3. Ninety-six
lifetimes with eight passes passed only 1/3. Holding the 96-lifetime evidence
fixed and increasing private consolidation to twelve passes passed 3/3 while
preserving every older capability.

This reduces new experience from 128 to 96 lifetimes (25%) and total unique
evidence from 288 to 228 lifetimes (20.8%) relative to the preceding
acquisition rung. Resetting inherited magnitude knowledge fell to 87.95%;
shuffling new outcomes fell to 89.39%.

The selected fixed-size controller passed a 32,768-lifetime audit at 90.45%.
Across eight more paired streams the untouched parent mastered 0/8 and the
child 8/8, with +0.4677 percentage points mean accuracy and an improvement on
every stream. This is verified compounding sample efficiency: accumulated
skill reduces the new verifier evidence required for the next genuine
frontier.

Full record:
`session_records/pair_magnitude_experience_compounding_2026-07-29/README.md`.

## Compounding sample efficiency repeats

The 21.484375% controller generalized zero-shot through contours 56/256 and
57/256, then failed three fresh 16,384-lifetime streams at 58/256
(`22.65625%`). This established the next genuine frontier before training.

The first 48-lifetime pilot passed 3/3. A downward experience search found that
32 and 40 failed, 42 passed only 1/3 complete gates, and 44 passed 3/3. The
selected recipe therefore uses 44 new lifetimes / 264 verifier bits—54.2% less
than the preceding 96-lifetime acquisition—with the same twelve private
consolidation passes and unchanged parameter count.

At the exact 44-lifetime budget, reset inherited knowledge scored 87.62% and
shuffled outcomes 88.66%; both failed. A complete 32,768-lifetime audit passed
at 90.26% with every magnitude, relation, unrelated-skill, missing-evidence,
counterfactual, and inherited-read gate intact.

On eight matched population streams, the parent mastered 2/8 and the child
8/8; every child stream improved. This is the second consecutive verified
experience reduction in the magnitude curriculum: 128 → 96 → 44 new
lifetimes on progressively harder contours.

Full record:
`session_records/pair_magnitude_repeated_compounding_2026-07-29/README.md`.

## Cross-operation transfer becomes verifier-grounded

The `visible_pair_numerosity_smaller` primitive reuses exactly the same visible
count relation as the mastered larger-count operation but requests its inverse
under a distinct public operation cue. A new zero-output slot may read the
frozen parent's accumulated intention; it is trained from uniformly logged
opaque actions and their scalar outcomes.

Replacing independent selected-logit BCE with a centered attempted-action
policy gradient preserved truthful learning and eliminated misleading random
drift. At 1,024 new lifetimes, two truthful replicas averaged 57.20%; matched
outcome-shuffled arms averaged 38.34%, and matched inherited-intention
ablations averaged 37.41%. Prior relation, magnitude, and numerosity remained
retained.

The promoted claim is a replicated causal cross-operation learning advantage,
not mastery. Equality was rejected as a too-large perceptual jump, and scalar
or feature-wise multiplicative intention operators were rejected after tiny
races.

Full record:
`session_records/cross_operation_policy_gradient_2026-07-30/README.md`.

## Retracted fixed-operation cue interpretation

The overlaid public operation cue was not a harmless task marker. At full
strength it reduced the frozen parent's larger-count accuracy from 90.46% to
63.15%, explaining why both reward learning and disposable supervision
plateaued near 60%.

`CognitiveLifetimeBatch` can now carry an optional cue-only prestimulus frame.
`rollout` consumes it as ordinary sensory time without taking an action or
delivering reward, then processes the clean action-bearing frame. No task
metadata reaches the controller.

With the legacy action residual canonicalized into amodal intention, two
reward-only replicas reached 83.96% and 83.65% from 1,024 lifetimes. However,
the original 38–40% missing-cue control also removed the entire prestimulus
timestep. A corrected control kept the extra frame and blanked only its cue
pixels; accuracy remained 81.93%. The model had primarily identified the
fixed-smaller task from stream timing, so visual cue use was not established.

Full record:
`session_records/stream_separated_operation_2026-07-30/README.md`.

## Conditional operation closes the shortcut

`visible_pair_numerosity_operation` presents the same clean count relation but
requests larger or smaller on every event through one of two cue-only frames.
Each lifetime is operation-balanced. `reverse_operations=True` preserves all
stimulus pixels while swapping cue identity and complementing every answer.
When cue intensity is zero, the forward and counterfactual public streams are
bit-identical and their paired verifier answers are complementary.

A generic zero-output successor slot reads the inherited amodal intention and
adds a learned multiplicative feature:

```text
tanh(W × recurrent_state) × inherited_intention
```

No task ID, semantic operation, count, action label, or unattempted outcome
enters this path. At the matched 128-update seed, the product interface reaches
75.76% versus 71.60% for concatenation alone.

Two independent 256-update runs reach 81.08% and 79.68% from 2,048 lifetimes.
Timing-matched blank cues score 50.18% and 49.82%. Shuffled outcomes score
49.87%; removing inherited read content with identical architecture scores
49.78%. All three inherited skills remain near or above their arrival levels.

The stronger history-free counterfactual resets state for every event and
measures the same count scene under opposite cues. It reaches 70.26% and
67.27%, with 46.25% and 40.11% prediction flips; the paired blank control is
exactly 50% with zero flips. This establishes causal conditional-operation
learning, but the checkpoint remains unpromoted until it reaches 90% mastery
without retention loss.

Full record:
`session_records/conditional_operation_2026-07-30/README.md`.

## One-event sensory RAM removes the history shift

Fresh-state training localized a representation problem: it produced 84.21%
history-free operation accuracy but only 49.83% in a continuous sequence.
Previous-action augmentation, staged training, and a 50/50 distribution mixture
did not make one slot robust to both regimes.

`ControllerState.latest_event` is a generic one-step sensory buffer. Each call
to `step` overwrites it with the current vision latent. A new slot can read the
immediately preceding latent and bind it to the inherited intention:

```text
tanh(W × latest_event) × inherited_intention
```

Older checkpoints and slots do not read this field, and the appended slot
starts with exactly zero output. The interface adds state, not a task-specific
head or an extra inference step.

At 1,024 new lifetimes, two replicas reach 84.75% and 84.33% sequential
accuracy and 78.15% and 73.52% history-free accuracy. A matched snapshot-content
ablation reaches only 72.18% and 55.54%; matched shuffled outcomes reach
50.03%. Blank cues remain exactly chance and inherited skills remain above 90%.

At 2,048 lifetimes, the selected research candidate reaches 84.82% sequential
and 84.58% history-free with 82.25% counterfactual cue flips. It remains
unpromoted pending the 90% mastery gate.

Full record:
`session_records/event_snapshot_operation_2026-07-30/README.md`.

## Replicated previous-item sequence operation

The procedural-shape branch now crosses from passive short-term lookup into a
minimal sequence manipulation. A visual operation glyph selects direct lookup
or `previous item` while a separate unary visual cue selects the anchor. The
learner sees neither name: both are ordinary pixels, and learning still uses
only the scalar outcome of its own attempted opaque action.

The generator keeps the visual anchor separate from the verifier-private target
ordinal. At span three, direct and previous operations share cue identities but
can require different answers. Training can focus either valid previous anchor
and can place the operation at query position one, two, or three. Evaluation
reports every operation × cue and query-position × target cell, plus the
conflict subset where following the directly cued item would be wrong.

The curriculum is deliberately one-dimensional:

1. neutral operation glyph;
2. span-two previous atom;
3. first span-three anchor;
4. first anchor after one and two prior queries;
5. second anchor;
6. second anchor after one and two prior queries;
7. both anchors in natural query order.

Two independent lineages pass the final 24,576-lifetime audit at 98.30% and
98.59% overall. Previous-item accuracy is 97.78% and 98.23%; conflict accuracy
is 96.65% and 97.43%; every populated position/target cell exceeds 96.4%.
Complete memory reset returns to chance.

The operation audit is a valid pixel-level counterfactual: the same logical
lifetime is rerendered with the other glyph and its verifier target recomputed,
then replayed through the recurrent controller. Final counterfactual accuracy
is 97.57% and 97.84%. A matched shuffled-outcome control leaves conflict
accuracy near chance at 52.57% and damages inherited behavior.

The replica provides direct compounding evidence. Learning the second anchor in
isolation required 41,472 target verifier bits; delaying it to query two used
23,040; delaying it to query three began at 94.92% zero-shot and crossed every
gate after only 5,760 bits.

Full record:
`session_records/procedural_shape_previous_operation_2026-07-30/README.md`.

## Span-nine acquisition with event-age routing and protected rehearsal

The next working-memory rung exposed two separate issues. A zero-initialized
workspace-volatility write scale was a clean diagnostic but did not learn a
causal direction: four truthful and four outcome-shuffled runs converged to
the same negative scale and the same accuracy. The transient fast-memory
state was therefore rejected as a habit mechanism; volatility must be tied to
persistent, receipt-backed experience rather than inferred from one episode.

The useful representation change was smaller and task-agnostic: an optional
normalized event-age trace (a generic stream clock) is available to a new
successor slot. It is not a span, operation, task, or answer label. At 1,024
new span-nine lifetimes, the age-aware slot reached 82.66–83.40% across four
seeds, versus 76.50–78.29% for the matched replay/rehearsal smoke. Blank and
complete-reset controls stayed at chance. At 4,096 lifetimes the age-aware
race reached 88.54–89.12%, improving the earlier 86–88% range, but still
opened on old span-eight tasks.

The promotion recipe then combined the clock with a capped old-span replay
buffer, a small `0.003` residual/logit penalty on replay rows, and private
consolidation rather than more verifier data. With 8,192 new span-nine
lifetimes, 16,384 replay transitions, and 384 optimizer passes, the selected
checkpoint is:

`artifacts/checkpoints/span9_age_replay_pen003_e384_seed48001.pt`

Checkpoint SHA-256: `0c40c7f478d14234ae29108ef6236c50b6b5c73448b596d47910646827c9db1d`.

It reaches 90.46% on a fresh 4,096-lifetime span-nine audit (reverse-operation
accuracy 90.34%; blank sequence 49.64%; complete memory reset 49.42%). A
paired 2,048-lifetime audit against the span-eight parent retains every old
span with the worst margin −1.01 points and reaches 90.19% on span nine. Two
exact outcome-shuffle controls score 52.44% and 44.79%, so the gain is not
reward-independent drift. The checkpoint, high-count audit, retention audit,
and controls are stored under
`session_records/sequence_working_memory_2026-08-02/`.

This is the first span-nine result to pass mastery, memory-dependence,
causal-reward, and old-skill-retention gates together. The important lesson is
also architectural: a generic stream clock improved sample efficiency, while
habit protection still belongs at the persistent-memory boundary; a transient
workspace scalar alone is not enough.

The recipe independently replicated with seed `48002` without changing the
parent, replay cap, penalties, or training budget. Its 2,048-lifetime report
reached 90.69%; the independent 4,096-lifetime audit reached **90.58%**
(90.85% reversed-operation accuracy, 48.01% non-palindrome operation flips,
49.82% blank, and 50.00% complete reset). The paired 2,048-lifetime retention
audit reached **90.49%** on span nine, with every old span retained and a worst
old margin of **−1.21 points**. This turns the original result into a
replicated promotion, not a single-seed outlier. The new checkpoint and reports
are in `session_records/sequence_working_memory_2026-08-02/`. Two matched
outcome-shuffled controls reached only **53.78%** and **55.04%**, with
blank/reset controls at chance and no corresponding causal gain. This closes
the adversarial replication gate as well.

The next frontier is persistence across the memory boundary: serialize the
acquired skill, run private consolidation, reload it into a fresh process, and
re-run the same span-nine/retention/reversal gates. Until that is measured, the
claim is limited to replay-protected controller acquisition, not long-term
disk-memory consolidation.

## Span-nine skill memory survives external serialization

The first memory-boundary rung is now causal. The learned successor-slot state
was extracted into the separate artifact
`artifacts/memory/span9_skill_memory_seed48002.pt` (253,081 parameters), while
the parent controller remained the frozen computation core. A fresh process
rehydrated that artifact on top of the parent and reproduced the direct child
at **90.96%** on a 4,096-lifetime audit, with identical reverse-operation
accuracy (90.63%), blank accuracy (49.89%), reset accuracy (49.76%), and
operation-flip rate (48.11%).

The causal corruption control zeroed only the serialized successor-slot state;
the parent core and all verifier inputs were unchanged. Accuracy fell to
**75.64%** (reverse-operation 75.39%), while blank/reset remained at chance.
This is direct evidence that the new capability is carried by the external
skill artifact rather than being a reward-independent change in the core. The
artifact contains only learned slot parameters and parent provenance; no
correct actions, task labels, or semantic rule fields are stored.

Full evidence:
`session_records/sequence_working_memory_2026-08-02/span9_skill_memory_audit_seed48002.json`.
Artifact SHA-256:
`228341bd120757bb4ad287530f11f36773788ab44098ec837e89a8c6d25d8a04`.

The next frontier is to connect this serialized skill artifact to the generic
`DiskLatentMemory`/hot-working-set path and test a new span acquisition after a
fresh-process reload, with the same retention and corruption gates. This is a
memory-boundary result, not yet proof of learned multi-skill consolidation.
