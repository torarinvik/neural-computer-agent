# Unified cognitive controller

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

The same vision encoder, recurrent controller, generic workspace operations,
latent intention, and replaceable actuator adapter process every trial. The
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
