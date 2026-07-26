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

The capacity-4 policy was additionally applied zero-shot to capacity 8. It
reached 93.99%, but correct eviction fell to 71.48% and age corruption reduced
accuracy by only 3.91 points. That arm failed its causal gates and was not
promoted. The next gradual rung is capacity 5, with capacity-4 rehearsal.

The promoted checkpoint is
`artifacts/checkpoints/unified_memory_replacement_seed6101.pt`, SHA-256
`0178b15228e3d75a445abdb2376be1291a078f8b47236444fbd1824fab3d3b76`.

Claim boundary: this demonstrates controller-created, content-addressed latent
storage and recall across active-state resets, with actual disk serialization.
It also demonstrates learned write/skip and read/no-read decisions operating
together within the storage-efficiency gate, plus learned bounded replacement
under one simple recency-predictive utility distribution. Richer or
nonstationary utility, replacement beyond capacity 4, consolidation across an
unbounded stream, deletion/merging, modality transfer, and broad reasoning
remain open.

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
