# Span-three to span-four working-memory compounding (2026-08-02)

## Question

Does an acquired two-item retention skill make a harder three-item sequence
operation faster to learn, and can the new skill be added without forgetting
the old one? The learner sees only RGB frames, its own opaque attempted
actions, and scalar verifier outcomes. Sequence identity, operation, and
correct actions remain private to the verifier.

## Main result

The robust two-item checkpoint was used as the parent. The matched fresh arm
started from random weights. Both arms received the same 128-update span-three
curriculum (three items, two irrelevant X-shaped events, mixed forward/reverse
queries, and position augmentation):

| Arm | New verifier bits | Held-out accuracy | Stable 90% threshold |
| --- | ---: | ---: | ---: |
| inherited, seed 30003 | 24,576 | 93.55% | 21,504 bits |
| inherited, seed 30004 | 24,576 | 93.42% | 18,432 bits |
| fresh, seed 30005 | 24,576 | 74.98% | not reached |

The fresh model mostly learned the operation-blind shortcut: its valid
operation-reversal flip rate was 0.20%, while the inherited replicas were
58.37% and 58.61%. This is evidence of reuse, not merely a larger model or
more compute. The inherited arms crossed the mastery gate in roughly 19,968
bits on average; the fresh arm had not crossed it after 24,576 bits, giving a
conservative lower bound of 1.23x sample-efficiency improvement.

## Retention repair

The first inherited replicas exposed a real risk: span-two retention measured
80.23% and 90.87% on two seeds. The original rehearsal alternated distractor
counts but still trained only span three. A tiny, task-agnostic repair was
added: alternate span-two and span-three episodes, with the same two-distractor
distribution. Only 64 updates (32 of each span, 10,240 verifier bits) were
needed:

- span two: **100.00%** on an 8,192-episode independent audit;
- span three: **95.75%** on an 8,192-episode independent audit;
- span-three report audit: 95.72% over 4,096 episodes;
- blank sequence: 49.88% (chance);
- complete fast-memory reset: 49.91% (chance);
- valid sequence reversal: 66.67% prediction flips on non-palindromes;
- held-out position blends: 95.74% at every tested blend;
- workspace disabled: 72.27%, showing a partially redundant but causal
  workspace contribution.

This is the promoted checkpoint:
`artifacts/checkpoints/unified_sequence_working_memory_span3_seed30003_span2_rehearsal64.pt`.

## Adversarial controls

The inherited span-three run with outcomes shuffled between lifetimes stayed
at **50.00%** with **0.00%** operation flips. Blank-sequence and complete-reset
audits were also at chance on the non-shuffled runs. These controls rule out a
pixel-only or generator-order explanation for the gain.

## Span-four escalation

The next one-axis escalation started from the promoted span-three checkpoint,
not from random weights. Each 16-update run used two distractors, position
augmentation, and the balanced schedule `[4, 2, 2, 3, 3]`; the target span is
exposed first and the two mastered spans are rehearsed more often. The runs
used 2,944 fresh verifier bits. A fresh span-four arm used the same budget and
stayed at chance, providing the matched sample-efficiency control.

On a common 2,048-lifetime audit seed, the inherited parent scored 72.13%,
while two independent balanced children scored 78.08% and 77.23%. The
improvement is replicated at roughly +5--6 percentage points after only 2,944
new verifier bits. Their independent retention audits were:

| Checkpoint | Span 2 | Span 3 | Span 4 | Blank/reset controls |
| --- | ---: | ---: | ---: | --- |
| balanced seed 30514 | 99.96% | 95.80% | 78.14% | ~50% / ~50% |
| balanced seed 30517 | 100.00% | 95.54% | 76.87% | ~50% / ~50% |

Both children retain the earlier primitives within the two-point gate and
show a real span-four operation signal: their valid reversal flip rates are
about 46% and 43%, while blank and complete fast-memory reset remain at
chance. This is the first replicated span-four compounding result, not merely
an extrapolation from the span-three run.

The shuffled-outcome controls must be interpreted against the inherited
zero-shot baseline: span four is already partly transferable before training,
so a shuffled run is not expected to fall to 50%. On the same audit seed as
the normal child, the parent was 72.41%, normal seed 30514 was 78.06%, and
shuffled seed 30515 was 74.95% with a weaker 39.8% reversal flip rate. A
second shuffled arm fell to 60.03% with a 23.3% flip rate. Thus shuffling does
not reproduce the normal gain or causal operation sensitivity, but it is not a
clean chance control for this partially transferable task; we do not claim it
is one.

## Span-five escalation

After the span-four continuation reached 80.9%, a 16-update span-five smoke
test used the schedule `[5, 2, 3, 4]` (equal exposure to the new span and each
mastered span), two distractors, and 3,584 fresh verifier bits. Two inherited
seeds reached 73.48% and 74.50% on their 2,048-episode audits. The matched
fresh arm stayed at 50.00% with zero valid operation flips. On a common audit
seed, the parent was 69.76%, the normal child 73.48%, and the shuffled arm
64.74% with all-memory-reset accuracy 49.51%.

Retention remained intact. The first inherited child measured span-2/3/4/5
at 99.67% / 95.56% / 82.03% / 73.79% on 4,096-episode audits; the second
measured 99.88% / 95.59% / 82.32% / 73.60% on 2,048-episode audits. Blank and
complete-reset controls stayed approximately 50%, and span-five reversal
flip rates were 48.3% and 49.5%. This is a replicated five-item transfer
signal, not a claim of five-item mastery.

## Span-six escalation and rehearsal failure

A 16-update span-six smoke test used `[6, 2, 3, 4, 5]`, two distractors, and
4,224 fresh verifier bits. The inherited arm reached 71.20%, while the
matched fresh arm reached 50.50% with zero operation flips. This confirms that
the compounding signal extends to six items, but the first schedule exposed a
retention failure: span-2 and span-3 fell to 94.29% and 91.41%.

The failure was repaired without changing the model. A second 16-update run
used the span-heavy schedule `[6, 2, 2, 3, 3, 3, 4, 5]`. Its independent audit
measured:

| Span | Accuracy | Reversal flips | Blank/reset |
| --- | ---: | ---: | ---: |
| 2 | 100.00% | 100.00% | ~50% / ~50% |
| 3 | 94.84% | 67.38% | ~50% / ~50% |
| 4 | 84.51% | 48.11% | ~50% / ~50% |
| 5 | 75.59% | 51.51% | ~50% / ~50% |
| 6 | 71.07% | 48.36% | ~50% / ~50% |

The outcome-shuffled span-six control reached only 54.17%, with a 31.75%
operation-flip rate and a 50.10% complete-reset control. The lesson is
important: increasing span is itself a continual-learning stress test, and
rehearsal must be weighted toward the earliest fragile primitives. The
span-six transfer is real, but the default equal rehearsal schedule is
rejected.

## Span-seven escalation

The repaired span-six checkpoint was then tested on seven items with two
distractors. The weighted schedule `[7, 2, 2, 3, 3, 3, 4, 5, 6]` exposed the
new span while continuing to protect the earliest primitives. After 3,776
fresh verifier bits, the inherited arm reached 68.29%; a matched fresh arm
stayed at 50.00% with zero operation flips. The independent inherited audit
measured spans 2--7 at 100.00%, 94.61%, 84.07%, 75.96%, 72.37%, and 68.36%.
Blank and complete-reset controls remained approximately 50%, and span-seven
reversal flips were 48.66%.

The outcome-shuffled span-seven control reached only 58.23%, with a 26.67%
sequence-reversal flip rate and a 50.41% complete-reset control. This is the
next replicated compounding result, while still only a smoke-level transfer
measurement rather than seven-item mastery.

## Plasticity-regulation probes

The span-seven result also motivated a small architecture probe. Three
task-agnostic safeguards were tested offline before changing the trainer:

1. A uniform parameter trust-region and a diagonal gradient/Fisher anchor did
   not replace rehearsal. With a lighter rehearsal schedule they left span-3
   around 89--91% while span-7 was about 70%.
2. A zero-output frozen-core skill adapter preserved the inherited controller
   (about 100% span-2 and 95.6% span-3) and reached 67.9% span-7 after 32
   updates, essentially matching the 68.0% full-plasticity baseline. This is
   the most promising plasticity direction because it creates a genuinely
   protected growth compartment.
3. Training that adapter without rehearsal preserved the core less well
   (span-3 about 88.7%), because its gate opened on old inputs. Adding a
   strong gate-usage penalty mostly closed the gate and reduced learning.

Conclusion: plasticity should be regulated by a learned, usage-conditioned
locality gate with a protected core, not by a uniform weight penalty or a
manually fixed gate sparsity target. The current weighted rehearsal remains
the promotion path; the adapter is a diagnostic candidate until it passes a
matched multi-seed retention and transfer audit.

## Span-eight escalation: transfer signal, not yet mastery

The next one-axis escalation used the same early-span-heavy schedule,
`[8, 2, 2, 3, 3, 3, 4, 5, 6, 7]`, two distractors, and position
augmentation. The inherited arm started from the span-seven checkpoint and
the fresh arm had the identical controller size, optimizer, budget, and
verifier stream shape. After only 512 episodes / 2,048 new verifier bits:

| Arm | Accuracy | Operation reversal flips | Complete reset |
| --- | ---: | ---: | ---: |
| inherited, seed 30564 | **66.85%** | **44.27%** | 49.17% |
| fresh, seed 30565 | 50.00% | 0.00% | 50.00% |
| outcome-shuffled inherited, seed 30566 | 57.71% | 56.72%* | 50.39% |

The inherited child also scored 65.45% under sequence reversal and 66.46% at
the fully shifted position blend. Blank-sequence accuracy was 49.80%, so the
gain depends on retained sequence content rather than a constant action.
The shuffled arm is intentionally **not** treated as a chance control: it
inherits useful span-seven behavior, but it failed to reproduce the normal
child's 9.14-point gain and its sequence-reversal flip rate fell to 23.13%.
(*The operation-cue flip statistic alone is not a pass criterion for this
control.)

This was initially recorded as a new transfer signal rather than mastery. The
promotion audit then evaluated 512 lifetime-disjoint episodes on MPS. Relative
to the preceding span-seven audit, spans 2--7 changed by -0.20, -1.51, -1.89,
-0.57, -1.34, and +0.00 percentage points, respectively: every older skill
stayed within the two-point retention gate. Blank and complete-reset controls
remained at chance. The earlier 32-episode smoke audit is retained as a
low-count diagnostic, not as the promotion evidence.

The independent span-seven replication immediately before this escalation
reached 66.02% from 1,888 verifier bits, with 45.92% operation flips and
50.33% complete-reset accuracy. This lower-but-positive result is retained as
the appropriate seed-variance bound rather than hidden behind the stronger
span-eight child.

## Span-eight promotion audit

The second inherited span-eight seed used the identical schedule and 2,048
new verifier bits. The matched fresh and shuffled controls used the same
controller configuration and budget:

| Arm | Accuracy | Operation reversal flips | Blank | Complete reset |
| --- | ---: | ---: | ---: | ---: |
| inherited, seed 30564 | 66.53% | 43.15% | 50.24% | 50.10% |
| inherited, seed 30567 | 66.36% | 40.81% | 49.27% | 49.02% |
| fresh, seed 30568 | 50.00% | 0.00% | 50.00% | 50.00% |
| shuffled inherited, seed 30569 | 50.73% | 37.94% | 50.83% | 50.49% |

The inherited arms therefore reproduce a 16--17 percentage-point gain over
fresh weights, while outcome shuffling removes that gain. The operation and
reset controls show that the child is using retained sequence information and
the query operation, not a fixed action or an accidental pixel watermark.
This is now a **promoted span-eight compounding transfer result**. It remains
an acquisition result rather than eight-item mastery: the next run must use
private consolidation and then a larger span-eight mastery audit.

## What this establishes

This is a verified compounding working-memory result: a learned retention
primitive makes a harder manipulation primitive reachable with fewer fresh
verifier bits, and explicit rehearsal prevents the older primitive from being
overwritten. It is still a specialist sequence branch, not yet a generic
variable-capacity memory or a fully consolidated repertoire capability.

## Reward-buffer readout breakthrough and retention repair

The frozen inherited controller's latent state contains substantially more
usable information than its online action path extracts. A diagnostic probe on
the frozen state reached 92.26% on the span-eight relation, while the online
bandit adapter experiments stayed near 67%. This localized the next bottleneck
to action readout and credit assignment rather than missing representation.

We therefore trained only a zero-initialized, generic action adapter from a
replay buffer containing the controller-visible latent, the opaque action that
was attempted, and that attempt's one-bit outcome. The correct unattempted
action and task labels never entered the buffer. With 8,192 target lifetimes,
width 256, position augmentation, and 128 private optimizer epochs, the
adapter reached 90.14% on a lifetime-disjoint span-eight audit. An independent
seed reached 90.80%:

| Arm | Span-eight | Blank | Complete reset |
| --- | ---: | ---: | ---: |
| real outcomes, seed 30981 | **90.14%** | 49.24% | 48.93% |
| real outcomes, seed 30983 | **90.80%** | 49.83% | 50.10% |
| matched shuffled outcomes, seed 30982 | 47.61% | 49.83% | 49.66% |

The first width-256 run without rehearsal also reached 90.97%, but its spans
2--7 retention audit fell to 76--82%; it was rejected. The repair replayed
balanced earlier-span streams (2,2,3,3,3,4,5,6,7) while fitting span eight.
On 512 fresh audit episodes per span, the inherited parent versus the repaired
candidate was:

| Span | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| parent, seed 30983 audit | 98.83% | 90.69% | 79.83% | 73.55% | 70.93% | 68.89% | 66.06% |
| replay-trained candidate | **99.61%** | **99.93%** | **96.73%** | **91.02%** | **82.55%** | **80.02%** | **90.43%** |

Thus the repair did not merely preserve earlier skills: it improved every
audited span while adding span-eight mastery. The shuffled-outcome arm stayed
at chance, and blank/complete-reset controls stayed at chance. This is a
promoted **action-readout and anti-forgetting diagnostic breakthrough**, not
yet proof that the full controller discovers the adapter from online reward or
that the consolidated long-term memory stores it. The rehearsal streams here
are newly collected diagnostic data; persistent disk-memory reuse remains an
open experiment.

## Span-nine persistence and the stability--plasticity frontier

The replay bank is now serializable and reloadable. It stores only the
controller-visible latent features, base logits, opaque attempted actions, and
binary attempted-action outcomes, with a schema and parent provenance. A
save/load smoke test reproduced the same adapter behavior on a second process.

The first span-nine transfer used the controller's existing successor-slot
stack, preserving the promoted span-eight adapter as a frozen parent and
adding one zero-output slot. Fresh learning reached 86.98% at 4,096 new
lifetimes, but span-eight fell by roughly 14 points. Adding the persisted old
bank as a zero-outcome plasticity constraint reached 81.99% on span nine while
keeping every span-2--7 score within the two-point retention gate and span eight
within 1.88 points of the parent:

| Arm | Span nine | Span eight | Blank | Complete reset |
| --- | ---: | ---: | ---: | ---: |
| fresh successor slot, 4,096 | **86.98%** | 75.78% | 50.54% | 49.63% |
| persisted replay + protected plasticity, 4,096 | 81.99% | **88.11%** | 49.09% | 49.22% |

The matched parent on that audit was 75.52% on span nine and 89.99% on span
eight. Thus persistent replay gives a real, causal stability--plasticity
tradeoff: it learns a novel skill substantially above the parent while
preventing the catastrophic forgetting seen in fresh-only training. It is not
yet a mastery promotion because the new skill is below the 90% bar. Naive full
replay, residual penalties, staged gate refinement, nonlinear gates, and
source-provenance gate supervision were all retained as bounded controls; none
improved the Pareto frontier.

This localizes the next bottleneck more precisely: the controller can represent
old/new information, but routing provenance is not the same as decoding the
correct new action. A nearest-memory prototype was too weak (old/new means
1.485 versus 1.663), while a verifier-side classifier using hidden state plus
workspace reached 97.96% old/new held-out accuracy. A disposable correct-action
probe reached 84.49% from hidden/event, 86.75% after adding workspace, and
90.20% after also exposing the parent's adapter latent. Thus the useful action
information is present, but the sparse attempted-action objective is not
learning to exploit it reliably.

The workspace-aware successor controls make that distinction causal. A fresh
workspace slot reached 86.09%; a protected replay version fell to 75.95%; a
source-provenance gate reached 85.05%; a 64-wide read bottleneck reached
86.70%; removing the inherited adapter read fell to 82.47%; and weighting the
rare successful attempts threefold reached 87.22%. Rehearsing span eight while
training span nine reached 85.37% at the same budget. A guarded joint
adaptation of the parent action reader with span-eight rehearsal reached
83.92%, showing that unfreezing the reader is not a free fix at this sample
scale. These are bounded controls, not mastery claims, but they rule out “just
add workspace,” “just classify old versus new,” and “just rehearse the old
span” as immediate fixes.

The strongest fresh candidate also failed the promotion audit. Its span-nine
score was 87.22%, but paired retention margins versus the parent were -7.23,
-10.21, -10.74, -3.97, -3.66, and -14.11 points on spans 3--8 (span two was
-0.78 points). The outcome-shuffled control reached 46.27% with blank and
reset controls at chance. This is useful negative evidence: the candidate was
using real reward structure, but its learned plasticity was far too volatile
to promote.

The durable lesson for plasticity is that the successor gate is already a
per-transition scalar, but its target should not be a hand-written volatility
label. It should be learned from the same observed outcome stream and judged by
verifier-owned retention and transfer deltas. The next high-ROI experiment is
therefore a small action-conditioned critic (or equivalent dense use of the
observed reward) feeding the gate, with no correct-action labels in the learner.

## Critic and per-cell plasticity follow-up

The action-conditioned critic was implemented as a zero-impact auxiliary head:
it predicts success for each opaque action from the same latent, while a learned
scalar controls whether its centered preference reaches the actuator. The
critic reached 87.70% on the fresh span-nine arm; a fourfold critic-loss weight
reached 87.61%; critic plus protected replay reached 75.56%; and a critic that
also read the new RAM-usage EMA reached 87.43%. The best critic retention audit
still lost 2.5--13.3 points on spans 3--8. Thus the critic is a promising
credit-assignment component, but neither it nor replay makes plasticity safe.

The usage experiment exposed a more fundamental architectural issue. In the
promoted parent, all four RAM slots are exact clones: content and usage have
zero within-slot variance. The controller allocates four tensors, but it has no
generic slot identity, so a per-cell volatility scalar has nothing individual
to protect. Enabling fixed address tokens in a diagnostic control produced a
small nonzero usage difference, but it was not trained and did not establish a
capability gain. The fixed-address successor reached 87.33%; allowing the
generic read/write address scales to train reached 88.00%, but its usage slot
range remained only 0.00067 and it still lost 1--14 points on spans 2--8. This
is a bounded addressability result, not a promotion: address scores alone do
not make the workspace store distinct cell content. The next architecture must
first learn addressable RAM locations; only then can usage-conditioned
plasticity be meaningfully tested.

A scale sweep found a useful operating point: address strength 1.0 created a
0.102 mean content range and 0.0067 usage range across slots while preserving
the parent at 89.8% on span eight. A fresh usage-aware critic from that parent
reached 88.3%, but still lost 1.7--13.9 points on old spans. Protected replay
preserved every old score exactly, but also preserved the parent’s 75.9% span
nine score: the gate shut down completely. Addressability exposes the
stability--plasticity tradeoff cleanly; it does not solve it.

The write-content follow-up sharpens the target. Learning a single global
address-conditioned write scalar reached 87.7%. A fixed generic per-slot write
offset reached 88.8%, but lost 7--14 points on spans 2--5 and 8. The extra
transfer is therefore real but not retention-safe. The next write mechanism
must use a per-row plasticity gate or usage-conditioned write strength—not a
global scalar or an always-on offset.

## Provenance-gate correction and the habit hypothesis

The first staged-gate run exposed a training-control bug rather than a new
capability result. Its provenance term was evaluated only on persisted replay
rows, so every source target was ``old``; the gate could only learn to close.
The corrected control trains the source term on the mixed fresh-plus-replay
set, with an explicit regression test for that split.

The correction improves the result but does not change the frontier. At a
matched 128-epoch fresh budget, a source-weight-1 gate reached 83.55% on span
nine while preserving spans two--eight within 0.8 points of the addressed
parent. A lighter source weight (0.2) reached 85.59% but lost 4.9 points on
span eight. Joint adapter training with replay reached 84.51% and lost 5.4
points on span eight. The original degenerate staged gate reached only 75.41%.
These controls rule out the easy interpretation that a provenance classifier
alone solves the stability--plasticity tradeoff: provenance is not available
at deployment, and the latent features of old and new spans overlap.

Scaling the fresh stream to 8,192 lifetimes did not rescue the tradeoff:
balanced provenance training reached 84.01% (span eight 86.21%), below the
promotion bar. A follow-up that regressed fresh gate scores to their
post-acquisition values while driving old scores to zero reached only 78.65%.
Gate magnitude is therefore not a stable teacher signal; this branch is closed
rather than tuned further.

The proposed ``habit`` mechanism is nevertheless the right memory-side idea.
The psychological claim is usually called **Jost's law of forgetting** (with
Ribot's law describing the related recency gradient): an older trace is more
resistant to interference than an equally strong new trace. Our persistent
memory already implements the computational analogue: each row has a generic
volatility scalar; verified success lowers it (freezes a useful habit), verified
failure raises it (thaws a bad habit), and stale rows slowly thaw. A reward-only
selector has already achieved 100% stable retention and 100% new acquisition in
the bounded external-memory audit, including row-shuffle and save/reload
controls. What remains unproven is end-to-end use of that scalar by the visual
sequence controller's online write path. The next experiment must therefore
attach row-local volatility to real writes and score both acquisition speed and
retention; it must not add a fixed age label or freeze all old content.

## Next frontier

The next highest-ROI experiment is an **online row-local habit gate**: expose
the existing volatility scalar at the controller's write decision, initialize
its influence at zero, and update it only from physical read receipts and
verified attempted-action outcomes. The gate should be trained/validated in
three tiny stages: (1) old stable versus failed-decoy rows, (2) a task shift
where a new row competes for bounded capacity, and (3) sequence span-nine
acquisition with the promoted span-eight parent. Acceptance remains span nine
at or above 90%, spans two--eight within the two-point retention gate, shuffled
outcomes at chance, and blank/reset/reversal controls intact. Do not increase
controller width or add modalities until this causal write-path test passes.

## Artifacts

- `span3_seed30003.json`, `span3_seed30004.json`: inherited replicas.
- `span3_fresh_seed30005.json`: matched fresh control.
- `span3_shuffled_seed30006.json`: shuffled-outcome adversarial control.
- `span3_seed30003_span2_rehearsal64.json`: retention repair.
- `span3_smoke_seed30001.json`, `span3_seed30002.json`: early budget checks.
- `span4_balanced_rehearsal_seed30514.json`, `span4_balanced_rehearsal_seed30517.json`:
  replicated inherited span-four runs.
- `span4_smoke_fresh_seed30512.json`: matched fresh span-four control.
- `span4_shuffled_seed30515.json`, `span4_shuffled_seed30516.json`:
  outcome-shuffled controls, interpreted against the non-chance transfer
  baseline as described above.
- `span4_continuation_seed30518.json`: 32-update span-four continuation.
- `span5_smoke_inherited_seed30519.json`, `span5_smoke_inherited_seed30522.json`:
  replicated inherited span-five transfers.
- `span5_smoke_fresh_seed30520.json`: matched fresh span-five control.
- `span5_smoke_shuffled_seed30521.json`: outcome-shuffled span-five control.
- `span5_continuation_seed30523.json`: 32-update span-five continuation.
- `span6_smoke_inherited_seed30525.json`, `span6_smoke_fresh_seed30526.json`:
  inherited/fresh span-six transfer pair.
- `span6_rehearsal_repair_seed30527.json`,
  `span6_rehearsal_repair2_seed30528.json`: weighted retention repairs.
- `span6_smoke_shuffled_seed30529.json`: outcome-shuffled span-six control.
- `span6_rehearsal_replica_seed30530.json`: second-seed weighted repair.
- `span7_smoke_inherited_seed30531.json`, `span7_smoke_fresh_seed30532.json`:
  inherited/fresh span-seven transfer pair.
- `span7_smoke_shuffled_seed30533.json`: outcome-shuffled span-seven control.
- `span7_replica_seed30563.json`: independent inherited span-seven replication.
- `span7_replica_prefix_seed30563.json`: bounded eight-update prefix.
- `span8_smoke_inherited_seed30564.json`: inherited span-eight transfer smoke.
- `span8_smoke_fresh_seed30565.json`: matched fresh span-eight control.
- `span8_shuffled_seed30566.json`: inherited outcome-shuffled control.
- `span8_retention_smoke_seed30564.json`: low-count spans-2--8 regression
  smoke audit; not a promotion-grade retention audit.
- `span8_retention_audit_mps512_seed30564.json`: promotion-grade 512-lifetime
  retention audit across spans 2--8.
- `span8_replica_inherited_seed30567.json`,
  `span8_replica_fresh_seed30568.json`, and
  `span8_replica_shuffled_seed30569.json`: matched second-seed promotion
  controls.
- `span8_representation_probe_seed30901.json`,
  `span8_richer_representation_probe_seed30921.json`: frozen-state readout
  localization probes.
- `span8_reward_buffer_normal_seed30941.json`,
  `span8_reward_buffer_4096_seed30951.json`,
  `span8_reward_buffer_8192_seed30961.json`: reward-buffer data curve.
- `span8_reward_buffer_8192_width256_seed30971.json` and its retention/
  shuffled controls: rejected high-capacity candidate without rehearsal.
- `span8_reward_buffer_8192_width256_rehearsal_seed30981.json`,
  `span8_reward_buffer_8192_width256_rehearsal_retention_seed30981.json`,
  `span8_reward_buffer_8192_width256_rehearsal_shuffled_seed30982.json`,
  and the corresponding `seed30983` normal/retention reports: promoted
  replay-repair replication.
- `span8_reward_buffer_parent_comparison_seed30981.json` and
  `span8_reward_buffer_parent_comparison_seed30983.json`: paired retention
  comparisons against the inherited parent.
- `span8_adapter_buffer_collection_seed31331.json` and the ignored
  `artifacts/replay_buffers/span8_adapter_old_experience_seed31331.pt`:
  persisted old-experience bank.
- `span9_skill_slot_fresh_4096_seed31371.json`,
  `span9_fresh_retention_seed31371.json`,
  `span9_skill_slot_persistent_plasticity_4096_seed31371.json`, and
  `span9_persistent_retention_seed31371.json`: matched fresh versus
  protected-persistence frontier.
- `span9_*gate*`, `span9_*sourcegate*`, `span9_*logitprotected*`, and
  `span9_*intention*` reports: bounded routing and plasticity controls.
- `span8_workspace_buffer_collection_seed31571.json`,
  `span9_workspace_skill_fresh_4096_seed31581.json`,
  `span9_workspace_skill_protected_4096_seed31591.json`,
  `span9_workspace_skill_sourcegate_4096_seed31601.json`,
  `span9_workspace_skill_bottleneck64_fresh_4096_seed31611.json`,
  `span9_workspace_only_bottleneck64_fresh_4096_seed31621.json`,
  `span9_workspace_skill_pos3_fresh_4096_seed31651.json`, and
  `span9_workspace_skill_curriculum_4096_seed31661.json`, and
  `span9_workspace_joint_reader_4096_seed31671.json`: workspace routing,
  plasticity, reward-balance, gradual-rehearsal, and joint-reader controls.
- `span9_workspace_skill_pos3_retention_seed31651.json` and
  `span9_workspace_skill_pos3_shuffled_4096_seed31681.json`: promotion-grade
  retention and outcome-shuffled adversarial audits for the best workspace
  candidate.
- `span9_workspace_critic_fresh_4096_seed31811.json`,
  `span9_workspace_critic_weight4_fresh_4096_seed31821.json`,
  `span9_workspace_critic_protected_4096_seed31831.json`,
  `span9_workspace_usage_critic_fresh_4096_seed31911.json`,
  `span9_workspace_critic_retention_seed31811.json`, and
  `span9_workspace_usage_critic_retention_seed31911.json`: action-conditioned
  critic and usage-conditioned plasticity controls.
- `workspace_symmetry_diagnostic_seed31941.json`: evidence that the promoted
  RAM slots are exact content/usage clones until generic addresses are added.
- `span9_fixed_address_usage_critic_fresh_4096_seed31951.json` and
  `span9_trained_address_usage_critic_fresh_4096_seed31961.json`: fixed versus
  trained generic-address controls and their retention/usage measurements.
- `workspace_address_strength_sweep_seed32011.json`,
  `span8_address_scale1_buffer_collection_seed32041.json`,
  `span9_address_scale1_usage_critic_fresh_4096_seed32031.json`,
  `span9_address_scale1_usage_critic_protected_4096_seed32051.json`,
  `span9_address_scale1_usage_critic_retention_seed32031.json`, and
  `span9_address_scale1_usage_critic_protected_retention_seed32051.json`:
  address-strength and addressed-RAM protected-plasticity audits.
- `span9_address_scale1_writecontent_usage_critic_fresh_4096_seed32121.json`,
  `span9_address_scale1_content05_usage_critic_fresh_4096_seed32131.json`, and
  `span9_address_scale1_content05_retention_seed32131.json`: learned versus
  fixed address-conditioned write-content controls.
- `span9_workspace_routing_diagnostic_seed31631.json`: disposable probe
  results separating old/new routing information from correct-action decoding.
- `span9_address_scale1_staged_gate_4096_seed32151.json`: rejected staged
  gate; its provenance target was degenerate because refinement saw replay rows
  only.
- `span9_address_scale1_provenance_gate_4096_seed32161.json` and
  `span9_address_scale1_provenance_gate_4096_seed32162.json`: corrected
  mixed-source provenance-gate controls at 64 and 128 fresh epochs.
- `span9_address_scale1_provenance_gate032_4096_seed32172.json`: lighter
  provenance weight control; higher transfer but retention gate failure.
- `span9_address_scale1_joint_replay_4096_seed32171.json`: matched joint
  adapter/replay control; retention-safe promotion bar not met.
- `span9_address_scale1_preserve_gate_4096_seed32190.json`: rejected
  gate-score-preservation control; numerical fresh-gate magnitude was not a
  stable teacher.

The ignored local checkpoint hashes are:

```text
span8_smoke_inherited_seed30564.pt  sha256 9a429e1eea0b1c1f2e30c02ce9d91c4e32ceac257d8d55e3d37ae0e5384c4b7e
span8_smoke_fresh_seed30565.pt      sha256 12d4dc2b11fc126f9cc1b613622d5edcb228fdda9c87d1eb545895db5005e099
span8_shuffled_seed30566.pt         sha256 518b443d6e765d70bcc71c5873e63ab65cf41de68550c96b442726d3b83cc895
span8_replica_inherited_seed30567.pt sha256 db155aa0ead94f1e5be43f88434a03e18f7a61d744f5f723e05dcb46543b681a
span8_replica_fresh_seed30568.pt     sha256 29e8e00296897f44a96c47ae3e6902f053c814031b0e5cbfb90d7679c16e3827
span8_replica_shuffled_seed30569.pt  sha256 efb70095a5ade9c72d8b702dff0dfc0cb6ac1b3d2f35608109a7375387552e36
```
