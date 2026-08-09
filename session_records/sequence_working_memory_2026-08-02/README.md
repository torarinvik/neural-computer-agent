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

## Multi-skill routing diagnostic (2026-08-03)

The cold skill bank now has a diagnostic-only `SkillAddressSelector` and a
reward-only audit harness. On three controller-produced address families, the
normal selector, the reward-shuffled selector, and the fixed cosine baseline
all reached **100.00%**. Candidate-key permutation also reached **100.00%**,
and the frozen controller stayed bit-identical. The audit therefore rejected
the result: the static query/key geometry solves routing without using the
scalar outcomes, so it cannot support a reward-only learning claim. This is a
clean negative result. The next routing experiment must scramble candidate
addresses per episode (or otherwise break the static shortcut), then retest
normal, shuffled-reward, permutation, and frozen-controller controls before
any selector is allowed into promotion.

The follow-up removed the shortcut with fixed random opaque candidate
addresses. A three-row seed reached **100.00%** after 32,768 verifier bits,
while reward-shuffled and cosine controls were **33.33%**; candidate
permutation remained **100.00%** and the controller stayed frozen. A second
seed reached the same 100/33/100/33 pattern at 65,536 bits, and a four-row run
reached **100.00%** with both controls at **25.00%**. This is a genuine
reward-dependent routing diagnostic, but not yet a promoted memory-bank
capability: the next work is a multi-seed learning curve, more candidate
skills, and integration with real promotion plus behavior/retention audits.

The selector has now been exercised through the real bank with the archived
span-nine and span-ten artifacts. After save/reload, all 64 held-out queries
selected the correct row; direct and rehydrated behavior matched at 91.45%
and 86.64%; wrong-skill controls fell to 86.59% and 79.96%. Reward-shuffled
and cosine controls stayed at 50%, candidate permutation stayed at 100%, and
the controller remained frozen. This closes the plumbing proof, not the
promotion claim: multi-seed retention and online task-shift audits remain the
next gate before learned routing can replace the safe default.

An independent seed reproduced the full routing gates (16/16 per skill,
100% normal, 50% shuffled/cosine, 100% permutation) and direct-vs-routed
behavior equivalence at 93.49% for span nine and 87.66% for span ten. The
remaining promotion gate is repeated acquisition under task shift with
retention of the previously stored artifact behavior.

The online selector audit now has a replicated retention repair. New-skill-only
updates moved the new route to 100% but erased the old route (0%). Adding a
small output-distillation loss on replayed old queries retained the old at
100% and learned the new at 100% and 95.31% across two seeds. Shuffled rewards
left the new route at chance, and candidate permutation remained exact. The
next experiment is the same update applied to real bank artifacts across
multiple task shifts, with behavior and artifact-retention gates.

The same method now passes through a real disk-backed replacement: after
reloading the bank, the distilled selector routed the old row and newly
replaced row correctly, while direct and routed behavior matched at 92.45%
and 88.05%. New-only updates erased the old route; distilled replay retained
it; shuffled rewards failed the new route. The next step is to extend this
from one two-skill replacement to repeated task shifts and a longer retention
window before changing the production default.

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
controls. A fixed-age label is not needed: the row-local scalar is updated only
by verified outcomes.

As a logistics/replication check, the existing reward-only volatility selector
was run on the four-GPU Vast instance with four fresh seeds (32231--32234),
using the exact previously successful 32-update, learning-rate-1.0 recipe.
All four passed every pre-registered gate: held-out accuracy 96.1--96.9%,
valid replacement 98.8--99.6%, outcome-shuffle and reversed-history causality,
old-utility retention, and the under-five-minute cap. This confirms that the
external-memory habit mechanism is reproducible on CUDA hardware.

The missing online path has now been closed by a separate controller-to-disk
probe. A frozen persistent-memory controller emitted its own latent
``memory_key``, ``memory_value``, and ``memory_write_strength`` tensors; those
rows were committed to real ``DiskLatentMemory`` stores, queried through
physical receipts, and updated only from the verifier's attempted-action
outcome. The probe included no task labels or semantic row IDs in the learner
path. On four RTX 5090 seeds (32231--32234), each run used 512 controller
writes, 1,536 physical reads, and 4,096 receipt-attributed outcome updates in
1.54--1.71 seconds. Every run passed the pre-registered controls:

- physical normal accuracy: 98.0--99.2%;
- no-memory control: 49.8--50.0%;
- shuffled-value control: 46.3--57.8%;
- successful-row volatility: about 0.1665 versus 1.0 for failed rows;
- shuffled receipts remove the protection gap (the gap becomes negative);
- high-volatility rows accept a larger elastic rewrite than low-volatility
  rows; and
- serialized disk round-trips are exact.

This is the first end-to-end evidence that controller-created writes, causal
outcome receipts, and the volatility/habit scalar compose at the actual
physical memory boundary. It is a plumbing and causal-memory breakthrough,
not yet evidence that the controller has learned a new write policy or that
span nine has crossed its 90% acquisition bar.

### Bounded task-shift acquisition

The first bounded-capacity task-shift rung now passes. Each bank began with
four controller-created rows. The verifier selected an actual failed decoy
transform without exposing its label to the memory mechanism, replayed eight
receipt-attributed outcome rounds, and then introduced one new
controller-produced row. The high-volatility policy replaced the failed row;
least-volatility and shuffled-volatility policies were registered controls.

Four independent CUDA runs (seeds 32331--32334, 128 banks each) passed every
gate in 2.70--3.31 seconds:

- the habit policy selected the failed decoy in 96.9--100.0% of banks;
- the new row was acquired at 94.5--98.4% accuracy;
- old-row accuracy after replacement was 86.7--88.7%;
- the composite old-plus-new score was 88.8--90.6%, versus 69.1--70.0% for
  least-volatility replacement;
- shuffled volatility stayed near the failed control at 70.0--72.3%; and
- every policy remained exactly bounded at four rows with exact disk
  save/reload.

This is the first causal evidence that the volatility scalar is not merely a
passive statistic: in a real physical task shift it protects stable knowledge
by admitting a new row into the slot made disposable by verified failure. The
rung is still a frozen-policy control. The controller has not yet learned the
volatility-to-replacement mapping from reward, and the candidate query is an
adjacent same-event acquisition to avoid conflating this memory test with the
harder cross-context key-generalization problem.

### Reward-trained row-local gate

The frozen control was followed by the smallest trainable version: the parent
controller was expanded with one zero-initialized generic residual coefficient
for row volatility. The controller body, memory writes, receipt attribution,
and disk stores stayed frozen. Each update compared ``alpha + 8`` against
``alpha - 8`` on the physical old-plus-new verifier score, then moved the one
coefficient by two in the winning direction. No task, row, or semantic labels
were supplied to training. The larger finite-difference probe was necessary:
the earlier ±1 probe did not cross an action boundary, so both arms selected
the same rows and correctly produced no learning signal.

Four independent RTX 5090 runs (seeds 18401--18404, eight updates, 8 training
banks and 32 held-out banks) all passed the gates in under five minutes. The
learned coefficient reached +16 in every normal run. Adjacent task-shift
results were:

| Metric | Range across four normal runs |
| --- | ---: |
| new-row accuracy | 90.6--100.0% |
| old-row accuracy | 86.7--90.6% |
| learned composite | 87.5--90.6% |
| reset composite | 73.1--78.8% |
| shuffled-receipt composite | 67.5--73.1% |

The learned gate therefore beats its zero-volatility reset by 11.9--17.5
points while remaining bounded, disk-round-trip exact, and preserving the
binary and four-rule retention audits. Two adversarial control families were
also run with four seeds each. Shuffling the plus/minus rewards produced
alphas in ``{-4, 0, +4}`` and every run failed the promotion gates; shuffling
the receipt-to-row attribution produced alphas in ``{-4, 0, +12}`` and every
run also failed. These controls rule out a fixed positive coefficient or an
unattributed reward shortcut as the explanation.

This is the first evidence that the controller can learn a memory-management
habit from physical verifier reward, rather than merely executing a frozen
volatility policy. It is still an adjacent same-event acquisition: held-out
cross-context candidate generalization is the next rung, followed by span-nine
acquisition with the promoted span-eight parent.

### Held-out candidate-context query

The held-out rung now uses the candidate's own new context but a different
query event, so it tests content-addressed reuse rather than exact-frame
matching. Four fresh CUDA seeds (18701--18704) passed the expanded gates:

- cross-context new-row accuracy: **90.6--96.9%**;
- learned cross-context composite: **86.9--89.4%**;
- zero-volatility reset composite: **71.9--81.9%**; and
- receipt-shuffled cross-context composite: **66.9--72.5%**.

Every normal run preserved the adjacent-task gates, bounded four-row disk
capacity, exact serialization, and the binary/four-rule retention audits.
The reward-shuffled controls (18801--18804) all failed the expanded gates and
never reached 85% cross-context acquisition. Receipt-shuffled controls were
intentionally retained as a mixed result: two of four happened to pass the
small behavioral gate, so receipt shuffling is not claimed as a definitive
negative control for this training setup. The clean reward-shuffle result and
the normal four-seed replication are the promotion evidence.

This is the next memory breakthrough: a one-coefficient gate learned from
physical reward continues to protect old rows while acquiring a new row for a
non-adjacent query in the new task context. It is still a memory-management
result, not span-nine mastery.

### Tie-safe GPU replication

The first finite-difference updater treated tied verifier scores as a positive
vote. That was an invalid shortcut: a discrete argmax plateau could drive the
volatility coefficient upward even when rewards were shuffled. The updater now
leaves the coefficient unchanged on a tie, and uses a larger
`learning_delta=16` so the two arms cross an actual replacement boundary. A
regression test asserts that ties are neutral.

With this corrected updater, four independent RTX 5090 runs (seeds
19701--19704; 16 updates, 16 training banks, 64 held-out banks) all passed:

| Metric | Range across normal runs |
| --- | ---: |
| learned volatility coefficient | +16 in all runs |
| adjacent new-row accuracy | 95.3--100.0% |
| adjacent old-row accuracy | 84.4--85.5% |
| adjacent learned composite | 86.9--88.1% |
| cross-context new-row accuracy | 95.3--98.4% |
| cross-context learned composite | 89.1--90.0% |

All four retained the binary/four-rule audits, bounded disk capacity, and
exact save/reload. Four reward-shuffled controls (19801--19804) ended at
coefficients 0, -10, +6, and 0 and all failed the gates; four
receipt-shuffled controls (19901--19904) ended at -1, 0, -3, and -2 and all
failed. This is the clean promotion evidence for reward-trained habit
selection; the earlier tie-biased pilot is retained only as a diagnostic
lesson, not as evidence.

## Span-nine acquisition with event-age routing and protected rehearsal

The zero-initialized workspace-volatility diagnostic did not learn a causal
habit mechanism: four truthful and four outcome-shuffled runs converged to the
same negative scale and the same accuracy. The useful representation change
was instead a generic normalized event-age trace, exposed only to the new
successor slot. It carries stream position, not a task, span, operation, or
answer label. At 1,024 new span-nine lifetimes it reached 82.66--83.40% across
four seeds, versus 76.50--78.29% for the matched replay/rehearsal smoke. At
4,096 lifetimes it reached 88.54--89.12%.

The promotion run started from the span-eight addressed parent and combined
the age trace with a capped old-span replay bank and small replay residual/logit
penalties. It used 8,192 unique span-nine lifetimes, 16,384 replay transitions,
and 384 optimizer passes. The selected checkpoint is:

`artifacts/checkpoints/span9_age_replay_pen003_e384_seed48001.pt`

Its SHA-256 is
`0c40c7f478d14234ae29108ef6236c50b6b5c73448b596d47910646827c9db1d`.

On a fresh 4,096-lifetime audit it reached **90.46%** overall and 90.34% on
the reversed operation. Blank sequence and complete memory reset were 49.64%
and 49.42%; reverse-operation prediction flips were 47.96% on non-palindromes.
A paired 2,048-lifetime audit against the span-eight parent retained every old
span, with worst margin **-1.01 points**, while span nine improved from 75.46%
to **90.19%**. Both exact outcome-shuffled controls removed the gain, scoring
52.44% and 44.79%, with all-memory-reset controls at chance. These are the
promotion-grade files:

- `span9_age_replay_pen003_e384_seed48001.json`
- `span9_age_replay_pen003_e384_highcount_seed48001.json`
- `span9_age_replay_pen003_e384_retention_seed48001.json`
- `span9_age_replay_pen003_e384_shuffle_seed48101.json`
- `span9_age_replay_pen003_e384_shuffle_seed48102.json`

This is the first span-nine result to pass mastery, memory-dependence,
causal-reward, and old-skill-retention gates together. It is evidence for
sample-efficient compounding: a task-agnostic stream clock helps the new
primitive, while persistent replay and small write/logit penalties protect the
older skills. The next frontier is an independent-seed replication followed by
private consolidation and recall; no claim is made here that the current
checkpoint has yet completed that long-term-memory integration.

## Independent span-nine replication

The same recipe was rerun from the same span-eight parent with seed `48002`:
8,192 new lifetimes, the same 16,384-transition replay cap, the same `0.003`
replay gate/logit penalties, and 384 optimizer passes. The 2,048-lifetime
training report reached 90.69%. A separate 4,096-lifetime audit reached
**90.58%**, with 90.85% reversed-operation accuracy, 48.01% non-palindrome
operation flips, 49.82% blank-sequence accuracy, and 50.00% complete-reset
accuracy.

Checkpoint SHA-256:
`0468508244e3574cec8bb937aad37bb6b34d140a5401e5084fc5d732efe05695`.

The paired 2,048-lifetime retention audit reached **90.49%** on span nine and
retained every old span, with a worst old margin of **-1.21 points**. This
independent seed therefore passes the same mastery, causal, memory-dependence,
and retention gates. Two exact matched outcome-shuffled controls reached only
53.78% and 55.04%; their blank/reset controls stayed at chance and neither
reproduced the normal causal gain. It promotes event-age routing plus protected
replay from a single-seed result to a replicated compounding result. The
remaining frontier is persistence across private consolidation and disk reload.

Replication artifacts:

- `span9_age_replay_pen003_e384_seed48002.json`
- `span9_age_replay_pen003_e384_highcount_seed48002.json`
- `span9_age_replay_pen003_e384_retention_seed48002.json`
- `span9_age_replay_pen003_e384_shuffle_seed48103.json`
- `span9_age_replay_pen003_e384_shuffle_seed48104.json`
- `artifacts/checkpoints/span9_age_replay_pen003_e384_seed48002.pt`

## Span-nine skill memory survives external serialization

The learned successor-slot parameters were extracted into the separate
artifact `artifacts/memory/span9_skill_memory_seed48002.pt`, leaving the
span-eight parent as the frozen computation core. The artifact contains
253,081 learned parameters and no verifier labels or correct-action fields.
After saving and reloading that artifact in a fresh model instance, the
rehydrated controller exactly matched the direct child at **90.96%** on a
4,096-lifetime audit: reverse-operation accuracy was 90.63%, blank accuracy
49.89%, complete-reset accuracy 49.76%, and non-palindrome operation flips
48.11%.

The causal corruption control zeroed only the reloaded successor-slot state.
The parent core and verifier inputs were unchanged, but accuracy fell to
**75.64%** (reverse-operation 75.39%); blank and reset stayed at chance. This
shows that the new capability is carried by the external skill artifact, not
by reward-independent drift in the frozen controller. Artifact SHA-256:
`228341bd120757bb4ad287530f11f36773788ab44098ec837e89a8c6d25d8a04`.

Evidence:
`span9_skill_memory_audit_seed48002.json`.

## Span-nine hot/cold skill-bank routing

The external artifact now lives behind a bounded hot/cold bank. Cold rows store
only controller-produced context keys and opaque artifact paths; a
content-addressed query promotes one artifact into the fast process-local
cache. The real row was selected before and after reloading the bank with
confidence 0.99994, and the cold row tensors reloaded exactly.

The promoted skill reached **90.76%** on a 4,096-lifetime audit (reverse
operation 90.66%, blank 50.05%, complete reset 50.02%, operation flips
48.19%). Evicting the hot artifact while leaving the cold disk bank intact
reduced accuracy to **75.68%**. Promoting a physically valid zeroed decoy also
gave 75.68%. The bank therefore controls fast-memory availability causally,
not merely as a bookkeeping layer.

The saved bank is
`artifacts/memory/span9_skill_bank_seed48002/`; evidence is
`span9_skill_bank_audit_seed48002.json`.

## Next frontier

Multi-skill addressing now passes after cold reload: controller-produced
context keys route span nine and span ten artifacts to the correct row, and
the routed span-ten result matches direct rehydration. The bank also records
artifact hashes and can abstain when confidence or top-row margin is too low,
so an ambiguous query no longer has to activate an arbitrary skill. The next
scientific rung is to learn a task-conditional selector across more than two
skills, then connect that selection to behavior on the eighth-back reader
without sacrificing the protected ladder. Keep the two-point old-skill
retention gate and the outcome-shuffle, blank, reset, reversal, and
wrong-skill controls.

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
- `span9_age_replay_pen003_e384_seed48001.json` and
  `span9_age_replay_pen003_e384_highcount_seed48001.json`: selected span-nine
  training report and independent 4,096-lifetime mastery audit.
- `span9_age_replay_pen003_e384_retention_seed48001.json`: paired 2,048-lifetime
  spans-2--9 retention audit against the span-eight parent.
- `span9_age_replay_pen003_e384_shuffle_seed48101.json` and
  `span9_age_replay_pen003_e384_shuffle_seed48102.json`: exact matched
  outcome-shuffled controls.
- `span9_age_replay_pen003_e384_seed48002.json`,
  `span9_age_replay_pen003_e384_highcount_seed48002.json`, and
  `span9_age_replay_pen003_e384_retention_seed48002.json`: independent-seed
  replication and paired retention evidence.
- `span9_age_replay_pen003_e384_shuffle_seed48103.json` and
  `span9_age_replay_pen003_e384_shuffle_seed48104.json`: independent-seed
  matched outcome-shuffled controls.
- `span9_skill_memory_audit_seed48002.json`: external successor-slot
  serialization, fresh-instance rehydration, and zeroed-artifact corruption
  audit.
- `span9_skill_bank_audit_seed48002.json`: bounded hot/cold row routing,
  cold-bank reload, hot eviction, and decoy-artifact controls.
- `artifacts/memory/span9_skill_bank_seed48002/`: committed cold rows,
  manifest, real skill artifact, and zeroed decoy artifact used by the audit.
- `span10_cold_rehydrated_8192_e384_seed9954815.json` and
  `span10_child_highcount_9954815.json`: corrected appended-slot acquisition
  after cold reload; the new span-ten skill reaches about 83% but the
  always-on residual is not retention-safe by itself.
- `span10_retention_9954815.json`: paired audit showing why the always-on
  residual is rejected (span-nine margin −4.28 points).
- `span_multi_skill_bank_audit_seed49011.json`: promotion-grade two-row
  external-bank routing. After save/reload, all eight controller-produced
  context queries select the correct span-nine or span-ten artifact; routed
  span nine is 90.997%, routed span ten is 83.225%, and direct rehydration
  matches exactly. Wrong-skill activation is the causal control.
- `controller_habit_gpu_seed32231.json` through
  `controller_habit_gpu_seed32234.json`: four CUDA replication reports for
  the reward-only row-volatility selector; all passed the full gate suite.
- `controller_online_disk_habit_gpu_seed32231.json` through
  `controller_online_disk_habit_gpu_seed32234.json`: four CUDA reports for
  the controller-output-to-physical-disk receipt probe; all passed the online
  memory, volatility, replacement, shuffle, and save/reload gates.
- `controller_online_task_shift_gpu_seed32331.json` through
  `controller_online_task_shift_gpu_seed32334.json`: four CUDA reports for
  bounded task-shift acquisition; all passed the decoy-selection, new-row,
  old-retention, shuffle, capacity, and save/reload gates.
- `task_shift_gate/task_gate_seed18401.json` through
  `task_gate_seed18404.json`: four CUDA reports for the reward-trained
  zero-initialized row-local volatility gate; all passed.
- `task_shift_gate/task_gate_reward_shuffle_seed18501.json` through
  `...18504.json`: shuffled-reward controls; all rejected.
- `task_shift_gate/task_gate_receipt_shuffle_seed18601.json` through
  `...18604.json`: shuffled-receipt controls; all rejected.
- `task_shift_gate_cross/task_gate_cross_seed18701.json` through
  `...18704.json`: four held-out candidate-context query runs; all passed.
- `task_shift_gate_cross/task_gate_cross_reward_shuffle_seed18801.json`
  through `...18804.json`: reward-shuffled cross-context controls; all
  rejected.
- `task_shift_gate_cross/task_gate_cross_receipt_shuffle_seed18901.json`
  through `...18904.json`: receipt-shuffled cross-context controls; mixed and
  explicitly not treated as a definitive negative control.
- `controller_online_task_shift_gate_delta16_seed19701.json` through
  `...19704.json`: tie-safe four-seed GPU replication of reward-trained
  row-local habit selection.
- `controller_online_task_shift_gate_reward_shuffle_delta16_seed19801.json`
  through `...19804.json`: reward-shuffled adversarial controls; all rejected.
- `controller_online_task_shift_gate_receipt_shuffle_delta16_seed19901.json`
  through `...19904.json`: receipt-shuffled adversarial controls; all rejected.

The ignored local checkpoint hashes are:

```text
span8_smoke_inherited_seed30564.pt  sha256 9a429e1eea0b1c1f2e30c02ce9d91c4e32ceac257d8d55e3d37ae0e5384c4b7e
span8_smoke_fresh_seed30565.pt      sha256 12d4dc2b11fc126f9cc1b613622d5edcb228fdda9c87d1eb545895db5005e099
span8_shuffled_seed30566.pt         sha256 518b443d6e765d70bcc71c5873e63ab65cf41de68550c96b442726d3b83cc895
span8_replica_inherited_seed30567.pt sha256 db155aa0ead94f1e5be43f88434a03e18f7a61d744f5f723e05dcb46543b681a
span8_replica_fresh_seed30568.pt     sha256 29e8e00296897f44a96c47ae3e6902f053c814031b0e5cbfb90d7679c16e3827
span8_replica_shuffled_seed30569.pt  sha256 efb70095a5ade9c72d8b702dff0dfc0cb6ac1b3d2f35608109a7375387552e36
```

## Three-row replay rebuild diagnostic (2026-08-03)

The selector was next tested on three sequential skill families (spans 9, 10,
and 11) behind fixed random opaque row keys. The controller was frozen. The
incremental arm used scalar-outcome updates plus output-distilled replay; the
rebuild arm discarded only selector weights and relearned from the accumulated
opaque query/attempt/outcome replay. No span identity or correct row was
exposed to either learner.

At 1,024 updates per arm and 65,536 verifier bits, the incremental arm scored
`[1.00, 1.00, 1.00]` after the third shift and the replay-rebuild arm scored
`[1.00, 1.00, 1.00]`. The rebuild arm already mastered the first two rows
before adding the third. The independent-random-outcome null scored
`[0.00, 1.00, 0.00]` (33.3% aggregate), and reversing the physical candidate
row order left the rebuild scores at `[1.00, 1.00, 1.00]`. The controller
weights were bit-identical before and after the audit.

This is a diagnostic of external replay supporting selector reconstruction,
not a claim that rebuilding dominates online updates: a 512-update pilot was
seed-sensitive and failed on the middle row. The report is
`skill_bank_router_rebuild_seed93401.json`. The next experiment must repeat
the three-row result across seeds using real disk-backed artifacts, then test
longer replacement sequences and behavioral/retention gates before learned
routing can replace the safe cosine default.

## Successor-slot extension probe: span eleven (2026-08-03)

The next bridge toward a real three-artifact bank appends one zero-output
successor slot to the cumulative span-ten checkpoint and trains only that new
slot. The insertion check was exact: logits and workspace state matched the
parent before any update. Older spans were interleaved as verifier-generated
rehearsal, and a zeroed-new-slot replay was used as the causal control.

| Arm | Fresh verifier bits | Span-11 parent → child | Zeroed-slot span 11 | Retention Δ span 9 / 10 |
| --- | ---: | ---: | ---: | ---: |
| 16 updates, no distillation | 2,576 | 81.68% → 83.52% | 81.68% | −1.56 / +0.31 pp |
| 32 updates, no distillation | 10,240 | 80.54% → 81.39% | 80.54% | −1.09 / +0.12 pp |
| 8× batch reuse | 2,560 | 81.11% → 81.61% | 81.11% | −2.56 / −1.48 pp |
| parent-logit distillation | 1,280–2,576 | unchanged | unchanged | 0.00 / 0.00 pp |

The apparent gains were small and the zeroed controls returned to baseline;
none crossed the 5-point causal gain bar. Reusing tiny batches also violated
the two-point retention gate, while parent-logit distillation protected old
behavior but did not learn the new span at this budget. These are bounded
negative results, not evidence that the slot architecture is impossible. The
reports are `span11_slot_extension_*.json`; the next fork is a
retention-constrained frontier curriculum using fresh target batches.

An activity probe found that the unregularized appended slot opened broadly
on spans 9, 10, and 11. Adding a task-agnostic penalty on its residual norm
for old-stream rehearsal reduced that leakage and kept retention within the
gate, but two short penalty runs reached only **+0.36 points** on span 11 and
the zeroed-slot controls stayed at baseline. The branch is paused with a
well-defined negative: the interface is safe, but locality regularization
alone is not enough to learn the next skill at this budget.

Giving the new slot a generic read of only the immediately preceding slot was
the next architectural fork. It preserved insertion and the two-point
retention gate, but 16 and 32 fresh updates reached only **+0.85/+1.92
points**, with zeroed-slot controls still at baseline. The prior-slot
interface is now tested and safe, but it has not yet produced causal
sample-efficient span-eleven learning.

A gradual target curriculum (zero distractors/fixed positions, then a small
position-and-distractor ramp) preserved retention but produced only **+0.07
points** at 10,240 fresh verifier bits. The span-eleven branch is paused here:
the short-run evidence points to a deeper credit-assignment or representation
bottleneck rather than a reason to keep extending the same recipe.

## Decisive successor-input probe (2026-08-03)

Before changing the controller again, a frozen diagnostic extracted the raw
input to the appended slot's first linear layer on lifetime-disjoint span-11
episodes. Throwaway probes decoded the correct action at **84.66% linear** and
**87.71% with a small MLP**; an independent random-label null stayed at
**50.57%**, with a second prior-slot variant at **87.43% / 48.15%**. The
controller and slot weights were unchanged and the diagnostic heads were
discarded.

The current failure is therefore localized to credit assignment: the relevant
next-action information is already present at the new slot's input, but short
reward-only updates do not discover/use it. The next branch should use denser
action-conditioned processing of the same verified outcome, while retaining
the two-point old-skill gate and all causal controls. Reports are
`span11_slot_input_probe_*.json`.

A temporary action-conditioned success critic (trained only from the attempted
action's scalar outcome, then discarded) was tested at weights 0.5 and 5.0.
Both preserved old skills, but produced only **+0.99/+0.71 points** with no
zeroed-slot causal separation. This is a bounded negative for the simplest
critic auxiliary; the representation probe says the next useful change must
improve how existing information drives the slot/action update. The two
unpromoted critic checkpoints are retained under
`artifacts/checkpoints/span11_slot_extension_critic*.pt`.

## Successor replay-credit sweep (2026-08-03)

The successor branch was moved to a frozen latent replay buffer containing
only controller-visible features, opaque attempted actions, and scalar
outcomes. The collector now automatically caches inherited workspace, usage,
event-snapshot, and age reads; targeted tests cover the new event-snapshot
path. Detached critics, a binary-complement bandit control, and a nonlinear
gate were evaluated with shuffled-outcome and zeroed-slot controls.

The strongest safe arm (seed 93712) reached a **0.89-point causal span-11
gain**, with span-9/span-10 retention changes of **−1.04/−0.43 points**. It
remains below the pre-registered 5-point causal bar. More unique data,
on-policy replay, higher learning rate, and longer reuse did not improve the
bound; restoring distractors after an easy arm also removed its gain. The
full sweep and exact commands are in
`span11_replay_credit_assignment_2026-08-03.md`.

This closes the current critic/gate knob fork. Since the independent frozen
input probe decoded action at **84.66% linear / 87.71% MLP**, the remaining
problem is reward-to-output credit and difficulty, not sensory representation.
The next branch should add a smaller intermediate primitive or an explicit
per-output curriculum.

## Outcome-only position weighting rejection (2026-08-03)

Position-difficulty weighting produced a tempting **+2.38-point** causal
screen, but it opened a missing-evidence shortcut: blank accuracy fell to
**34.87%**. The corrected hard gate rejected it, and the checkpoint was not
curated. Its accounting and receipts are in
`position_weighted_rejection_2026-08-03/`.

## Target-diversity and successor-slot screen rejections (2026-08-03)

Suffix-only credit, a matched full-window continuation, position rerender
augmentation, and a fresh appended slot all failed the first acquisition gate
against the accepted missing-evidence parent. Their blank/reset controls were
safe, but none improved held-out target behavior; the tempting positive
zeroed-slot gaps therefore do not count. Receipts and accounting are in
`target_diversity_rejections_2026-08-03/`.

## Missing-evidence population breakthrough (2026-08-03)

A four-arm population race using the accepted protected missing-evidence
recipe passed a common 1,024-lifetime audit at **+2.04–+2.33 points** for
every arm. The selected seed held **+2.02 points** at 4,096 lifetimes with a
95% interval of **[+1.71, +2.32]**; an independent second arm also passed at
**+1.56 points**. Blank/reset controls stayed at chance and old-span
retention stayed within tolerance. The selected checkpoint is curated and
the full accounting is in `missing_evidence_population_2026-08-03/`.

The intervening event-age, workspace-context, action-adapter, critic-policy,
and larger rehydration tests all failed acquisition or retention and remain
unpromoted. Their exact negative evidence is in
`credit_mechanism_rejections_2026-08-03/`.

## Missing-evidence rehearsal frontier (2026-08-03)

Protected blank/missing-evidence rehearsal produced the next control-safe
frontier. Two independent 512-target runs passed the corrected 1,024-lifetime
acquisition, causal-slot, retention, blank, and reset gates at **+2.43/+2.65
points**. The stronger seed's 4,096-lifetime audit remained accepted at
**+1.77 points**, with a 95% paired-lifetime interval of **[+1.48, +2.05]**;
blank/reset controls stayed at **49.76%/49.67%**, and old-span retention was
within the two-point gate. The missing-evidence controls are charged to the
budget, so the audit objective still has no replay-savings credit and is not
an autonomous-stop or mastery claim. Full receipts and accounting are in
`missing_evidence_rehearsal_frontier_2026-08-03/`.

## Protected-plasticity successor rejection (2026-08-04)

Aggregate rehearsal-gradient projection produced a real causal complement
signal, reaching **+10.06 points** at one seed with old-span changes of
−1.80/−1.24 points. The independent seed reached +9.01 points but lost
−4.50/−3.47 points on old spans. A verifier-visible behavioral-retention
rollback narrowed the loss to −2.53/−1.78 while retaining +6.86 points, but
still missed the span-nine gate. The branch is rejected before promotion; the
remaining problem is context-selective plasticity, not more aggregate replay.

The full accounting and the explicit complement-audit correction are in
`protected_plasticity_frontier_2026-08-04/`.

## No-replay span-retention breakthrough (2026-08-04)

The direct shared-parameter curriculum reached span 4 but forgot earlier
spans: span 2 fell by **12.5 points** and span 3 by **8.3 points**. A summed
append-only growth bank reduced some interference but still failed the strict
retention gate. Both mechanisms were recorded as rejected controls.

The successful design treats acquired capability as an external executable
file. A frozen span-2 parent receives independently learned span-3 and span-4
artifacts; a memory-side router selects one artifact from opaque controller
context and event-window occupancy. Two prefix-audited seeds passed with
**100% route accuracy**, **100% span-2 and span-3 accuracy**, and span-4 at
**87.5%/90.625%**. Span 4 remained above the 80% threshold from **12,288
stable verifier bits** onward in both runs. Blank controls were **50.0–51.2%**,
the parent core was bit-identical, and replayed examples were **0**.

This is a promoted narrow no-replay retention mechanism, not a claim that
shared-weight SGD has solved catastrophic forgetting. The retained state is
isolated, versioned capability memory, which is consistent with the intended
CPU/filesystem architecture. Full reports, ledgers, and artifacts are in
`canonical_no_replay_artifact_bank_512_2026-08-04/` and its replication
directory.

## Learned opaque address discovery (2026-08-04)

The occupancy-derived route was then replaced with a genuinely learned,
permutation-equivariant address path. Candidate artifact keys are independent
random opaque vectors; a factorized query/key router learns which key to select
from controller hidden state and fresh scalar attempted-row outcomes only. It
does not receive occupancy, span labels, candidate coordinates, or the correct
row. Each route example is generated and consumed once, with no route replay.

Two independent accepted seeds reached **100% route accuracy**, **100%
candidate-permutation accuracy**, and **33.3% reward-shuffled accuracy** (the
three-way chance baseline). Selected behavior retained spans 2/3 at
**100%/98.96%** and **100%/97.92%**, with span 4 at **86.72%/87.11%**. Stable
span-4 acquisition began after **16,384/12,288 verifier bits**. The route used
65,536 unique logical lifetimes and verifier bits per accepted run, with zero
replayed route examples. A same-seed lower-capacity replication reached the
route gates but failed the strict stable span-4 gate, so it remains rejected.

This is evidence for learned opaque address discovery in an isolated capability
memory, not yet evidence that arbitrary new computation can be learned by a
frozen controller. Full reports and ledgers are in the two
`canonical_no_replay_learned_route_factorized_2048*` accepted directories;
failed-capacity records are retained beside them.

## Counterfactual credit for opaque route decisions (2026-08-04)

The first general credit-assignment step replaces independent noisy route
labels with common-random counterfactual pairs. Each pair shares one fresh
controller query but attempts two distinct opaque artifact rows; only the two
scalar verifier outcomes train the score preference. Binary intervention
decisions use the same paired difference through a policy-gradient loss. No
correct row, semantic label, or counterfactual metadata enters the deployed
controller.

With the parent and capability budget held at the accepted 512-update rung,
route accuracy rose from **70.8%** at 512 paired updates to **83.3%** at 1,024.
At 2,048 paired updates, two independent seeds passed: **100% route**, **100%
candidate permutation**, and **33.3%/35.9% shuffled outcome** controls.
Selected execution retained spans 2/3 at **100%/95.3%** and **100%/97.9%**,
with span 4 at **88.7%/87.1%**. Each run used **0 replayed examples**,
**65,536 counterfactual pairs**, and **131,072 route verifier bits**. The
lower-update rungs are recorded as rejected for insufficient route acquisition.

This is a promoted narrow credit-assignment mechanism for memory-side route
decisions. It does not yet solve credit assignment when several writes,
actions, and newly learned computations interact over long horizons. Full
reports and ledgers are in the two
`canonical_counterfactual_route_credit_logistic_2048*` directories.

## Cloud archive

The current Vast instance was quiescent when archived. The load-bearing
parent, span-nine, corrected span-ten, and independent span-ten checkpoints
are now stored locally under `artifacts/checkpoints/`. The promotion-grade
two-row cold skill bank is stored under
`artifacts/memory/span_multi_skill_bank_seed49011/`. Their source paths and
SHA-256 hashes are recorded in `remote_artifact_manifest.json`; no temporary
`/tmp` files on Vast are required to resume the latest audit.

The bank manifest now records a SHA-256 for every opaque skill artifact.
Promotion verifies that hash before loading from disk, and a tampered-file
regression test rejects modified cold artifacts while legacy manifests remain
loadable and are upgraded on their next save.

## Three-factor parent credit boundary (2026-08-04)

The three-factor intervention separately credits probe action, write/skip, and
recall action. The recall factor retains a differentiable memory-value path
while detaching the forced write-gate gradient; the reward-shuffled control
stays at chance. Two seeds replicate strong parent acquisition and unseen-token
retention (`0.980/0.744` and `0.992/0.734`), but both retain a target-last
shortcut (`~0.99`) and fail target-first (`~0.50`). No checkpoint is promoted.

Parent-protected retention controls preserve old skills but fail to learn
target-conditioned write selectivity; the unprotected control forgets the
parent. The remaining bottleneck is counterfactual utility for overwriting an
already retained target, not generic memory addressing or reward plumbing.
Receipts and sample-efficiency ledgers are in the
`counterfactual_three_factor_value_gradient_*` and
`counterfactual_three_factor_retention_v2_*` directories.

## External memory-side overwrite credit (2026-08-04)

The protected-controller experiment isolated the next missing mechanism: a
memory-side writer must learn when a write preserves useful state and when it
overwrites it. `external_overwrite_v2` freezes the controller and trains only
an independently versioned writer using three outcome-only counterfactual
factors. The second factor was corrected to always mean target-then-true-
distractor; the third factor supplies the complementary distractor-then-target
case.

An unbounded residual failed at 64 updates by reverting to the last-write
shortcut. The accepted v10 writer uses a frozen generic relevance prior plus a
bounded `tanh` residual. On the 64-update rung, seed 17 scored `0.949` target
first and `0.971` target last, with `0.965` intact recall and `0.973` mastered
retention. Seed 69415 replicated `0.983`/`0.982`, `0.977` intact, and `0.965`
mastered retention. The reward-shuffled control stayed at `0.483` intact and
`0.508`/`0.499` target-first/last, with parent retention at `0.453` and no
stable qualification. Every arm used zero replayed examples.

This is a promoted narrow credit-assignment breakthrough for an isolated
memory growth component. It is not yet general continual learning: the
controller remains frozen, the verifier is a two-slot synthetic pressure test,
and transfer to larger banks, persistent reload, and genuinely new learned
write utilities remain open. Full accounting is in the three v10 report
directories and their `sample_efficiency_ledger.json` files.

## Stable controller-native value path for three-slot banks (2026-08-05)

The writer-only larger-bank rung exposed a representation bottleneck: its
write probabilities correctly selected the target in both orders, but a value
produced after a longer distractor prefix was not decoded reliably after a
reset. A zero-initialized controller-native value path keyed by the current
learned event and opaque feedback fixes that context dependence. The controller
learns it during parent acquisition and is frozen during retention; the
external writer remains the isolated growing component.

Seed 17 passed the three-slot/one-row/64-update gate with target-first/last
`0.963/0.940`, intact `0.947`, mastered retention `0.980`, unseen minimum
`0.945`, stable bits `20,480`, and zero replay. Seed 69415 passed after the
smallest tested phase-1 extension to 800 steps with `0.986/0.991`, intact
`0.989`, mastered retention `0.977`, unseen minimum `0.961`, and stable bits
`35,840`. The original 704-step seed-69415 run was blocked by the parent
stability gate; it is recorded as a curriculum-budget rejection. The
reward-shuffled control stayed at chance and failed all capability gates. The
same persistent backend reloaded at `0.965`/`0.996`, rejected checksum
corruption in both seeds, and recovered at `0.938`/`1.000`.

This is a promoted narrow three-slot retention mechanism, not general
continual learning, arbitrary new computation, or persistent-memory transfer.
The reports and sample-efficiency ledgers are in the six one-row
`external_write_stable_value_v14*` directories.

The same mechanism also passed the two-row three-slot bank: seed 17 retained
the one-row metrics and reloaded at `0.965`, while seed 69415 reloaded at
`0.996`; both rejected checksum corruption and recovered at `0.938`/`1.000`.
Those reports are in the two `external_write_stable_value_v14_two_row*`
directories.

## Four-slot temporal-bank replication (2026-08-05)

The same stable controller-native value path and isolated external writer passed
a four-slot/two-row bank. Seed 17 reached target-first/last `0.981/0.982`,
intact `0.986`, mastered retention `0.992`, and unseen minimum `0.980`. Seed
69415 replicated `0.982/0.970`, intact `0.983`, mastered retention `0.977`,
and unseen minimum `0.973`; persistent reload was `0.988`, corruption was
rejected, and recovery was `1.000`. The reward-shuffled control remained at
chance. All runs used zero replayed examples. Reports and ledgers are in the
three `external_write_stable_value_v15*` directories.

## Five-slot temporal-bank scaling (2026-08-05)

The stable controller-native value path and isolated external writer scaled to
a five-slot, two-row bank. Seed 17 passed at 704 requested phase-1 steps with
target-first/last `0.974/0.973`, intact `0.975`, mastered retention `0.961`,
and unseen-token minimum `0.965`. Seed 69415 failed at 800 phase-1 steps
because parent acquisition never stabilized; following the experiment ladder,
only the curriculum budget was changed. At 1,600 requested phase-1 steps, the
same seed stabilized after 1,152 effective phase-1 updates and passed with
target-first/last `0.978/0.987`, intact `0.983`, mastered retention `0.992`,
and unseen-token minimum `0.969`.

The successful replication's persistent-memory audit reloaded at `0.984`,
rejected checksum corruption, and recovered at `1.000`. The reward-shuffled
control stayed at chance (`0.476` intact; `0.514`/`0.503` target-first/last).
All runs used zero replayed examples. This is a promoted narrow five-slot
scaling/persistence result, not general continual learning or arbitrary new
computation. The phase-1 extension is part of the result; the 800-step failure
is retained as a curriculum-budget rejection. Full reports and ledgers are in
the five `external_write_stable_value_v16*` directories.

## Learned utility-based eviction (2026-08-05)

The full-bank eviction fallback is now qualified as a separate memory-side
learning problem. A frozen controller and stable external writer feed opaque
candidate rows to an independently versioned scorer. Paired common-random
counterfactual arms force row 0 versus row 1 and train only from scalar recall
differences; the controller receives no row index, target label, or
counterfactual metadata.

Seed 17 reached balanced/target-first/target-last recall
`0.916/0.903/0.981`, and seed 69415 replicated `0.963/0.912/0.999`.
Strength-based eviction was `0.488/0.512` on target-first, versus learned
`0.903/0.912`; random target-first was `0.737/0.756`. Both runs passed
clear-memory, corruption, persistent reload, checksum rejection, and recovery
controls. The reward-shuffled control remained at chance and failed parent
acquisition. All runs used zero replayed examples.

This is a promoted narrow learned-eviction boundary for the three-slot/two-row
verifier. It does not establish general episodic memory, arbitrary new
computation, or general continual learning. The three reports and ledgers are
in the `learned_eviction_v1_*` directories beside this record.

## Variable-capacity append-only memory (2026-08-05)

The fixed-row-count assumption is now isolated behind a new replaceable
`AppendOnlyContentAddressedMemory` backend. The frozen canonical controller
appends unmatched opaque learned keys, upserts matching keys, and retrieves
records after a state reset. Its persistent replacement stores variable-length
state atomically with checksums.

Across seeds 17 and 69415, the backend passed 64, 256, and 1,024 records with
`1.000` permuted exact recall at every scale, zero clear-memory hits, and
`1.000` persistent reload/recovery. Fresh-token hit rates remained below
`1.1%`; checksum corruption was rejected at every scale. No optimizer updates
or replayed examples were used.

This promotes the logical storage-growth boundary through the frozen runtime.
It does not establish learned compression, new procedure acquisition, or
general continual learning. Reports and ledgers are in the two
`memory_growth_append_only_v1_*` directories.

## Routed artifact compaction (2026-08-05)

The first compaction attempt exposed the next architectural boundary: putting
two independent procedures in one row and executing both at once changes the
controller’s behavior. The corrected memory contract keeps one physical row
with multiple opaque aliases, but each verified alias also returns an opaque
view identifier. The caller projects only that view into the frozen growth
state.

Two 512-update seeds passed behavior preservation, parent retention, reload,
checksum rejection, frozen-core equality, and rejected-candidate controls.
The source bank shrank from two rows to one, with zero optimizer updates and
zero replayed examples during consolidation. This is promoted routed logical
compaction, not unrestricted procedure induction or general continual
learning. Independent capabilities stay append-only unless a held-out
behavior verifier admits the compact routed form. Evidence is in
`artifact_consolidation_v1_2026-08-05/`.

## Outcome-only executable-view routing (2026-08-05)

The caller view-selection shortcut is now replaced by a learned
`FactorizedOpaqueAddressRouter`. It receives controller-produced query tensors,
opaque candidate keys, attempted-view outcomes, and scalar verifier outcomes;
semantic task/span identity remains trainer-private.

Two seeds reached `1.000` held-out route accuracy and `1.000` candidate
permutation accuracy. Reward-shuffled routing stayed at `0.438/0.500`, wrong
views were causally worse, and reload/corruption/frozen-core gates passed. The
64-update pilot is retained as a curriculum rejection because the second
procedure was not yet mastered, while route accuracy was already perfect.

This promotes learned routing between already-acquired executable views, not
general continual learning or arbitrary new program induction. Evidence is in
`artifact_view_routing_v1_2026-08-05/`.

## Four-view routing scaling (2026-08-05)

The learned view router now scales from two to four independently acquired
span-4 procedures in one physical row. Context-derived addresses collided at
four views, so the promoted design uses independent opaque storage identities
and the joint opaque scorer with paired counterfactual credit.

Seeds 69316 and 69317 passed route/permutation accuracy `1.000/1.000` and
`0.969/0.969`, reward-shuffled routing `0.215/0.250`, four-procedure mastery,
wrong-view causal separation, reload, corruption, exact-candidate, and
frozen-core gates. The failed address, factorized-router, and direct-credit
controls are retained in the scaling record.

This promotes bounded four-view routing, not unrestricted memory growth or
general continual learning. Evidence is in
`artifact_view_routing_scaling_v1_2026-08-05/`.

## Outcome-gated online view growth (2026-08-05)

The next boundary is now qualified across two seeds. After the four-view
router is frozen, a fifth `rotate` procedure is acquired into external growth
memory and attached through a zero-initialized route extension. The extension
uses fresh fifth-procedure paired scalar outcomes only; no old route examples
are replayed after the extension.

The evidence rejects optimistic preemption: the frozen four-view router can
confidently misroute a novel procedure. The promoted selector therefore gives
known routes priority and opens the new view only after an observed failed old
attempt. Seeds 69316 and 69317 recovered the new view at `1.000/1.000`, kept
old-route accuracy at `1.000/0.988`, reached combined five-view accuracy
`1.000/0.994`, preserved candidate permutation behavior, and recorded zero
old false positives and zero reward-shuffled new selections. Reload,
corruption, frozen-core, frozen-router, and causal wrong-view gates passed.

This promotes safe outcome-gated external capability addition with a bounded
one-failure cold start. It does not establish immediate novel-task routing,
unrestricted continual learning, arbitrary new computation, or unbounded
memory growth. Evidence is in `online_view_growth_v1_2026-08-05/`.

## Two-step replay-free external view growth (2026-08-05)

The one-failure boundary now survives two sequential capability additions.
The frozen four-view controller/router acquires `rotate` as view `4`, then
acquires `complement_rotate` as view `5`. The first extension is frozen while
the second is trained; the second procedure first fails through the old route
and then through the first extension before the second extension is opened.
Both additions remain in one physical artifact row with six opaque views.

Across seeds 69316 and 69317, base old-route accuracy was `1.000/0.984`, both
new-view routes were `1.000/1.000`, and the complete two-step chain reached
`1.000/0.995`. Candidate permutation accuracy matched the chain. The first
extension selected itself on the second procedure at `1.000/1.000`, while
reward-shuffled first and second extensions selected their new views at
`0.000/0.000`. Selected behavior, wrong-view causal separation, exact reload,
checksum rejection, frozen controller core, frozen first extension, and
no-replay gates all passed.

This promotes a bounded two-step outcome-gated external fallback and
replay-free consolidation result. It does not establish unrestricted memory
growth, arbitrary new computation, open-ended task discovery, or general
continual learning. Reports and accounting are in
`multistep_view_growth_v1_2026-08-05/`.

## Three-step replay-free external view growth (2026-08-05)

The cumulative fallback chain now survives a third sequential capability
addition. After the frozen four-view base, `rotate`, `complement_rotate`, and
`adjacent_xor` are acquired one at a time. Each later procedure passes through
the old router and every earlier extension as a failed opaque attempt. All
seven views remain isolated aliases in one physical artifact row.

Seeds 69316 and 69317 reached old-route retention `1.000/0.992`, all three
new-view routes `1.000/1.000`, and complete three-step routing
`1.000/0.998`. Candidate permutation accuracy matched. Every prior-extension
attempt rate was `1.000` on later procedures; reward-shuffled new-view
selection was `0.000` for every extension on both seeds. Behavior, causal
wrong-view, exact reload, corruption, frozen-core, frozen-extension, and
zero-replay gates all passed.

This is the next promoted bounded continual-memory result: three sequential
external additions without controller updates or replay of earlier route
examples. It still does not establish unrestricted memory growth, arbitrary
new computation, open-ended task discovery, learned compression, or general
continual learning. Evidence is in
`three_step_view_growth_v1_2026-08-05/`.

## Behavior-verified fixed-capacity artifact compression (2026-08-05)

The seven-view chain now has a real payload-capacity result, not only one-row
logical compaction. A caller-owned, versioned float16 codec compresses the
complete seven-view tensor payload before transactional promotion; the frozen
runtime explicitly casts it back at the growth boundary. Across seeds 69316
and 69317, raw tensor bytes fell from `202,944` to `101,472` (`0.500`), and
serialized artifact bytes fell from `212,863` to `111,167` (`0.522`).

Compressed and uncompressed selected behavior were identical for every one of
the seven views on both audits. Wrong-view causal separation, exact opaque
aliases, reload, checksum rejection, frozen-core, frozen-extension, and
zero-replay gates all passed. Compression used zero optimizer updates and no
replayed examples.

This promotes behavior-verified fixed-capacity tensor compression for the
bounded seven-view artifact chain. It is a storage codec, not learned new
computation; arbitrary compression, open-ended memory growth, and general
continual learning remain unqualified. Evidence is in
`three_step_view_compression_v1_2026-08-05/`.

## Behavior-verified int8 artifact quantization (2026-08-05)

The same seven-view artifact now survives per-tensor symmetric int8
quantization with explicit scale tensors. Across seeds 69316 and 69317, the
payload fell from `202,944` to `50,848` bytes (`0.2506`), and the serialized
artifact fell to `69,771` bytes (`0.3278`). Quantized selected behavior stayed
within the predeclared five-point retention tolerance and above the behavior
floor on both seeds; wrong-view causal separation remained true.

Exact aliases, reload, checksum rejection, frozen controller/extension state,
and zero replay all passed. Quantization used zero optimizer updates. This is
the strongest current storage result, but it is still a replaceable codec and
not learned new computation or general continual learning. Evidence is in
`three_step_view_quantization_v1_2026-08-05/`.

## Behavior-verified packed int4 artifact quantization (2026-08-05)

The seven-view chain also survives packed signed-int4 storage. The
caller-owned codec quantizes each floating tensor per output row, packs two
values per byte, and stores explicit scales and original shapes. Decompression
occurs before the strict frozen-growth loader, leaving the controller and
memory backend unchanged.

Across seeds 69316 and 69317, raw tensor payload bytes fell from `202,944` to
`30,184` (`0.1487`), and serialized artifact bytes fell from `212,863` to
`58,007` (`0.2725`). The three-step route chain remained `1.000/0.998`,
minimum packed behavior was `0.7227/0.7305`, and packed behavior, wrong-view
causality, exact reload, corruption rejection, frozen-core, frozen-extension,
and zero-replay gates all passed.

This promotes behavior-verified packed int4 storage quantization for a bounded
external artifact chain. It is not learned compression, arbitrary new
computation, open-ended memory growth, or general continual learning. Evidence
is in `three_step_view_int4_v1_2026-08-05/`.

## Replicated episodic context and causal credit (2026-08-05)

The next continual-learning bottleneck now has a promoted memory-side
pressure test. `EpisodicContextEncoder` consumes ordered learned events,
opaque actions, scalar outcomes, and presence. It learns context from paired
augmented episodes without task labels, while paired write-intervention
outcomes train per-event credit. A frozen opaque router then appends a new
external route from fresh outcomes only.

Across seeds 69316 and 69317, context-based old-route accuracy was
`0.9688/1.000` versus `0.500/0.500` for pooled events. Candidate permutation,
new-route recovery, extension ablation, decisive-position credit, old-route
retention, shuffled-outcome rejection, and zero-replay gates all passed.

This promotes a bounded episodic-context and counterfactual-credit mechanism,
not unrestricted memory growth, arbitrary program induction, or general
continual learning. Evidence is in
`episodic_context_credit_v1_2026-08-05/`.

## Replicated two-step isolated-credit growth (2026-08-05)

The episodic context/credit loop now survives two sequential fresh additions.
The shared context encoder and old credit head are frozen; each new external
capability receives isolated event-credit state and a route extension trained
only from fresh paired outcomes. A later procedure must pass the old route and
the earlier extension before its own route activates.

Across seeds 69316 and 69317, both new routes selected at `1.000/1.000`, old
route retention and candidate permutation passed, prior-extension attempts
were present, and old/new credit-position accuracy was `1.000/1.000`.
New-route ablations and shuffled-outcome extensions selected `0.000/0.000`,
with zero replay after either append.

This promotes bounded two-step external growth with isolated credit state. It
does not establish unrestricted memory growth, learned eviction,
nonstationary discovery, arbitrary program induction, or general continual
learning. Evidence is in
`episodic_context_credit_multistep_v1_2026-08-05/`.

## Four-step isolated-credit growth (2026-08-05)

The same frozen episodic boundary now survives four sequential fresh
additions. Families `2,3,4,5` each receive an isolated route extension and
credit head. A later family must first fail through every earlier extension;
future inactive extensions are not counted as prior attempts. The per-event
credit target uses the opaque temporal position represented by each family
pattern rather than treating a family identifier as an event position.

Across seeds 69316 and 69317, old and new route selection, candidate
permutation, old-route retention, and old/new credit-position accuracy were
all `1.000`. Required prior-extension attempts were `1.000`, disabling each
required extension reduced selection to `0.000`, reward-shuffled extensions
were selected at `0.000`, and replay remained zero. Each seed used `122,880`
unique verifier bits, `30,976` logical lifetimes, and `2,048` optimizer
updates.

This promotes bounded four-step replay-free external growth with isolated
episodic credit state. It does not establish unbounded memory growth, learned
consolidation, arbitrary program induction, or general continual learning.
Evidence is in
`episodic_context_credit_four_step_v1_2026-08-05/`.

## Eight-step episodic-context growth (2026-08-05)

The finite four-token pattern bank was replaced for this pressure test by ten
same-statistics temporal patterns of length five. After the old context and
router were frozen, families `2` through `9` were acquired sequentially with
independent route and credit state.

Across seeds 69316 and 69317, old-route accuracy, pooled-baseline separation,
candidate permutation, all eight new routes, old-route retention, and old/new
credit accuracy were `1.000`. Required extensions were attempted at `1.000`,
required-extension ablations and reward-shuffled extensions selected at
`0.000`, and replay remained zero. Each seed used `286,720` unique verifier
bits, `62,464` logical lifetimes, and `4,352` optimizer updates.

The short-budget control failed old-route retention at `0.500` on both seeds;
the promoted schedule increases context and router training while preserving
fresh isolated external credit updates. This is a measured scaling cost, not
an omitted failure. The result promotes bounded eight-step replay-free growth,
not unbounded memory, learned consolidation, arbitrary program induction, or
general continual learning. Evidence is in
`episodic_context_credit_eight_step_v1_2026-08-05/`.

## Generated-pattern length-six growth (2026-08-05)

The pattern vocabulary is now generated from episode length rather than
limited to the compatibility bank. With length six, the harness provides 20
same-statistics procedures; families `2..9` were acquired sequentially after
the old context and router were frozen.

Across seeds 69316 and 69317, old-route accuracy, candidate permutation, all
eight new routes, old-route retention, and isolated credit accuracy passed;
required extensions were attempted at `1.000`, extension ablations and
reward-shuffled extensions selected at `0.000`, and replay was zero. The
promoted protocol used `393,216` unique verifier bits, `75,776` logical
lifetimes, and `5,632` optimizer updates per seed.

The under-budget control failed seed 69316 at `0.500` old-route accuracy and
passed seed 69317, so it remains rejected. This makes context-acquisition
scaling an explicit requirement for longer histories. Evidence is in
`episodic_context_credit_generated_len6_eight_step_v1_2026-08-05/`.

## Retention-safe memory boundary (2026-08-05)

The next implementation bottleneck was silent replacement of a mastered
capability. `CapabilityRetentionLedger` now sits outside the frozen
controller and records only opaque learned addresses plus scalar verifier
outcomes. Stable prefix mastery protects rows from both canonical content
memory and executable-artifact eviction; one noisy failure is tolerated, while
sustained low outcomes trigger a reversible new mastery era. If the entire
bank is protected, the write fails explicitly and asks the caller to grow or
perform verified consolidation.

The ledger persists beside disk snapshots, survives artifact compaction, and
`evaluate_retention_gate` rejects a candidate consolidation whenever any
retained capability falls below its declared floor. The canonical package now
has 220 passing tests, including persistence, runtime-checkpoint round trips,
noisy-failure hysteresis, reversal, protected compaction, full-capacity
protection, and rejected-retention-gate controls.

This is a foundational safety mechanism, not a Brain Workshop mastery claim.
The generated length-six extension below now reports stable prefix acquisition,
complete retention reversal, and growth-when-full controls.

## Generated length-six growth with retention reversal (2026-08-05)

The generated length-six sequence is now composed with the external retention
ledger. Across seeds 69316 and 69317, all ten opaque capabilities initially
became protected; a fully protected bank refused eviction; four sustained low
outcomes released only the newest capability; and four fresh successful
outcomes re-protected it. The route, permutation, causal-ablation,
isolated-credit, reward-shuffle, and zero-replay gates remained passing.

Each seed accounted for `393,304` unique verifier bits, `75,864` logical
lifetimes, `5,632` optimizer updates, and `88` retention observations. This
promotes bounded replay-free growth with a retention-safe reversible memory
boundary. It does not establish learned consolidation, unrestricted memory
growth, arbitrary new computation, or general continual learning. Evidence is
in `episodic_context_credit_generated_len6_eight_step_retention_v1_2026-08-05/`.

## Retention-aware artifact consolidation (2026-08-05)

The memory-side compaction boundary now protects mastered source rows and
registers fresh mastery evidence for the replacement. A two-phase audit first
verifies a candidate without adoption, then records eight held-out retention
probes per source capability before the final consolidation. Across seeds
69316 and 69317, two rows became one, aliases and executable views survived
reload, the frozen core and behavior were preserved, corruption was rejected,
and consolidation used zero optimizer updates and zero replay. The short
64-update control failed stable candidate mastery and was not adopted.

This promotes retention-aware behavior-verified logical compaction. It remains
bounded memory management, not learned byte compression, unrestricted growth,
arbitrary new computation, or general continual learning. Evidence is in
`artifact_consolidation_retention_v2_2026-08-05/`.

## Opaque learned sequential consolidation (2026-08-06)

The canonical package now contains a permutation-equivariant memory-side
consolidation policy. It learns pair selection from scalar rewrite utility over
opaque controller-native rows, while immutable transactions and fresh
retention outcomes gate adoption. Across seeds 69316 and 69317, it selected
verifiable pairs on every 512-bank held-out audit and composed four accepted
rewrites from eight rows to four. Candidate permutation, corruption,
reward-shuffle, retention, source immutability, and zero-replay controls passed.

This promotes learned opaque latent compaction, not yet executable-artifact
behavioral consolidation, learned byte compression, unrestricted memory growth,
arbitrary new computation, or general continual learning. Evidence is in
`opaque_consolidation_v1_2026-08-06/`.

## Learned executable-artifact consolidation (2026-08-06)

The learned opaque policy was next applied to four independently acquired
executable growth artifacts. At 1,024 acquisition updates, both canonical
seeds passed three sequential retention-gated rewrites from four physical rows
to one. All four views remained behaviorally usable through each rewrite and
after persistent reload; aliases, frozen-core equality, checksum corruption,
and zero-replay controls also passed. The controller received zero optimizer
updates during consolidation. The protected-source transaction now builds a
disposable candidate, probes fresh outcomes, applies retention, and only then
invokes the behavior verifier.

The matched 512-update control was rejected because its candidate retention
prefix did not establish stable `.75` mastery. This promotes bounded
executable-artifact logical compaction and identifies acquisition depth as the
current bottleneck; it does not establish byte compression, arbitrary program
induction, unrestricted memory growth, or general continual learning. Evidence
is in `learned_executable_consolidation_v1_2026-08-06/`.

## Learned executable route acquisition (2026-08-06)

The compacted executable bank now has a learned acquisition path as well as a
storage path. Across seeds 69316 and 69317, an opaque permutation-equivariant
router learned four executable addresses from attempted outcomes, resolved
the selected alias through generic memory promotion, and preserved routed
behavior after reload. Route accuracy was `1.000` and `0.945`; permutation
accuracy matched those values; reward-shuffled controls stayed at `0.277` and
`0.250`; and wrong-view behavior was causally lower for every operation.

This is promoted bounded learned address acquisition over compacted external
executable memory. It remains a foundation for continual learning, not
arbitrary new skill induction, byte compression, unrestricted memory growth,
or general continual learning. Evidence is in
`learned_executable_route_acquisition_v1_2026-08-06/`.

## Retention-safe online executable growth (2026-08-06)

The fifth-view online extension now uses the external retention ledger as a
transaction boundary. Across seeds 69316 and 69317, four old executable views
were protected from fresh outcomes, a new `rotate` view was probed in a
disposable candidate store, and the replacement row was adopted only after
candidate retention and independent behavior gates passed. One physical row
held five opaque views; the controller and old router stayed frozen; the new
route was learned with zero replay after extension.

Both seeds passed old-route retention, new-view learning, candidate permutation,
wrong-view causality, reward shuffle, reload, checksum corruption, and frozen
core/router controls. Combined five-view route accuracy was `1.000` and
`0.9844`; the new route was `1.000` on both seeds. The short acquisition
control was rejected for failing the `0.70` retention contract. This promotes
one bounded retention-safe online executable addition, not unrestricted memory
growth, arbitrary new computation, or general continual learning. Evidence is
in `online_view_growth_retention_v2_2026-08-06/`.

## Retention-safe two-step executable growth (2026-08-06)

The online extension now composes twice under the retention ledger. Across
seeds 69316 and 69317, four old executable views were protected, `rotate` was
acquired and protected, and `complement_rotate` was then acquired through a
second disposable candidate transaction. All six opaque views remained in one
physical row while the controller, old router, and first extension stayed
frozen. Two-step route accuracy was `1.000` and `0.9974`, with zero replayed
route examples after either addition.

Both seeds passed intermediate and final retention, behavior-preservation,
permutation, causal, reward-shuffle, reload, corruption, and frozen-core
controls. The short acquisition control was rejected when the first protected
extension fell below its retention floor. This promotes bounded two-step
retention-safe growth, not open-ended additions, arbitrary new computation, or
general continual learning. Evidence is in
`multistep_view_growth_retention_v2_2026-08-06/`.

## Protected artifact capacity growth (2026-08-06)

`ExecutableArtifactMemory.grow()` is now the explicit escape hatch when a
full bank contains only protected rows. Across seeds 69316 and 69317, a third
write was refused rather than evicting either mastered row; a separate larger
bank copied verified artifacts and retention state; and the new artifact was
admitted only after growth. Source state remained unchanged and all artifacts
reloaded successfully. This promotes a memory safety boundary, not learned
capacity planning or general continual learning. Evidence is in
`artifact_capacity_growth_v1_2026-08-06/`.

## Three-step retention growth promotion (2026-08-06)

The three-step harness applies retention-safe transactions to seven opaque
views and to float16, int8, and int4 storage replacements. Seeds 69316 and
69317 both passed the full audit, including three protected intermediate
replacements, all storage controls, frozen controller/earlier extensions, and
zero replay. The earlier seed-69317 rejection was traced to inconsistent raw
minimum versus stable-prefix retention accounting and is retained as a
historical control. Evidence is in
`three_step_view_growth_retention_exploration_2026-08-06/`.

## Concurrent compositional transfer (2026-08-09)

Two direct capabilities and a fresh decoder/bridge learned concurrently while
the decoder exposed a frozen chain of three acquired source capabilities.
Both seeds passed the causal, missing-evidence, consolidation, and exact
source-retention gates with zero replay. This promotes bounded compositional
reuse under concurrent plasticity; multiple independently sampled programs,
longer horizons, and fresh-learner transfer remain open. Evidence is in
`external_register_real_basis_acquisition_2026-08-08/interleaved_composition_transfer/`.

## Multiple concurrent composition programs (2026-08-09)

Two independently ordered frozen source programs were learned through fresh
decoder/bridge pairs while two direct capabilities learned concurrently. Both
seeds promoted all four candidates with zero replay and exact source
retention. This strengthens bounded compositional reuse; larger program
grammars and fresh-learner transfer remain open. Evidence is in
`external_register_real_basis_acquisition_2026-08-08/interleaved_multi_composition_transfer/`.

## Full three-source permutation grammar (2026-08-09)

All six orderings of the three acquired source operations were exposed through
fresh decoder/bridge candidates while two direct capabilities learned
concurrently. Both seeds promoted all eight candidates with zero replay and
exact source retention. This validates the complete tested finite grammar,
not unrestricted program induction. Evidence is in
`external_register_real_basis_acquisition_2026-08-08/interleaved_full_composition_grammar/`.

## Matched fresh-learner transfer diagnostic (2026-08-09)

The full permutation grammar was compared with fresh three-instruction
learners under the same verifier budget. The inherited memory path remained
promoted, but strict positive transfer was not replicated across seeds. This
localizes the next bottleneck to efficient adaptation of fresh decoders and
event bridges, rather than retention. Evidence is in
`external_register_real_basis_acquisition_2026-08-08/interleaved_full_grammar_transfer/`.

## Mastered decoder prior diagnostic (2026-08-09)

Copying a mastered source decoder into new composition candidates was tested
as an adaptation shortcut. It was seed- and program-dependent, so the strict
replicated transfer gate rejected it. Raw action-decoder reuse is not adopted
as the general prior. Evidence is in
`external_register_real_basis_acquisition_2026-08-08/interleaved_decoder_prior_diagnostic/`.

## Shared event-bridge prior diagnostic (2026-08-09)

An event bridge trained from mastered source outcomes and then frozen for new
compositions was tested across both seeds. It was seed-dependent and failed
the replicated composition gate, so it is not adopted as the general prior.
Evidence is in
`external_register_real_basis_acquisition_2026-08-08/interleaved_shared_bridge_prior_diagnostic/`.

## Capability-conditioned bridge prior diagnostic (2026-08-09)

A shared event bridge conditioned on opaque learned program vectors showed a
positive result in seed 69317 but failed both composition gates in seed 69316.
It remains an opt-in research path, not a promoted general continual-learning
mechanism. Evidence is in
`external_register_real_basis_acquisition_2026-08-08/interleaved_conditioned_bridge_prior_diagnostic/`.

## Conditioned bridge source-restart repair (2026-08-09)

Increasing source acquisition to two restart candidates repaired the mixed
conditioned-bridge behavior result: both seeds promoted both compositions with
zero replay. Fresh stable-transfer mastery still failed, so this is a bounded
robustness promotion rather than a general sample-efficiency claim. Evidence
is in
`external_register_real_basis_acquisition_2026-08-08/interleaved_conditioned_bridge_restart_repair/`.
