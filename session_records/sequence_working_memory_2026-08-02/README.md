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

This is a new **compounding transfer signal to span eight**, not a
claim of eight-item mastery. A 32-episode retention smoke audit found no
catastrophic collapse in spans 2--7 (98.44%, 91.67%, 82.81%, 73.12%, 70.83%,
and 70.54%, respectively), but its sample is too small to replace the
high-precision retention audits used for promotion. The span-eight checkpoint
therefore remains an unpromoted experimental artifact until a larger,
lifetime-disjoint audit and a second inherited seed are run.

The independent span-seven replication immediately before this escalation
reached 66.02% from 1,888 verifier bits, with 45.92% operation flips and
50.33% complete-reset accuracy. This lower-but-positive result is retained as
the appropriate seed-variance bound rather than hidden behind the stronger
span-eight child.

## What this establishes

This is a verified compounding working-memory result: a learned retention
primitive makes a harder manipulation primitive reachable with fewer fresh
verifier bits, and explicit rehearsal prevents the older primitive from being
overwritten. It is still a specialist sequence branch, not yet a generic
variable-capacity memory or a fully consolidated repertoire capability.

## Next frontier

The next highest-ROI experiment is now a second inherited span-eight seed with
the same schedule, preceded by a 512--2,048-lifetime retention audit of spans
2--7. Promote only if the child clears the existing two-point retention gate,
keeps blank/reset controls near chance, and shows the same inherited-versus-
fresh gap. Then spend additional private compute on consolidation before
consuming more verifier bits. Do not add learned variable-capacity memory,
registers, or new modalities yet: the evidence still says that gradual
difficulty plus weighted rehearsal delivers the highest return per verifier
bit.

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

The ignored local checkpoint hashes are:

```text
span8_smoke_inherited_seed30564.pt  sha256 9a429e1eea0b1c1f2e30c02ce9d91c4e32ceac257d8d55e3d37ae0e5384c4b7e
span8_smoke_fresh_seed30565.pt      sha256 12d4dc2b11fc126f9cc1b613622d5edcb228fdda9c87d1eb545895db5005e099
span8_shuffled_seed30566.pt         sha256 518b443d6e765d70bcc71c5873e63ab65cf41de68550c96b442726d3b83cc895
```
