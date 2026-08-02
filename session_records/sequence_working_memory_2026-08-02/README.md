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

## What this establishes

This is a verified compounding working-memory result: a learned retention
primitive makes a harder manipulation primitive reachable with fewer fresh
verifier bits, and explicit rehearsal prevents the older primitive from being
overwritten. It is still a specialist sequence branch, not yet a generic
variable-capacity memory or a fully consolidated repertoire capability.

## Next frontier

The next highest-ROI experiment is a second inherited span-seven seed using
the same weighted schedule, followed by a 16--32 update continuation only if
spans 2/3 remain within the two-point gate. If that replicates, test span-eight
with the same early-span-heavy rehearsal principle and a matched fresh
control. Do not add learned variable-capacity memory, registers, or new
modalities yet: schedule design remains the highest return per verifier bit.

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
