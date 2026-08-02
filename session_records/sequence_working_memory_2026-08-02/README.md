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

## What this establishes

This is a verified compounding working-memory result: a learned retention
primitive makes a harder manipulation primitive reachable with fewer fresh
verifier bits, and explicit rehearsal prevents the older primitive from being
overwritten. It is still a specialist sequence branch, not yet a generic
variable-capacity memory or a fully consolidated repertoire capability.

## Next frontier

The span-four gate is now passed. The next highest-ROI experiment is a short
32--48-update span-four continuation from the best balanced checkpoint, with
the same `[4, 2, 2, 3, 3]` schedule and checkpoint probes every 8--16 updates.
Stop if span-two or span-three falls by more than two points, or if blank/reset
controls move away from chance. If span-four rises while retention holds, the
next escalation is **five items with two distractors**, again starting from
the promoted span-four checkpoint and using a matched fresh control. Do not
add learned variable-capacity memory, registers, or new modalities yet: the
current evidence says rehearsal and gradual one-axis span escalation have the
highest return per verifier bit.

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
