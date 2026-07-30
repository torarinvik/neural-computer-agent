# Sequence working-memory breakthrough

## Question

Can the same compact controller progress from passive short-term retention to
an active operation over retained content, and does mastering retention make
the operation faster to acquire?

Each episode renders two abstract binary shapes, followed by query frames that
request either the original order or the reverse order. The controller sees
only RGB frames, its own opaque attempted actions, and the scalar success of
each attempt. Sequence values, operation bits, and correct actions remain
verifier-private.

All recurrent-state and differentiable-workspace tensors remain on the selected
device. On MPS/CUDA this is literal unified RAM/VRAM-resident fast memory; disk
does not participate in the real-time episode.

## Curriculum

The first jump—two items plus a distractor—was flat at chance after 8,192
verifier bits. Following the one-axis rule:

1. one-item forward retention reached 100% held-out;
2. two-item forward retention reached 100%;
3. mixed forward/reverse training was then initialized either from that
   forward skill or from fresh weights.

The forward parent used 12,288 prior verifier bits. The transfer comparison
counts the *new* mixed-operation evidence separately; it therefore measures
reuse on the next skill, not total lifetime experience from birth.

## Primary result

At the same 16,384 new verifier bits:

| Arm | Held-out | Forward | Reverse | Valid cue-reversal flips |
| --- | ---: | ---: | ---: | ---: |
| forward-parent curriculum, seed 26001 | 99.11% | 100.00% | 98.23% | 96.53% |
| fresh mixed learner, seed 26001 | 81.51% | 75.51% | 87.50% | 72.44% |
| forward-parent curriculum, seed 26101 | 93.84% | 100.00% | 87.68% | 75.00% |
| fresh mixed learner, seed 26101 | 75.34% | — | — | 0.37% |

Seed 26001 reached and retained the 90% gate at 14,336 new verifier bits. The
fresh arm had not reached it by 16,384 bits, so the measured novel-skill
sample-efficiency ratio is already greater than `16,384 / 14,336 = 1.14x`.
More importantly, the curriculum preserved forward recall at 100% in both
replicas while acquiring reversal; the fresh learners either remained near
the 75% operation-blind shortcut or traded one operation against the other.

This is the first direct evidence in this branch that an acquired retention
primitive accelerates learning a manipulation primitive without forgetting
the retained behavior.

## Causal and adversarial controls

The selected seed-26001 checkpoint was evaluated on 8,192 fresh episodes:

- blank sequence evidence: 50.00%;
- all fast memory reset immediately before query: 50.24%;
- valid operation-cue reversal: 99.15% accuracy and 96.53% prediction flips on
  non-palindromic sequences;
- valid input-sequence reversal: 99.26% accuracy and 96.75% flips;
- operation cue blanked: 74.66%, the analytically expected operation-blind
  shortcut rather than mastery;
- differentiable workspace disabled: 94.95%;
- recurrent active state reset while preserving workspace: 87.48%;
- outcomes shuffled between lifetimes during training: 50.00%, zero flips.

The two partial ablations show redundant fast-memory carriers: neither the GRU
state nor the physical workspace is solely responsible, but removing all fast
memory destroys the skill. Shuffled outcomes eliminate learning, so pixels or
generator regularities alone do not explain the result.

## Honest limitations

This is a two-item working-memory atom, not general working memory.

- A previously unseen distractor reduces the selected checkpoint to 84.08%
  and cue-reversal flips to 36.13%.
- A disjoint position layout reduces it to 11.40%; the controller learned
  position-specific perception rather than invariant identity.
- The checkpoint is a separate sequence branch, not yet consolidated into the
  larger relation/magnitude/numerosity repertoire.
- The learned policy uses both recurrent state and differentiable workspace;
  it has not yet learned explicit write/keep/erase decisions or variable
  capacity allocation.

The next frontier is therefore gradual appearance invariance, then one
distractor with selective retention. Architecture changes are not yet
justified: the existing controller already demonstrates the required
retention-and-manipulation computation on the matched visual distribution.

## Artifacts

- `curriculum_seed26001.json`
- `fresh_seed26001.json`
- `shuffled_outcomes_seed26001.json`
- `curriculum_seed26101.json`
- `fresh_seed26101.json`
- `replication_seed26101.json`
- `distractor_audit_seed26001.json`
- `artifacts/checkpoints/unified_sequence_working_memory_seed26001.pt`

The curated checkpoint is a verified two-item sequence specialist, not the
promoted general repertoire controller.
