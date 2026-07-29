# Natural latent-memory equivalence

## Question

Can the single controller recognize that independently acquired latent
memories implement the same behavior, retrieve any equivalent memory, and
reject conflicting memories without receiving a semantic rule identity?

This replaces the previous four-row benchmark's generator-created opposite
value. Every stored value in this experiment is emitted by the controller
after a real visual support trial and scalar feedback.

## Probe before repair

A discarded diagnostic first tested whether the relation was present at all.
Pairs of independent memory values were balanced by verifier-private rule
equivalence and evaluated on held-out appearances:

- raw cosine threshold: 87.03%;
- linear pair probe: 99.79%;
- 32-unit pairwise probe: 100%.

The probe weights were discarded. Rule bits never enter the deployed module,
training loss, or evaluation policy. This localized the problem to reading an
existing latent relation rather than perception or memory writing.

## Architecture

The promoted 298,945-parameter parent remains frozen. A 12,354-parameter
shared scorer receives, for each stored row:

- the fresh feedback-derived memory value;
- the independently stored memory value;
- their absolute latent difference;
- their elementwise product.

It scores all four rows with shared weights. A straight-through hard choice
selects one of the existing generic rank-interval proposals, preventing a
soft average from landing between disconnected but equivalent intervals. A
zero-initialized scalar opening makes insertion exactly behavior preserving.

Only the scorer and opening train. Candidate credit is the probability mass
placed on rows whose retrieved values earn scalar verifier reward. No target
row, rule bit, correct action, semantic task ID, or unattempted action label is
learner-visible.

## Gradual curriculum and accounting

Each accepted run uses four batches of 32 lifetimes:

1. exact latent duplicates;
2. partially exact and independently equivalent memories;
3. fully independent memories;
4. fully independent memories.

Every generated bank is filtered using four actually observed verifier
outcomes so it contains at least one equivalent and one conflicting memory.
Rejected generated lifetimes and all four mining outcomes remain in the
experience accounting.

The first successful formal schedule used 4,096 verifier bits. Holding
internal compute near 512 optimizer updates while shortening the curriculum
reduced this to a replicated 1,024 verifier bits. A 768-bit schedule passed
two seeds but failed a third at 92.77%, so it is recorded as a best case and
not promoted.

## Results

| run | verifier bits | held-out | one / two / three equivalents | physical banks | accepted |
|---|---:|---:|---:|---:|---:|
| seed 20252 | **1,024** | **100%** | **100 / 100 / 100%** | **128/128** | yes |
| seed 20253 | **1,024** | **100%** | **100 / 100 / 100%** | **128/128** | yes |
| reward shuffled 20252 | 1,024 | 43.55% | 16.15 / 43.58 / 65.24% | 41.41% | no |
| exact-only 20252 | 1,024 | 86.91% | 76.92 / 88.99 / 92.07% | 84.38% | no |

The inherited retrieval policy scored 46.88–50.59%. In both accepted runs:

- the relation selector itself scored 100%;
- probe shuffling reduced behavior to 49.22–51.95%;
- stored relation-value shuffling reduced behavior to 52.73–53.52%;
- retrieved-value corruption reduced behavior to 35.16–35.35%;
- unpermuted and randomly permuted physical rows both scored 100%;
- all 256 disk banks reloaded keys, values, and usage exactly and behaved
  correctly;
- parent continuous retrieval was 99.02%, parent conditional retrieval was
  100%, and binary/four-rule behavioral gates passed;
- only the five equivalence-module tensors changed.

## Counterfactual replay

The strongest audit replays the same held-out visual lifetimes and identical
candidate banks while reversing only the target verifier's hidden binary
rule. Keys, values, usage, queries, and physical row order are tensor
identical. The changed support outcome produces a different fresh latent.

Both accepted seeds achieved:

- 100% ordinary accuracy;
- 100% reversed accuracy;
- 100% fresh-probe change rate;
- 100% physical selection-flip rate.

The policy therefore follows the experience-derived latent relation rather
than a fixed row, fixed rule, rendering cue, or generator label.

## Promoted artifact

`artifacts/checkpoints/unified_natural_memory_equivalence_seed20252.pt`

SHA-256:
`ece57f7a02d725c75177d7df217b5c940459ea728f9d2ec6eb619af2f3c4f628`

## Conclusion and next frontier

The system now recognizes behavioral equivalence between independently
learned, non-identical latent memories and uses that relation through real
disk-backed retrieval. The gradual exact-to-independent bridge is causally
necessary at the measured budget, and replay converts 1,024 verifier bits
into stable mastery without forgetting.

The next frontier is to use this learned equivalence relation for
capacity-limited consolidation: merge or discard redundant memories while
retaining every behaviorally distinct skill, then test whether the smaller
bank reduces the experience needed for the next novel primitive.
