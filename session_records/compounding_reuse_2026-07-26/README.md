# Compounding reuse horse race — pre-registration

## Question

Does retrieving a previously verified latent skill reduce the attempted-outcome
experience needed to master a nearby harder environment?

This experiment changes memory capacity from 6 to 7 while preserving the
controller, sensory stream, four generic evidence values, two physical
operations, scalar outcome, and action-value objective. This makes the change
harder without changing operation semantics.

## Phase 0: viability

Before training, a fresh capacity-7 stream must contain:

- at least 2% cases where re-query helps;
- at least 2% cases where re-query harms;
- at least a 2-point oracle utility advantage over the strongest fixed action.

Seed 8021 passed: help `8.87%`, harm `91.13%`, oracle gap `8.05` points.

## Phase 1: sub-minute transfer smell test

Run matched stored-child and reset action-value challengers with the same
controller, seed, active disagreement allocation, optimizer, and verifier-bit
budget.

Primary evidence:

1. the stored child has at least a 2-point zero-update held-out utility
   advantage over reset; and
2. it reaches a positive proposal lower bound in fewer verifier bits, or the
   reset arm produces no positive proposal within the same budget.

This phase is diagnostic only. Its short confirmation window cannot promote a
new durable skill and must not weaken the established 2,400-fresh-bit
confirmation requirement.

## Phase 2: safe confirmation

Run only if phase 1 passes. Preserve the established gates:

- proposal no earlier than 480 attempted-outcome bits;
- frozen proposal;
- at least 2,400 disjoint fresh attempted outcomes;
- positive lower 95% confidence bound;
- unchanged mastered incumbent and old-task retention;
- all ancestor skills remain hash-verifiable and loadable.

The stored-child lineage wins only if it reaches confirmed promotion in fewer
verifier bits than a matched reset lineage. Final accuracy alone is not the
objective.

## Accounting and controls

- Count attempted scalar outcomes as verifier bits.
- Count all unlabeled candidate contexts.
- Never expose unattempted outcomes, correct actions, task names, or private
  evaluation metrics to the learner or promotion rule.
- Use fresh seeds for replication if phase 2 passes.
- A failure may mean negative transfer; do not overwrite the stored parent or
  child.

## Results

Two independent sub-minute tests passed the diagnostic rung. On seeds 8022 and
8023 the stored child began at `0.6166`/`0.6138` utility versus
`0.5502`/`0.5498` for reset, and proposed at 448 bits while reset had no proposal
by 784 bits.

The full confirmation race then failed the strict compounding gate:

| Initialization | Proposal | Confirmation | Lower 95% |
|---|---:|---:|---:|
| stored child | 1,232 bits | 3,696 bits | `+0.1131` |
| reset | 560 bits | 3,024 bits | `+0.0122` |

The transferred child began stronger (`0.6164`) than the reset lineage's
eventual confirmed incumbent (`0.6056`), but did not confirm in fewer bits.
Fine-tuning also reduced it on some streams. This is zero-shot reuse, not yet
compounding learning speed.

An eight-stream frozen causal audit separated genuine learning from global
improvement:

| Policy/control | Mean utility |
|---|---:|
| mastered parent | `0.6346` |
| stored action-value child | `0.6235` |
| reset action-value head | `0.5587` |
| child with shuffled context features | `0.5662` |
| child with reversed actions | `0.5086` |

The child is causal and learned: it beats reset by 6.48 points, feature
shuffling costs 5.72 points, and action reversal costs 11.49. But the older
mastered parent still beats it by 1.11 points, so the global compounding gate
fails.

## Provenance correction

The historical manifests linked the action-value child to the mastered skill,
but the child was trained and promoted against a reset gap incumbent. That link
described task relatedness, not optimization ancestry. Historical artifacts
remain immutable. Future commits record such a branch as a root with
`baseline_kind=reset_action_policy` and retain the mastered policy only as a
`related_skill_id`.

The durable milestone remains valid but narrower: the system safely learned,
confirmed, stored, and reloaded a policy that improves a reset branch. It did
not improve the best existing policy.

## Next gate

The next learner must start from an immutable retrieved global champion, train a
separate challenger, and win only by:

1. beating that champion on a fresh related task with a positive independent
   lower confidence bound;
2. reaching an absolute capability bar in fewer verifier bits than reset;
3. retaining all older audited skills; and
4. passing feature-shuffle, action-reversal, and fresh-seed replication audits.

Cheap frontier scans rejected capacity 8, capacity 9, and capacity-7 costs
`0.00`/`0.02` as immediate training targets. The champion had only `1.98`,
`1.80`, `1.95`, and `2.45` utility points of oracle headroom. These are poor
high-ROI learning surfaces because even an optimal challenger barely clears the
practical improvement gate.

The next experiment should therefore use a new but nearby cognitive primitive
with a larger verified champion-to-oracle gap. Reuse is measured by an absolute
capability threshold and samples-to-threshold; promotion still requires beating
the immutable global champion, never merely a reset branch.
