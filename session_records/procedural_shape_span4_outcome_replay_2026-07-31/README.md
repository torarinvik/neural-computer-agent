# Four-item relation: interleaved outcome-replay breakthrough

## Claim

A frozen three-item controller already contains a representation of the new
third-item-to-fourth-item relation, but its ordinary action reader cannot use
it. A newly added, initially zero-output action residual can learn that reader
from the controller's **own binary attempted-action outcomes**, without any
learner-visible labels, when every small block of new-task replay is
interleaved with old-skill outcome replay.

This is a diagnostic capability result. The action-residual weights are not
yet promoted into the persistent controller; promotion requires the normal
integration ladder and broader retention audit.

## Setup

- Parent: promoted span-three error-balanced checkpoint.
- New task: four sequentially presented procedural shapes; query asks whether
  a candidate is the item after the third cue. The learner receives RGB,
  opaque binary action, and scalar outcome only.
- One generic internal thought/read step is permitted before action. It adds
  latency but no information.
- A 64-wide zero-output nonlinear action residual is the only trainable
  component.
- Four unique new-task batches × 256 outcomes = **1,024 target outcomes**.
- Each target batch is replayed for 16 reader updates, in blocks of two.
- After every target block, each of the three old-skill streams supplies one
  replay update at its original zero-thought timing. These are 3,072 distinct
  rehearsal outcomes and 96 rehearsal optimizer updates.

Outcome replay changes compute per experienced outcome, not the information
available to the learner: each binary outcome identifies the unique correct
action only because the action space has exactly two opaque actions.

## Replicated results

| arm | new overall | new strict relation | old overall | old relation conflicts |
|---|---:|---:|---:|---:|
| primary, seed 44905 | 91.31% | 86.26% | 96.40% | 98.44% |
| replica, seed 44906 | 93.85% | 93.08% | 97.40% | 97.92% |

Both new-task audits pass the causal checks: complete memory reset is at
48.24% / 50.39% chance, and candidate counterfactual accuracy is 91.41% /
93.95%. The controller is therefore responding to the visual sequence and
candidate, not a reward shortcut.

## What failed first

1. A four-item generator initially required 6,144 examples to exhaustively
   cross all query permutations. It now keeps identity/answer balance exact in
   256-example batches and samples query order independently when the full
   factorial cross is too expensive.
2. Direct reward-only training, loss weighting, thought steps, and a gated or
   ungated reader residual did not learn the new relation in 1,024 outcomes.
3. A disposable frozen-state probe localized the failure: a nonlinear MLP
   decodes the correct action at about 99% held-out from controller state;
   a linear probe reaches about 77%. The information was present but the
   behavioral reader could not acquire the nonlinear mapping quickly.
4. Outcome replay without rehearsal learned the new relation (~90%) but
   overwrote old skill (~77%). Block-ordered rehearsal protected old skill but
   suppressed new learning. Fine-grained interleaving resolves this trade-off.

The next step is to integrate this schedule into the regular trainer, then
repeat the full causal and retention ladder before promoting any checkpoint.
