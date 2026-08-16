# Live operator transfer across Workshop and two rendered mazes (2026-08-16)

Status: **development diagnostic; no holdout, promotion, or curated-bank
admission**.

This audit runs one canonical agent through the real public path:

```text
rendered Workshop -> learned event tensors -> one controller
-> rendered source maze -> rendered target maze -> rendered Workshop
```

The controller receives only learned event tensors, opaque feedback, and its
own memory. The source maze's successor table is not copied to the target.
The candidate operator is admitted only after 40 source evaluation outcomes
pass the stable-prefix gate and a digest retention probe. The matched control
uses the same seeds, Workshop warm-up, source-maze budget, and target budget,
but receives no operator in the target maze.

## Three-replicate development result

| measure | result |
| --- | --- |
| source candidates admitted | **3 / 3** |
| stable-prefix indices | **14, 11, 17** (of 40 outcomes) |
| one core across all four stages | **3 / 3** |
| controller unchanged | **3 / 3** |
| admitted target final return | **1.000, 1.000, 1.000** |
| matched no-operator final return | **0.000, 0.000, 0.000** |
| target stable bits to threshold | **1,922; 2,163; 1,763** |
| positive final-return advantage | **+1.000 in every replicate** |

The earlier live run was correctly rejected because its source evidence was
unstable. Two fixes made the evidence and planner faithful without relaxing
the gate: evaluation now retains each authenticated episode outcome, and the
terminal rendered frame is recorded so a rewarded arrival becomes a learned
goal. The planner also holds at a known rewarded goal instead of drawing a
random action there.

This is strong development evidence that a verifier-gated, world-independent
planning operator reduces target-maze experience in the actual rendered loop.
It is not yet a promotion claim: the operator control-flow remains
hand-specified, the seed block is not a reserved holdout, and the target
control did not itself reach the threshold.

## Next step

Freeze this development contract and run the preregistered controls (fresh,
action-shuffled, missing-evidence, poisoned, reversal, and exact-equivalence)
before considering any holdout amendment. Do not admit this artifact to the
curated bank yet.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.live_operator_transfer \
  --neural-workshop '/Users/torarinvikbjarko/Documents/Coding Projects/Python Projects/neural-workshop' \
  --output session_records/brainworkshop_live_operator_transfer_2026-08-16 \
  --replicates 3 --trials 4 --source-maze-training-episodes 40 \
  --target-maze-training-episodes 40 --maze-evaluation-episodes 2 --maze-steps 20
```
