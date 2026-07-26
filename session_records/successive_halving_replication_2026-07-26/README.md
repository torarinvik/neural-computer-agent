# Successive-halving population-race replication

Date: 2026-07-26

## Frozen protocol

Replicate the complete promoted ladder on unseen physical stream 7078 with
clones 7140–7143. Hyperparameters and selectors are unchanged from the
pre-registered stream-7077 experiment:

1. all four clones train to round 18 and receive the four-seed read-only shadow
   acquisition audit;
2. rank by descending minimum context-mean best-slot reward advantage, then
   descending number of context-specializing audit seeds, then ascending clone
   ID; advance two;
3. resume both to round 42 and rank by descending mean verified
   learned-minus-frozen reward over exactly the first six `old_return` rounds,
   then descending worst-round advantage, then ascending clone ID; advance one;
4. resume the winner to round 54;
5. complete fixed lowest-ID clone 7140 as the replication control.

No result from stream 7077 may alter the protocol. Adversarial intervention
controls are repeated only if the replication produces an unusually large new
effect or contradicts the already established causal result.

## Replication gate

Replication passes if the selected winner is better than fixed control in the
direction of both reliability acquisition and old-return performance, while
binary/four-rule retention, physical/tensor parity, persistence, and exact
prefix continuation all pass. Magnitude need not match stream 7077.

## Results

The acquisition screen ranked the clones:

| Clone | Conservative shadow advantage | Specializing seeds | Decision |
|---|---:|---:|---|
| 7140 | **+2.0833 points** | 0/4 | advance |
| 7141 | 0.0000 points | **3/4** | advance by tie-break |
| 7142 | 0.0000 points | 1/4 | stop |
| 7143 | 0.0000 points | 0/4 | stop |

At the retention rung, 7140 averaged +1.389 reward points over the first six
return rounds with a zero worst-round advantage. Clone 7141 was exactly zero
on every return round. Clone 7140 therefore advanced.

The completed 7140 trajectory passed the full gate and retained both inherited
primitives. It achieved:

- reliability target accuracy 20.83% versus 11.11% frozen
  (**+9.72 points**);
- old-return target accuracy 6.94% versus 0% frozen
  (**+6.94 points**);
- old-return reward advantage **+0.926 points**.

Every resumed trace prefix remained exact.

## Verdict

Scientifically useful but comparative-inconclusive. The frozen lowest-ID
control, 7140, was itself selected as the winner. Completing it twice would
produce no comparison, and substituting another control after seeing the
ranking would be post-hoc. The selected trajectory itself repeated safe
acquisition and return gains, but the pre-registered winner-versus-control
replication gate cannot be evaluated.

The next replication must define its validation control so it can never
coincide with the winner. Freeze the control as the highest-ranked clone
eliminated at round 18, using the same acquisition ranking and no later
outcomes. This compares the winner against the strongest early-pruned
trajectory without altering the production selector.
