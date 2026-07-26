# Prospective successive-halving population race

Date: 2026-07-26

## Question

Can the promoted acquisition and retention screens preserve the best learning
trajectory while spending less training compute than extending every clone
through all 54 physical rounds?

## Pre-registered protocol

Use fresh physical stream 7077 and clones 7130–7133. All clones use the
promoted standard curriculum, four persistent banks, four value-diverse latent
strategy slots, context learning rate 0.5, soft perturbation 1.2, temperature
0.08, and sixteen cost-free latent direction proposals.

1. Run every clone to round 18 and perform the four-seed read-only shadow audit.
2. Rank clones by descending minimum context-mean best-slot reward advantage.
   Break ties by descending number of audit seeds whose best slot differs
   between old and reliability contexts, then by ascending clone ID. Advance
   exactly two clones.
3. Resume both survivors to round 42. Rank them by descending mean verified
   learned-minus-frozen reward over the first six genuine `old_return` rounds.
   Break ties by descending worst single-round reward advantage, then by
   ascending clone ID. Advance exactly one clone.
4. Resume the winner to round 54.
5. For this first prospective validation only, also complete fixed clone 7130
   to round 54. This is an audit control, not part of the production compute
   budget.

No accuracy, target label, hidden state, or semantic rule enters either
selector. Prefixes remain non-graduating. Exact prefix continuation must
preserve all prior parity and accounting gates.

## Promotion criteria

The ladder is promising if:

- the selected winner beats the fixed control on reliability acquisition and
  old-return performance without violating inherited binary/four-rule
  retention;
- the selected trace remains causal under the existing reward and
  strategy-key interventions if the result is unusually strong;
- exact unique-experience and verifier-bit accounting remains separable from
  read-only shadow selection;
- production training compute is lower than four uninterrupted 54-round runs.

All negative claims remain bounded to this one fresh physical stream and this
four-clone population.

## Results

### Round-18 acquisition screen

| Clone | Conservative shadow advantage | Specializing audit seeds | Decision |
|---|---:|---:|---|
| 7130 | 0.0000 points | 0/4 | stop |
| 7131 | **+2.08333 points** | 0/4 | advance |
| 7132 | 0.0000 points | 0/4 | stop |
| 7133 | +2.083331 points | 0/4 | advance |

The difference between 7131 and 7133 was only floating-point scale and did not
matter because both advanced. No specialization tie-break was used.

### Round-42 retention screen

The pre-registered first-six-return-round reward differences were:

- 7131: `[0, +4.17, +4.17, 0, 0, 0]` points; mean **+1.389**,
  worst 0;
- 7133: `[-4.17, -4.17, -8.33, -4.17, -4.17, -4.17]` points;
  mean **-4.861**, worst -8.33.

Clone 7131 advanced without a tie-break.

### Completed winner versus fixed control

| Measurement | Selected 7131 | Fixed 7130 |
|---|---:|---:|
| Reliability target accuracy | **18.06%** | 0.00% |
| Frozen reliability accuracy | 5.56% | 0.00% |
| Reliability target advantage | **+12.50 points** | 0.00 points |
| Old-return target accuracy | **13.89%** | 0.00% |
| Frozen old-return accuracy | 5.56% | 0.00% |
| Old-return target advantage | **+8.33 points** | 0.00 points |
| Old-return reward advantage | **+0.463 points** | 0.00 points |
| Binary retention | pass | pass |
| Four-rule retention | pass | pass |
| Full gate | pass | pass |

Every resumed report preserved every earlier trace row exactly.

### Adversarial controls

The strong selected seed justified two full causal controls:

- shuffling physical verifier-reward alignment made the full gate fail and
  reduced both reliability and return target advantage to zero;
- shuffling latent strategy keys immediately before transfer reduced both
  reliability and return target advantage to zero.

The winner therefore depends on correctly aligned verifier outcomes and on
the learned association between strategy values and their latent addresses.

## Compute accounting

The production ladder executed:

- `4 × 18 = 72` acquisition rounds;
- `2 × 24 = 48` post-acquisition rounds;
- `1 × 12 = 12` final rounds;
- total: **132 physical training rounds**.

Four uninterrupted clones would cost 216 rounds, so the production ladder
saved **38.9%** of physical training compute. Including the one-time fixed
control completion cost 168 rounds and still saved **22.2%**. The four
round-18 audits separately consumed 448 held-out logical lifetimes and 3,840
selection verifier bits, with no optimizer updates.

## Verdict

Promoted provisionally. On its first fresh prospective stream, the two-stage
selector preserved a useful acquisition-and-retention trajectory, rejected a
harmful return trajectory, saved substantial training compute, retained both
mastered primitives, and survived causal intervention checks.

The magnitude is smaller than the exceptional stream-7075 result, so the
correct claim is not that the ladder always finds a spectacular learner. The
supported claim is that this ladder is a compute-efficient way to preserve a
better member of a small stochastic population. Replicate the complete ladder
on another fresh stream before increasing population size or allowing the
selector to influence larger training budgets.
