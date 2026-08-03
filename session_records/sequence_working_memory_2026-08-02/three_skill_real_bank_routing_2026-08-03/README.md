# Three-skill real-artifact bank routing (2026-08-03)

## Question

Can a reward-trained selector choose among three externally stored skills using
only controller-produced query context, opaque row keys, attempted actions, and
scalar outcomes, then reproduce the selected skill after a bank save/reload?

This is a routing diagnostic, not a new span-eleven acquisition. The span-nine,
span-ten, and span-eleven rows were built from real checkpoint artifacts relative
to the common frozen parent
`artifacts/checkpoints/span8_addressed_parent_scale1_seed32001.pt`.
The span-eleven artifact is retained as a routing row only: its acquisition was
previously rejected as below the 90% mastery bar.

## Results

The 1,024-update arm passed every pre-registered diagnostic gate, and an
independent replica with a different seed reproduced it:

| gate | seed 93422 | seed 93423 |
|---|---:|---:|
| reward-trained routing accuracy | 100.0% | 100.0% |
| reward-shuffled control | 33.33% | 33.33% |
| cosine-only baseline | 33.33% | 33.33% |
| candidate-permutation audit | 100.0% | 100.0% |
| bank save/reload exactness | pass | pass |
| all held-out queries routed correctly | pass | pass |

Each run used 32 train and 32 held-out queries per skill, a batch size of 64,
256 behavior episodes, and 65,536 verifier bits. The 256- and 512-update arms
did not pass the routing assertion. Those are bounded, data/optimization-budget
negatives; they do not overturn the 1,024-update result.

## Behavioral control

For every row, routed behavior exactly matched direct rehydration. Across the
two replicas, direct/routed accuracy was:

- span nine: 92.06% and 92.97%;
- span ten: 86.33% and 87.70%;
- span eleven: 81.68% and 80.61%.

The span-eleven number is reported for routing only, not as a mastery claim.
The wrong-skill control is useful as a stress test, but it is not required to be
lower on every finite sample: span-ten's wrong-row sample was slightly higher
than its correct-row sample in the first replica. The decisive evidence is
reward dependence, exact reload, candidate permutation, and direct/routed parity.

## Interpretation

This is the strongest current evidence that a disk-backed skill bank can hold
multiple acquired skills without forcing every skill into one always-on residual
or permanently perturbing older behavior. It supports the architecture direction
`controller + hot working set + cold disk skill bank + learned selector`.

It does **not** yet establish:

1. multi-skill routing learned from a genuinely cold start;
2. behavioral acquisition of a new skill after a cold reload;
3. long-horizon retention under repeated additions; or
4. span-eleven mastery.

No production default or checkpoint was changed, and no checkpoint was promoted
from this diagnostic. The next high-ROI rung is a fresh-process replacement
sequence with the same routing, retention, corruption, and reversal audits,
followed by a separately gated span-eleven acquisition or an easier intermediate
primitive.

## Reports

- `nca_three_skill_bank_1024.json`
- `nca_three_skill_bank_1024_replica.json`

The command was `audit_sequence_reward_routed_bank.py` with the three real child
artifacts listed above, 1,024 selector updates, and MPS execution.
