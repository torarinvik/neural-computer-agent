# Learning survives one label in five being wrong (2026-08-15)

Status: **diagnostic**, adversarially calibrated. Nothing admitted;
`AgentBrain.bank` unchanged at `07319eb1`.

Noise was the deepest flaw left standing. Every result in this session held
only at zero label noise: `build_tree` raised on the first contradicted prefix,
the exact search either abstained or returned an eleven-state machine at chance
accuracy, and an agent reading rewards from any real verifier would have
crashed on its first mis-scored tick.

## Three failures, and what they were each telling us

**Statistical merging (ALERGIA).** Compare output proportions with a Hoeffding
bound. At the counts a short-episode prefix tree produces the bound is vacuous
-- most cells are seen once or not at all -- so everything looks compatible and
a four-state threshold rule came back as **one state at zero noise**.

**A violation budget in the exact search.** Let a disagreement be spent rather
than be fatal. A violation may be spent anywhere, so the branching multiplied
and the search stopped finishing: the same rule went from identified to
nothing.

**Prefix-level majority voting.** Outvote the noise where the evidence is thick
and drop the rest. It dropped **84%** of it, and with only shallow constraints
left it identified nothing even at zero noise -- threshold-3 fell to **0.02**.

The third failure is the informative one. The unit that recurs often enough to
vote on is not the prefix, it is the **state**. A four-state machine over 5376
labelled steps visits each of its sixteen cells hundreds of times; prefixes at
that depth are visited once. But states are what noise stops us from learning,
so voting per state needs the machine and the machine needs the vote.

## What works: score machines, do not derive them

Stop treating the machine as something to be derived from consistency and start
treating it as something to be **scored**: pick the machine that disagrees with
the fewest labels. That objective is defined for every machine, requires no
consistency, and degrades smoothly.

Two things make it cheap. Given transitions, the output table follows by
majority per cell -- no search -- so only transitions are searched over. And
the search is local: change one transition, keep it if disagreements fall,
repeat to a local minimum, restart. Model size is chosen by description length,
because fewest-disagreements alone always prefers the largest machine offered.

## Result

Twenty-four sampled Mealy rules of one to six states, 112 episodes of 48 steps,
scored against the **clean** rule so that reproducing corrupted evidence does
not count:

| Label noise | exact recovery | mean accuracy | correct state count |
| ---: | ---: | ---: | ---: |
| 0% | 24/24 | 1.0000 | 24/24 |
| 2% | 24/24 | 1.0000 | 24/24 |
| 5% | 24/24 | 1.0000 | 24/24 |
| 10% | 24/24 | 1.0000 | 24/24 |
| **20%** | **24/24** | **1.0000** | **24/24** |
| 30% | 22/24 | 0.9868 | 23/24 |

**One label in five flipped, and every rule is still recovered exactly.**
Against a stack that crashed at 2% and returned chance-accuracy garbage at
0.5%.

The fit's own error rate is a calibrated noise estimate -- 0.0195 at 2%, 0.0519
at 5%, 0.1049 at 10% -- so the learner can report how dirty its evidence was
without being told.

## Adversarial checks

**Search seeds.** At 10% noise, 24/24 at each of four independent restart
seeds. Not a lucky local minimum.

**Data seeds.** 24/24, 24/24, 23/24 across three draws of the episodes.

**Data budget** at 10% noise, and it degrades gracefully rather than falling
over: 16/24 at 336 labels, 19/24 at 672, 21/24 at 1344, 23/24 at 2688, 24/24 at
5376.

**Structureless data.** Random labels over random symbols: it returns a
two-state machine with a fit error of **0.476**, which is chance. It does not
dress noise up as a rule, and its own report says so.

**Out-of-class targets.** Running majority is not finite-state at any size; the
fit lands at 0.46 to 0.59 and never claims exactness. The state count *shrinks*
as noise rises (7 to 5) rather than exploding to fit the noise.

## What is not claimed

**This does not beat Gold's ceiling.** On the same zero-noise evidence the
exact search also recovers 4/4 at every state count from one to six, and does
it faster (7.9s against 18.7s at six states). The earlier record that five- and
six-state rules were unidentifiable was about a thinner data regime, not a
fundamental limit. The advantage here is **entirely** noise tolerance.

**Rare positives defeat the model-size criterion.** A threshold-6 rule presses
so seldom that a one-state machine that never presses has low error, and
description length prefers it. Class imbalance is unhandled.

**Non-stationary targets are still unlearnable.** A rule that switches halfway
through each episode fits at 0.75 with a fit error near 0.20 that does not fall
with more data -- honest, and no better than before.

**Nothing is admitted, and the controller executes none of this.**

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_noise_tolerant_induction.py -q
```

About one minute. The calibration sweeps are `calibration.txt` in this record.
