# When being wrong stops naming the answer (2026-08-15)

Status: **diagnostic**, on the already-consumed development seed. Nothing
admitted; `AgentBrain.bank` unchanged at `07319eb1`.

Every learning result in this session rests on one line:

```
target[t] = action[t] if reward[t] else 1 - action[t]
```

Feedback inversion is exact, and it is exact **because there are two actions**.
With two, being wrong names the right answer as surely as being right does, so
a scalar reward is secretly a full supervision signal — and noise tolerance,
the library, recognition and composition were all doing supervised learning in
a bandit's clothes. Nothing ever tested a third action, so nothing ever
revealed the dependence.

This is the first rung off that.

## What changes with `k` actions

A success still names the target. A failure rules out one of `k` and leaves
`k-1` standing. The evidence becomes **partial**, most of it negative, and the
learner has to *choose what to try* — which under two actions was a question
with no content.

The objective generalises without becoming a search. Given transitions, each
cell of the output table is still filled by counting: a candidate action `a`
disagrees with a success that chose something else, and with a failure that
chose `a`. Minimising is `argmin(neg[a] - pos[a])` per cell, one pass. At two
actions this is exactly the majority rule the binary fitter already used, so
nothing earlier is disturbed. A correct machine has **zero** disagreements,
because a failure it predicted would fail is evidence *for* it.

## Result

Sixteen sampled rules — action counts 2 to 5, state counts 1 to 4 — every one
held to a best-constant-answer rate of at most 0.6 so no task is clearable by a
fixed reply. Budgets are prefixes of one draw, so a curve runs through one
world rather than several.

**Rules identified exactly (uniform exploration / fixed policy), of 4 per cell:**

| actions | 4 eps | 8 | 16 | 28 | 56 |
| ---: | :--: | :--: | :--: | :--: | :--: |
| 2 | 3/3 | 2/2 | 4/4 | 4/4 | **4/4** |
| 3 | 2/0 | 2/0 | 4/0 | 4/0 | **4/0** |
| 4 | 1/0 | 2/0 | 3/0 | 4/0 | **4/0** |
| 5 | 2/0 | 2/0 | 4/0 | 4/0 | **4/0** |

**Exploration is exactly as load-bearing as the theory says.** At two actions a
policy that never varies identifies every rule — the reward names the target
regardless. At three or more it identifies **none**, at any budget, while
uniform exploration identifies all of them.

**Information per step falls as 1/k, and it is measured rather than argued.**
The fraction of steps where the target is known outright rather than merely
constrained:

| actions | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: |
| resolved fraction | 0.517 | 0.319 | 0.259 | 0.191 |
| 1/k | 0.500 | 0.333 | 0.250 | 0.200 |

## What did not happen, and is worth saying

**The cost did not blow up with `k`.** I expected the evidence budget to scale
with the action set, since each step carries roughly `1/k` of the information.
It does not: identification lands at 16–28 episodes at every action count. Two
effects cancel — a success gets rarer, and a wrong hypothesis gets easier to
contradict, because a random wrong answer agrees less often when there are more
ways to disagree.

## Controls

**Fixed policy.** Always answer 0. Reported at every cell above; it is the
control that separates learning from exploring.

**Shuffled rewards.** Permuting each episode's rewards keeps how often the
agent was right and destroys which choices it was right about: **1 of 80** fits
reach the 0.8 gate, 1 of 80 identifies.

**Best constant answer.** Every rule is sampled under a cap of 0.6 and every
accuracy is reported against it, so nothing is credited for replying the same
thing every time.

## A flaw in the first version of this record

The constant-answer cap was applied only above two actions. Binary rules came
in under the historical press-rate window, which admits constants as high as
**0.835**, while three-action rules were held to 0.6 — so the binary column was
not comparable with the rest, and the shuffled control looked four times worse
than it is (19 of 80 "beating the constant", against 1 of 80 solving anything
once the sampler was made consistent).

## It is an agent result, not only a fitter one

Three things had to widen for a `k`-action capability to be compiled, kept and
found again, and each is additive so nothing binary moves:

- the **counter ABI** reserved counter zero for a press; it now uses one
  counter per action and reads the largest. Verified to reproduce its machine
  exactly at every action count 2–6 and state count 1–5;
- the **library record** carries an action count, written to disk only when it
  is not two — so all six libraries already committed load, and digest,
  byte-identically;
- **recognition** cannot test a consistency rate directly, because that rate's
  floor climbs with the action set: a machine answering at chance is consistent
  with 0.500 of outcomes at two actions and **0.625** at four, so a 0.8 bar
  would wave a coin flip through. Under a uniform probe the relation inverts
  exactly, `a = (c·k − (k − 2)) / 2`, which returns chance at chance for every
  `k`; the trial count is then discounted by `(k/2)²` for the noise inversion
  adds.

A stream of nine tasks drawn with repeats from three rules, growing library
against matched control:

| actions | tasks solved | recognised | admitted | false recognitions | acquisition ratio |
| ---: | :--: | ---: | ---: | ---: | ---: |
| 2 | 9/9 | 6 | 3 | 0 | **0.500** |
| 3 | 9/9 | 6 | 3 | 0 | **0.600** |
| 4 | 9/9 | 5 | 4 | 0 | **0.909** |

**The library still pays, and its benefit erodes as the action set grows** —
which is the honest consequence of the same information argument. Recognising a
stored program from partial feedback needs `(k/2)²` times the evidence, so by
four actions retrieval is barely cheaper than relearning.

## What is honestly weak

**Composition does not extend to `k` actions as written.** The library composes
with `and`, `or` and `xor`, which are binary operations. What it means to
combine two three-action programs is an open design question, not plumbing.

**The probe policy is uniform and unexamined.** It is the policy whose cost can
be reasoned about, not one chosen for being good. An agent that spent its
guesses where it was most uncertain should need less evidence, and nothing here
measures that.

**Actions still do not change the world.** Each tick is scored independently;
the agent's answer never affects what it sees next. Multi-step actions,
planning, goals and objects are all still untouched — this rung is only about
the answer no longer being one bit.

**Four rules per cell, one seed, no holdout.**

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.choice_ceiling
```

About fifty seconds.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_choice_induction.py -q
```
