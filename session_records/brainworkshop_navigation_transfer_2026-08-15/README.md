# Actions that change what happens next (2026-08-15)

Status: **diagnostic**, on the already-consumed development seed. Nothing
admitted; `AgentBrain.bank` unchanged at `07319eb1`.

Every environment in this repository generates its symbol stream in advance.
Whatever the agent answers, the next observation is the same — so an action has
no consequence, there is nothing to plan, and "agent" has been a courtesy title
for a classifier reading a tape.

Here the tape is gone. The agent occupies one of the rendered grid places, an
action moves it, and what it sees next is where it went. Reward arrives only
when it stands on a place nobody named.

## The approach, and why it is a claim rather than an implementation detail

Delayed reward is usually attacked head-on: try policies, see what pays,
propagate credit backwards. That throws away what this world hands over for
free — the agent **sees where it ended up**, so the dynamics are fully
observed. `(place, action) → next place` is supervision, not a bandit signal.

So the delayed reward stops being a credit-assignment problem and becomes a
**search** problem over a model the agent builds directly. That is only worth
its complexity if the same experience spent guessing does worse, which is what
the model-free control is for.

## Result

Twelve sampled worlds — eight places, four actions, dynamics drawn per task, a
hidden goal at least two moves from the start. Twelve exploration episodes of
twenty-four steps each.

| | |
| --- | ---: |
| Goal discovered by stumbling on it | **12/12** |
| ...and nothing else mistaken for it | **12/12** |
| Plans that are the true shortest route | **12/12** |
| Return achieved | **0.955** |
| Best return available | 0.955 |
| Model-free, same experience | 0.556 |
| Random wandering | 0.171 |
| Rewards shuffled | 0.156 |
| **From starts it never began an episode at** | **0.981** |
| ...best available from those starts | 0.981 |

**It plans the exact shortest route in every world and achieves the optimal
return to the digit.** The held-out-starts row is the one that separates the
two ways of appearing to have learned: dropped at places it never departed
from, it is still optimal. A memorised trajectory cannot do that; a model can.

## Where the advantage actually lives

Sweeping the exploration budget, with the model-free control given **ten times**
the experience:

| explore episodes | coverage | planned | model-free (same) | model-free (10×) | random |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.49 | **0.851** | 0.056 | 0.507 | 0.171 |
| 2 | 0.68 | **0.885** | 0.128 | 0.726 | 0.171 |
| 4 | 0.88 | **0.913** | 0.323 | 0.865 | 0.171 |
| 8 | 0.95 | 0.917 | 0.465 | **0.955** | 0.171 |
| 12 | 0.98 | **0.955** | 0.556 | 0.955 | 0.171 |

**The model-based advantage is a sample-efficiency advantage and it is largest
where experience is scarcest.** One exploration episode buys 0.851; policy
guessing needs ten episodes to reach 0.507 and eight to catch up at all. By
eight episodes the two meet, which is the honest shape of it — in a world this
small, enough guessing eventually works.

Coverage is what everything tracks. The model knows which cells it has never
tried and the planner refuses to route through them, so a thin model produces
short honest plans rather than confident wrong ones.

## Controls

**Model-free, matched and 10×.** Random reactive policies scored on the same
episodes. A weak searcher, which is why it is also given ten times the budget.

**Random wandering.** The policy exploration itself used: 0.171 throughout.

**Shuffled rewards.** Where the reward arrived is destroyed while how often it
arrived is kept. The planner still runs and still has somewhere to go; it is
just not the goal. 0.156.

**Held-out starts.** Every reachable place the agent never began from.

**Stimulus noise.** Identical results at 0.00, 0.05 and 0.10 — not a plumbing
failure but the correct outcome, since clustering is robust enough that the
agent's experience is unchanged step for step. At 0.20 the alphabet is refused
outright, which is **the same boundary** `brainworkshop_environment_widening`
found for the n-back tasks.

## Three things I got wrong, and how they announced themselves

**The optimum was off by one.** Reward is paid on the step the agent *arrives*,
so a goal `d` moves away pays on steps `d..steps` — `steps - d + 1` paying
steps, not `steps - d`. The agent duly "beat" the optimal return, which is how
it surfaced.

**Cluster indices are not place indices.** The agent works entirely in clusters
and never noticed; the *record* compared the two namespaces directly and
reported the goal as never found while the agent was reliably standing on it,
and one-step plans for two-step journeys. This is the third time this session
that a permutation between the frontend's names and the verifier's has produced
a confidently wrong number.

**Places cannot be clustered from one look each.** The first version rendered a
catalogue of eight frames; the tolerance estimator refused, correctly — with
one observation per place there is no within-place mode, so there is no
boundary to estimate. Discovery now walks the world like every other task.

## What is honestly weak

**The world is small and fully observed.** Eight places, four actions, thirty-two
cells. Random walks cover it, which is why enough guessing eventually competes.
Nothing here shows the approach scaling to a world where coverage is the hard
part.

**Every place looks different.** There is no perceptual aliasing, so the agent
never needs memory to know where it is — the Mealy machinery that carries state
elsewhere in this repository does nothing here. Aliased places are the obvious
next axis and would make the model a genuine state-estimation problem.

**The dynamics are deterministic.** Counts are kept and majorities taken, so a
mis-clustered reading is outvoted, but nothing here is tested against a world
that is actually stochastic.

**One goal, no goal-conditioning.** The agent seeks whatever paid. It cannot be
asked for a *different* goal, and nothing tests whether one model serves many
goals — which is most of what a model is supposed to be for.

**None of this is in the library.** The model is a table built and discarded
per task; it is not compiled, not admitted, and not composed. The accumulation
machinery and the navigation machinery have not met.

**Twelve worlds, one seed, no holdout.**

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.navigation_transfer
```

About one second.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_navigation_transfer.py -q
```
