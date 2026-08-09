# Architecture and goals

The precise statement, current as of 2026-08-09. Every claim here is
either marked with the evidence that establishes it or marked open.
Nothing is asserted from design intent alone — this project's recurring
failure mode is a plausible story built on an unvalidated measurement,
so the standard is: if it has no finding number, it is a hypothesis.

---

## 1. The goal

Build a controller of **fixed size** that can keep acquiring new
competences indefinitely, without forgetting old ones and without
replaying old data.

The mechanism is a storage rule, and it is the whole thesis:

> **Task-specific content lives in an external, growing memory bank.
> The controller holds only what is common to every task.**

If that holds, capability grows by adding entries rather than by growing
or overwriting the network, and forgetting stops being a thing to fight
because there is nothing task-specific in the weights to overwrite.

### What "task-specific" means, precisely

The distinction that matters is not "hard vs easy" but **shared vs
contradictory**. Two tasks that want the same thing can share machinery.
Two tasks that want opposite things on identical observations cannot —
their compromise is chance, measured (F50). So:

- **Shared across tasks → plant.** Geography, dynamics, how to pursue a
  goal, how to search. This is not a violation of the storage rule; it
  is not skill, it is infrastructure.
- **Differs or contradicts across tasks → bank.** What to want, in this
  world, right now.

---

## 2. The architecture

Three parts, and the interfaces between them are the design.

### 2.1 Plant (fixed, shared, task-invariant)

A small recurrent controller with amodal encoders and decoders. It
receives a state, receives a **goal**, and acts. It holds:

- the transition model / map of how the world works,
- the ability to pursue **any** goal expressible in its goal vocabulary,
- (planned) the search procedure that turns a distant goal into a next
  action.

It must hold **no** task content. This is the invariant that every
failure in this project has violated in a new way (see §4).

### 2.2 Bank (growing, external, per-task)

A set of **fragments**. A fragment names a goal — under the current
formulation, a **predicate over states**: the set of states that count
as success. A singleton predicate is "reach cell (3,4)"; a large one is
"any position where the king is mated". Distance to a set is the
distance to its nearest member, so the same machinery serves both.

Fragments are small, per-task, and composable. They are the only thing
verifier reward is allowed to shape.

### 2.3 Addressing (which fragment, now)

Two validated routes:

- **Cued**: a label rendered in the world, read like any other percept,
  selects the fragment. Verified causal — swapping the label swaps the
  behaviour (F57).
- **Probe**: for worlds that render identically, act first and read the
  consequence. Identity is only knowable from consequences (F44); the
  probe must therefore run mid-episode, and its outcome must not become
  a second context channel (F53, F-leak).

---

## 3. The objective

Not "perform well on task T" but:

> **Produce a program such that having learned task A makes a NOVEL task
> B faster to learn than from scratch.**

This is the objective and the measurement at once — there is no gap
between the claim and its test, which is where this project has
repeatedly gone wrong. "From scratch" is the only baseline that cannot
be gamed: a system that learns each task independently and never forgets
scores exactly zero against it, so the objective is unsatisfiable
without genuine reuse. The load-bearing word is NOVEL — if B shares a
family with A, "faster" is nearly free and means little.

Operationally, for a reacher: from any state X, reach any goal G at
least cost, and get cheaper at it as more tasks are seen.

Three consequences, in order of how well established they are:

1. **Goals must be plentiful and varied.** With few goals, ignoring the
   goal channel is competitive, and under isolation it is *optimal* — so
   the plant learns an unconditional habit and never reads its
   instruction. Measured across every optimiser-side fix (F58). A goal
   space too large to memorise makes reading the instruction the only
   representable solution.
2. **Cost belongs in the objective.** A system rewarded only for success
   will relearn everything from scratch; one rewarded for *cheap*
   success must check its library first. Efficiency is therefore what
   produces reuse, and reuse is what produces abstraction. **Not yet
   implemented** — this is the top open item.
3. **The accounting must be lifetime, not immediate.** Building a
   reusable abstraction costs more today and less forever. Minimising
   current cost alone selects for lookup tables — which is exactly the
   failure in (1). **Not yet implemented.**

### The measurement that decides it

Every rung so far has measured *retention* — did task 3 survive task 4.
That is the weak claim. The strong one is:

> **acquisition cost for task N falls as N grows.**

Flat curve = a library of separate entries. Downward = a map. This is
the headline gate and it is falsifiable in a way "we did not forget"
never was.

---

## 4. What is established

| claim | evidence |
| --- | --- |
| The bank carries the skill and is NECESSARY — remove or replace it with norm-matched noise and performance falls to the measured floor | F54, 3 seeds, two independent gates |
| Skills COMPOSE: held-out task combinations assembled from trained fragments with zero learning, at 85-111% of trained performance; scrambled control at floor | F-composition, `goal_composition_v1_2026-08-09` |
| A world-label rendered in the world causally drives fetch (label-swap collapses behaviour) | F57, 5/8 seeds clear every gate |
| Goal-conditioned reaching works, at provably near-optimal path length | F59: numeric 0.938 reach, path ratio 1.010 |
| The 2D walled grid is NOT intrinsically hard | ladder: r1-r4 all master, r4 0.996 @150 updates |
| Consolidation retires catastrophic forgetting in a plastic core | promoted EWC rung, 5/5 seeds |

## 5. What is open

1. **Cross-domain transfer is NEGATIVE, and freezing the plant does not
   fix it (F61).** Warm-starting the walled grid from the line: 0.277
   trainable, 0.211 frozen-plus-adapter, against 0.996 cold. The adapter
   re-maps what the goal means but cannot change how the plant pursues
   goals — and a plant trained on a line has learned "hold one
   direction" as its pursuit POLICY, which is wrong in 2D. So the plant
   absorbs domain-specific control, not merely style, and §2.1's
   assumption that goal-pursuit is the legitimately shared part is
   measured false across unrelated domains. Two untested routes: train
   on diverse domains concurrently so nothing can become the prior, or
   hold only the transition model and DERIVE the policy by search, so
   there is no learned controller to carry habits at all.
2. **Cost is not in the objective** (§3.2, §3.3), so nothing yet
   pressures reuse.
2b. **Positive transfer is NOT demonstrated, on cheap or expensive
   targets (F62, F63).** Concurrent multi-domain warm-start recovers
   most of F61's damage but never exceeds cold start: 0.824 vs 1.000 on
   a cheap target, 0.613 vs 0.625 on a sparse-reward target expensive
   enough that cold never masters. Single-domain priors are harmful in
   all three tests. **Diversity converts harmful priors into neutral
   ones; nothing so far converts them into helpful ones.** Refined by
   F64: a multi-domain plant, FROZEN, reaches 0.520 zero-shot on a novel
   walled grid against a 0.172 floor — real transferred competence, with
   no gradient steps on the target. But it does not become speed (0.613
   vs 0.625 cold at matched budget). **Prior learning transfers
   capability but not learnability: the architecture stores and reuses,
   it does not yet compound.** The bottleneck is now the adaptation
   channel — a 4160-param goal adapter can re-map what a goal means but
   not what the frozen policy does with it — not the plant's knowledge.
   *Closed by F65/F66:* widening the channel changes nothing (0.613
   both), and freezing the plant merely MOVES forgetting into the
   adapter — across three sequential targets the frozen arm's zero-shot
   DECLINES (0.531 -> 0.129) while a cold arm stays flat (0.422 ->
   0.363), costs 60% more, and ends worse. Freezing relocates the
   storage-rule violation to whatever is still plastic. One untried
   mechanism remains: a plant holding NO policy, with behaviour derived
   by search over a learned model.
3. **Retrieval is never tried before learning.** The bank is only ever
   *given* fragments; it is never asked whether it already contains a
   solution. Without that loop, reuse cannot be selected even if it is
   cheaper.
4. **Task diversity is cosmetic, not structural.** The game family
   shares dynamics by construction, so it cannot test which level of
   abstraction is the shared one. Numeric-vs-grid is currently the only
   genuine cross-structure pair we have.
5. **Adversaries and irreversibility** are outside the formulation.
   Shortest-path assumes you control the transitions and can retry;
   chess and our intercept games violate both.

## 6. The recurring failure, and the rule it produced

One pathology has appeared at four levels: the plant absorbing content
that belongs in the bank. F11 (a default context kept in weights), F50
(one twin taking the plant), F58 (an unconditional habit instead of
reading the goal), and now cross-domain habit transfer. **Every
architectural failure in this project so far is the storage rule being
violated somewhere new.**

A second pattern is methodological and just as consistent. Across this
session, eight failures resolved into the *measurement or the task*
rather than the learner: a gate the harness passed on the agent's behalf
(F53), scores with no measured floor (F51/F52), two battery games unable
to discriminate at all (F52), a Manhattan metric in a walled map, an
unreachable-target sentinel leaking into a mean, a dead-detector
self-reward, a broadcast bug severing the goal channel, and an entire
"the grid is hard" narrative that the ladder refuted in one run.

The rules that came out of it, now standing:

- **Run every gate with no agent at all, first, and confirm it fails**
  (weakness 18). This caught more bugs today than any other practice.
- **Report every score against a measured floor**, never against zero
  (F52).
- **Re-measure any signal at its point of use** (F46).
- **A threshold is part of the measurement** and must be calibrated on
  the curves it will judge.
- **An intervention, not a correlation, settles a causal claim** — the
  position-decodability story looked strong and its own fix refuted it.
- **Check process count, not file existence**, before believing a run is
  progressing.
