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

**Holds a MODEL, not a policy (F67).** Six findings were one failure —
a stored policy going stale — and the fix is to store no policy. The
plant learns dynamics; behaviour is computed by search at act time.

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
3. **The accounting must be lifetime, not immediate.** ~~Building a
   reusable abstraction costs more today and less forever.~~
   **RETIRED 2026-08-09** — see the gate definition below. The intuition
   was right and the accounting rule drawn from it was wrong: charging
   pre-training against the per-task saving double-counts, since the
   pre-training IS the prior task the objective says should help. What
   survives from this point is its real content, now clause (c) of the
   primary gate: per-task cost must not DRIFT UPWARD as the bank grows.
   That is the anti-lookup-table condition (1) needs, and unlike
   break-even it is measurable without extrapolation.

### The measurement that decides it

Every rung so far has measured *retention* — did task 3 survive task 4.
That is the weak claim. The strong one is:

> **acquisition cost for task N falls as N grows.**

**Gate definition, settled 2026-08-09.** Cost was the primary objective
through F70-F82 and it earned its place: it is falsifiable, it caught
the nesting artefact (F71), and it stopped F75 and F76 from being called
wins. But it acquired a second, illegitimate form along the way —
LIFETIME break-even, which charges pre-training against the per-task
saving and asks when the system repays its own education.

That framing is retired, because it double-counts. The founding
objective is:

> Produce a program such that given task A makes novel task B faster to
> learn than chance or starting from scratch.

It says prior experience must make new tasks cheaper. It does not say
the prior experience must be free. Pre-training IS task A. Demanding
that the saving also repay the cost of acquiring the prior is asking the
system to pay for the same education twice, and it sent this project
down an axis (F80/F82) that turned out to have an interior optimum and
nothing beyond it.

**PRIMARY gate — capability under exact retention.** As the bank grows
over N novel families:

  (a) the set of families acquired to mastery keeps growing;
  (b) retention of every earlier entry stays exact;
  (c) per-family acquisition cost does not drift upward with bank size.

(c) is what makes this a real gate rather than a restatement of "we
stored things": if cost grows with the bank, the architecture does not
scale and no amount of per-task cheapness rescues it.

**SECONDARY regularizer — cost.** Among systems that pass the primary
gate, prefer lower per-family acquisition cost. Lifetime totals and
break-even are reported as DIAGNOSTICS and never as pass/fail.

**Status (F83, 64 families acquired sequentially through one frozen
plant, 2 seeds).**

- **(a) MET.** 64/64 mastered on both seeds; 59/64 and 56/64 by reading
  alone, zero gradient steps.
- **(b) MET.** Retention drift max 0.0, mean 0.0, across the whole grown
  bank. The skill is in the bank, not the weights, by full double
  dissociation (present 0.907 / withheld 0.236 / corrupted 0.037, F81).
- **(c) PASSES, WEAKLY — and the weakness is the next work.** Cost does
  not drift (4.9 vs cold 50.6 overall, saving +41 to +52 in every
  quartile, correlation with position +0.200/+0.092). But entry i+1 is
  fitted without ever seeing entries 0..i: the plant is frozen and
  entries are independent tensors, so nothing in this implementation
  COULD make cost grow with bank size. **A gate that cannot fail is not
  evidence.**

**ALL THREE CLAUSES PASS (F86).** Content addressing closed F85's O(N)
failure: keys identify the right entry among 64 at **1.000 with zero
plant forward passes**, and retrieve-then-verify holds 1.000 at a
CONSTANT 4 passes. Recognising a known task is now cheaper than
relearning it and stays so as the bank grows.

The two addressing routes in §2.3 turn out to be complementary, not
alternatives, and the measurement says why:

- **keys ADDRESS** — 0 plant passes, perfect shortlist, but a stranger
  still matches its nearest key at 0.862, so keys alone cannot say
  "none of these" and key-only reuse would reuse constantly;
- **consequence VERIFIES** — 4 plant passes, constant in N, and it
  supplies exactly the "none of these" that reuse-or-mint requires.

This also implements §3 consequence 2 — "cost belongs in the objective
so the system checks its library first" — which has been marked *not yet
implemented* since this document was written. Checking the library is
now cheaper than relearning, measured.

**Current frontier (F87, F91).** The gate passes at N=256: 256/256
mastered, retention drift 0.0, acquisition flat across a 4x-larger bank,
retrieve-then-verify 0.994 at a constant 4 plant passes. The last
capability gap is closed — `toggle` went 0.096 to 0.917 by combining a
wider op vocabulary with slot-count balancing.

Two rules this architecture now supports, both measured twice:

1. **The bank interface should be as narrow as the task allows.** Adding
   conditioning capacity by modulation (F77) or by token count (F89)
   both left in-distribution accuracy unchanged and made the hardest
   families worse.
2. **Distribution shifts within a fixed schema trade off; a better
   schema raises the ceiling for everyone.** Balancing slot counts alone
   bought wide families by taking from narrow (F90); adding the missing
   op primitive recovered the narrow loss and kept the gain (F91).

**Measured to N=1024 (F93, F94).** 1024/1024 families mastered,
retention drift exactly 0.0, acquisition ~1.2x higher across a
1024-entry bank while staying 5-10x cheaper than cold. Retrieve-then-
verify retrieves at 0.980 on a CONSTANT 4 plant passes where a
1024-pass linear scan has fallen to 0.853.

Key-only discrimination is effectively gone at that scale — a
never-seen family matches its nearest stored key at 0.954 — so the
verify step's importance grows with bank size rather than shrinking.
Consequence verification is the only remaining source of "none of
these" (gap 0.171).

**The ceiling, now exact (F92, F95, F96): the bank stores RULES and
cannot store EXCEPTIONS.** The reacher's open grid is read perfectly at
zero gradient steps; its WALLED variant reads 0.894 at every seed and
every budget. That number is not a partial success — `grid` and `walled`
agree on exactly 229/256 = 0.8945 of transitions, so the reader gets
every non-wall transition right and every wall transition wrong. It
reads "8x8 grid movement" and ignores the obstacle entirely.

Three candidate fixes were tried and all three are refuted: more budget
(80000 updates repaired every other family and left this one at 0.894),
a conditional op primitive (made everything worse), and more entry
capacity (F77, F89). The obstacle is ~121 bits of arbitrary,
incompressible content; an entry read by a plant applying uniform
functions of slot values can only ever name a rule.

**This is §2.2 splitting in two, from measurement:**

- **rules** — compressible, apply everywhere, live in bank entries. The
  mechanism built in F76-F96: 0.982 read, 3.0x cheaper acquisition than
  cold, retention exact to N=1024.
- **exceptions** — arbitrary, per-state, incompressible. They need a
  STORE rather than a rule: content-addressed episodic memory holding
  (state, action) -> outcome where the rule fails, consulted first.

**BUILT AND MEASURED (F97).** `walled` goes 0.894 -> **1.000** with
exactly **27** stored exceptions on both seeds — precisely the number of
transitions on which `grid` and `walled` differ. Plant frozen, entry
unchanged, zero gradient steps; exceptions are recorded only where the
rule is observably wrong.

The degeneracy check is what makes this a result rather than a lookup
table: the store holds **zero** entries for `grid`, `perm` and `line` at
every observation budget up to 1024. It grows only where rules fail.

Its limit is observation rather than capacity — coverage runs 8 -> 18 ->
23 -> 27 as the world is watched longer, exactly as random sampling of
(state, action) pairs predicts.

Scope: this store is an exact dictionary, an idealised content-addressed
memory. A learned or approximate one would be lossy and is untested.
What is established is that the information needed is small, precisely
localised, and obtainable by watching.

*Two corrections recorded rather than quietly fixed: F87 claimed the
discrimination gap was approaching an asymptote, from four points, right
after criticising a linear extrapolation from those same four points —
F94's measurements support neither. And the wrong-context null silently
stopped being a null when near-duplicate families were added, reading
1.000 where the truth was 0.117 (F93).*

*Superseded (F85): retrieval built, clause (c) falsifiable at last.*
Consequence probing (F44) identifies the right entry among 64 at 0.969
against a 0.016 chance floor, both seeds, with a discrimination null
that works: in-bank tasks score 1.000, strangers 0.642.

So (c) now splits cleanly, and this is the current frontier:

- on ACCURACY it passes — retrieval holds to N=64;
- on COST it fails — a linear scan is O(N), so identifying a known task
  among 64 costs 64 forward passes while MINTING a fresh entry costs
  2.7-7.0 update steps. At N=64, recognising is already dearer than
  learning. A naive linear bank does not scale.

The fix is content-addressed keys for sublinear lookup — infrastructure
this project has carried unused from the start, and F85 is the first
measured reason to wire it in. §2.3's two addressing routes are the
candidates.

Watched but not yet actionable: the in-bank/stranger score gap shrinks
about 0.07 per doubling (0.571 at N=8 to 0.358 at N=64). Extrapolating
it to zero in the low thousands is NOT supported by four points and two
seeds, and the runner-up margin's decrements are decelerating. Re-measure
at N=128 and 256.


Flat curve = a library of separate entries. Downward = a map. This is
the headline gate and it is falsifiable in a way "we did not forget"
never was.

**MET (F69, seed-widened F70), with a model-holding plant.** Three novel
rungs in sequence, five seeds, means: policy cost 100 -> 260 -> 400
(rising, total 760, quality decaying 0.944 -> 0.444); model cost
60 -> 160 -> 45 (falling, total 265, quality flat 1.000 -> 0.881).
Roughly 3x cheaper and uniformly better, with the third and hardest task
the cheapest of the three. The shape holds on each seed individually —
model cost falls 5/5, policy cost rises 5/5 — and final reach separates
with no overlap (model 0.812-0.938, policy 0.234-0.547, floor 0.219).
Policy r4 is right-censored at the budget cap on 5/5, so the cost gap is
a lower bound.

**RETRACTED as a general claim by F71. It was nesting.** r2/r3/r4 share
one state space and AGREE on every shared (state, action) pair, so
training on the later rung reinforces the earlier one and no gradient
ever conflicts. On four families whose dynamics genuinely differ
(`schema_family.py`), the same policy-free model forgets to the chance
floor — `line` retained at 0.138 against a 0.125 floor — and the
sequential cost saving is smaller than a scrambled-dynamics control's.
The gate stands for NESTED families only.

**What replaced it.** F73/F74 split the claim in two, and this is the
useful form:

| what is stored | where it must live | evidence |
| --- | --- | --- |
| structure (task-general, small) | plant weights | F73: 2.36x cheaper acquisition, 1.03x on scrambled dynamics — causally structure |
| content (per-family, unbounded) | external bank | F71/F74: in weights it is erased, and weight sharing makes it worse |

The headline gate is therefore restated: **acquisition cost for task N
falls as N grows, on families that do not nest.** That is not yet met.
F73 supplies the structural half of the mechanism; the bank must supply
the content half. See §6.

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
   storage-rule violation to whatever is still plastic. **RESOLVED by
   F67:** hold no policy at all. A plant that learns only a transition
   model, with behaviour derived by search, beats every policy-storing
   variant — 0.969 on its own domain against a learned policy's 0.441,
   and 0.573 on a novel domain with ZERO target training, also beating
   0.441. The founding claim is satisfied: prior learning made a novel
   task cheaper. A model is factual and transfers as incompleteness (fix
   by observing); a policy is preferential and transfers as error (must
   be unlearned). Depth buys competence (0.531 -> 0.573, depth 6 -> 12),
   the first evidence for the deliberation property.
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
