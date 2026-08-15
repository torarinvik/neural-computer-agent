# Capability audit: what this architecture can do, and what blocks it

Written 2026-08-15 against measured behaviour, not against
`AMODAL_N_TO_M_ARCHITECTURE.md`'s target. Where the design doc describes an
instruction kernel or a capability that is specified but not exercised, this
file says so.

## 1. What is verified today

> Since this audit was written, its first prediction has been tested. The
> sampled-rule baseline in §3 O1 shows the searcher solving 4/4 hand-written
> rules and 0/15 sampled rules of two or more states. Read §1 as a description
> of a system that fits its four training rules well.

| Quantity | Value | Where |
| --- | --- | --- |
| Frozen controller parameters | 3342 | `temporal_controller_previous_event_seed1001.pt` |
| Trainable per task | **20** (4 address logits + 16 prototype) | `PretrainedControllerProgramMachine` |
| Event width / history / sources / actions | 16 / 4 / 1 / 2 | controller configuration |
| Curated bank | 3 slots, 1-2 instruction rows each | `AgentBrain.bank` |
| Search grammar | 5 operators, 19 proposals, 5 executable | `program_search.py` |
| Public task rules | 4 (`n_back`, `current_symbol`, `changed`, `onset`) | `rendered_environment.py` |
| Verified transfer ratio | 2.81x warm vs fresh, 32 seeds | executive composition record |
| Cost of one standing lease | ~2700 verifier bits, 33-42 lifetimes | discriminating lease ledgers |

Genuinely established, with controls and held-out seeds:

- a frozen controller can execute an external program file and never update
  during holds; the program, not the network, carries the capability;
- programs compose and the composed child is retrievable, reloadable, and
  checksummed; composition is verifier-gated;
- one capability transfers to a related task at 2.81x the sample efficiency of
  a matched fresh learner;
- a task needing two operator families (`onset`) is solved only by their
  conjunction, and the winner beats its strongest rival by margins with
  reproduction probabilities near `1e-49`;
- the whole apparatus fails closed: digests, frontend binding, seed ledger,
  trial floors, unequal-primitive composition, corrupt banks.

That is a real result about *program-carried capability under a frozen
interpreter*. It is not yet a result about general ability.

## 2. Where each layer actually stops

**Substrate expressiveness is not the bottleneck.** `control_flow.py`
implements a two-counter machine ABI — `inc`, `dec`, `jump`, `jump_if_zero`,
`jump_if_nonzero`, `halt`. Two counters with conditional jumps is
Turing-complete in the limit; runtime is bounded only by an explicit step
budget and counter limit. The machine can, in principle, express anything.

**Proposal is the bottleneck.** Nothing infers programs from evidence. The
temporal family enumerates a closed grammar of 19 candidates; the control-flow
family enumerates by length with a learned prior only over *local structural
edits*. Enumeration reaches:

| Program length | Candidates (2 counters) |
| ---: | ---: |
| 4 | 1.4e4 |
| 5 | 7.1e5 |
| 6 | 4.5e7 |
| 7 | 3.5e9 |
| 8 | 3.2e11 |

Every verified program in the repo is 1-2 rows. The wall is at length 6-8, and
it is exponential, so no amount of compute moves it more than a step or two.

**Perception is frozen by fiat.** Encoders are fixed random or curated
adapters; the "amodal" event bus is a hand-chosen 16-d width, not a learned
abstraction. Prototypes bind to a frontend digest and score at chance across a
frontend swap — every lease records `cross_encoder` at 0.500 (current-symbol)
or the base rate 0.749 (onset). No representation is learned or shared.

**Memory is a 4-tick window plus a 3-slot library** — and the window is barely
used. `max_history = 4` bounds every temporal relation the controller can
address in one tick, but the enumeration in §3 O1 shows the whole program
family scoring `-0.003` against a memoryless policy on sampled rules. The
limit that bites is not how far back the machine can see; it is that the only
thing it can do with what it sees is test equality against one lagged symbol.

## 3. Obstacles, ordered by how much they block

### O1 — Generality was untestable, and the first test fails

*Status: the measurement now exists, and it is negative.*

Four hand-written rules is not a distribution. A solver covering all four is
indistinguishable from one fitting four special cases, and every promotion
standard in `AGENTS.md` was powerless against that: held-out *seeds* re-sample
episodes of a rule already seen, never a rule unseen.

`rule_automata.py` replaces the rule list with the general class of
finite-state rules over the symbol stream, sampled rather than written. Rules
carry canonical identity (minimise, relabel, digest), a ground-truth
complexity axis (minimal state count), and a digest-stable held-out split. The
class is defined independently of what the controller can express, so it can
falsify. The four hand-written rules embed exactly, at 1, 2, 4, and 4 states.

The baseline
(`session_records/brainworkshop_sampled_rule_baseline_2026-08-15/`):

| Rules | Solved |
| --- | ---: |
| Hand-written | **4 / 4** |
| Sampled, 1 state | 2 / 3 |
| Sampled, 2-6 states | **0 / 15** |

Mean gain over a never-press constant policy on multi-state sampled rules:
**+0.049**. The failure does not track complexity: `onset` is a 2-state rule
that needed a whole lease, `changed` and `n_back-1` are 4-state rules solved by
one retrieve or invert, and sampled 2-state rules are not solved at all. The
searcher is tuned to four particular rules.

That separation has since been measured by enumeration
(`session_records/brainworkshop_rule_expressiveness_2026-08-15/`), and it
refutes the geometry hypothesis:

| | Mean over 18 sampled rules |
| --- | ---: |
| Memoryless (`w=1`) ceiling | 0.789 |
| **Current program family ceiling** | **0.786** |
| Window-5 ceiling | 0.931 |
| Search achieved | 0.680 |

**Zero** rules are blocked by memory: every one has a window-5 ceiling of at
least `0.830`. **Seven** are expressible by programs the machine already
supports, and search found **two**. And the family's entire ceiling is
`-0.003` against a memoryless policy — carrying a four-tick history buys
nothing, because the only thing the family can do with history is test
equality against one lagged symbol. The hand-written four are exactly the
rules for which that one operation is the answer.

### O2 — Composition only repeats one primitive

`compose_admitted_temporal` fails closed on unequal parents: `compose(0,1)` is
recorded as `unequal temporal primitives cannot compose in this family`. So
depth-2 exists only as "the same delay twice". A library whose elements cannot
combine with *each other* cannot accumulate; it can only lengthen.

The executive family already composes heterogeneous fragments (receive-only
plus persistent loop), so the capability exists on one side of the repo and not
the other. That asymmetry is the cheapest large win available.

Resolved when: a child built from two *different* admitted parents is admitted
on held-out evidence and beats both parents, in the same family the leases use.

### O3 — Nothing proposes programs from evidence

The stated next step in every recent record is "a learned proposer". Today,
search order is a fixed hand-authored preference (retrieve, compose, invert,
and, invent). That ordering is doing real work — it is why `and` beats `invent`
on onset — and it was written by hand, which means the system's apparent
"discovery" is partly the researcher's prior.

Resolved when: proposal ranking is learned from observed event statistics and
outcome history, and it beats the hand-written order on held-out rule families
by a measured bits-to-threshold ratio. Note this is only meaningful after O1;
without held-out families, a learned proposer can memorise four rules.

### O4 — Capability is not shown to accumulate

One transfer ratio (2.81x) on one pair of tasks is a data point, not a curve.
The stated objective in `AGENTS.md` is verified reusable capability per unique
experience, which is a claim about a *slope*: does task N+1 get cheaper as the
library grows? Nothing in the repo measures that.

Resolved when: a sequence of 10+ held-out rules is learned in order, and
bits-to-threshold per new rule is plotted against library size, against a
matched fresh learner. This is the single most informative experiment
available, and it is mostly bookkeeping once O1 exists.

### O5 — No temporally extended goals

Reward is a per-tick scalar and every verified program is a reactive predicate
over a 4-tick window. There is no planning, no subgoal, no credit across a
horizon. `BRANCH`, `LOOP`, `CALL`, and `RETURN` are specified in the
architecture doc's kernel; the counter machine implements jumps, but no
promoted capability uses control flow to pursue a goal over time.

Resolved when: a task requires an action now whose payoff is verifiable only
several ticks later, and a program-carried solution beats a reactive one.

### O6 — The frozen-controller bet is untested at the margin

The design says capability grows in the bank while the controller stays
frozen at 3342 parameters. That is a strong and interesting bet, but nothing
tests where it breaks. With `max_history = 4`, `max_sources = 1`, and 20
trainable numbers per task, some rules are simply inexpressible, and the
architecture cannot currently tell "inexpressible" from "search was too slow" —
`program_search.py`'s own docstring notes search cost must be reported by
family for exactly this reason, but no experiment does it.

Resolved when: a rule family is demonstrated inexpressible at fixed controller
geometry and expressible after a *blueprint* change, with the weight-reset
control from `AGENTS.md` applied.

### O7 — Perception does not transfer

Cross-frontend scores are at chance in every lease. Any capability the bank
holds is welded to one encoder's digest. Nothing that must survive a change of
sensor can be claimed.

Resolved when: a program admitted under one frontend holds above chance under a
different frontend of the same modality, without relearning.

## 4. Recommended order

1. ~~**O1** — rule generator with held-out families.~~ **Done**, and the
   baseline is 0/15 on multi-state sampled rules.
2. ~~**O6** — separate inexpressible from unsearchable.~~ **Done.** Not the
   geometry: 0/18 rules are memory-blocked. Two deficits instead, both real.
3. ~~**Candidate templates.**~~ **Done** — the searcher now saturates its own
   family: 7/18 solved against a ceiling of 7, mean accuracy 0.782 against a
   ceiling of 0.786, and the solved set is exactly the expressible set. Both
   template polarities were needed. Record:
   `session_records/brainworkshop_template_proposals_2026-08-15/`.
4. ~~**A program family with state.**~~ **Done** — `counter_state_programs.py`
   bridges the existing two-counter machine to the rendered stream through a
   fixed press/input/working-state interface. All 18 sampled rules compile and
   run at `1.000`, against 7/18 for the temporal family, and so does 2-back at
   16 states. Record:
   `session_records/brainworkshop_counter_state_ceiling_2026-08-15/`.
5. **O3 — the proposer, now the whole problem.** Expressiveness is no longer
   the constraint anywhere in the rule class; search is. The programs are
   21-130 instructions, so enumeration would sift 10^50 to 10^472 candidates
   against a practical reach near 10^9. Nothing incremental closes that. A
   proposer that infers structure from evidence and picks its next test by
   expected information is the only remaining route — and unlike a month ago,
   there is now a task distribution, a complexity axis, and two ceilings to
   measure it against.
5. **O4** — the accumulation curve over a held-out rule sequence: does rule
   N+1 get cheaper as the library grows? The project's actual thesis, now
   measurable.
6. **O2** — heterogeneous composition, which the state-carrying family will
   need in order to build anything hierarchical.
7. **O7**, **O5** as the results make urgent.

The honest summary: the mechanism is sound and unusually well controlled, the
expressive substrate is already Turing-complete, and the missing pieces are
not exotic. What is missing is a task distribution wide enough to falsify a
generality claim, a library whose parts combine with each other, and something
that writes programs instead of enumerating them. In that order.
