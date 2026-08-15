# Adversarial audit: a bug in `minimize` was suppressing every result (2026-08-15)

Status: **correctness fix plus retraction**. Development seed 41. Nothing is
admitted and `AgentBrain.bank` is unchanged at `07319eb1`.

The stack was probed for the two assumptions nothing had tested: that feedback
labels are exact, and that the target is finite-state. The first probe crashed
it, and chasing the crash surfaced a defect in the core representation that had
been quietly degrading every measurement in the session.

## The bug

`rule_automata.minimize` numbered its merged blocks by **signature order**.
`RuleAutomaton.expected` always starts at state 0. Nothing made the block
containing the start state come first, so minimisation could return a machine
with *different behaviour* -- and `canonicalize`'s own docstring asserted the
invariant that `minimize` did not enforce.

The clearest case is a two-state parity rule. It minimises to the same
transition table with its outputs swapped: the exact inverse of the machine
that went in.

Over 2000 random machines of one to five states:

| | behaviour changed by `minimize` |
| --- | ---: |
| before | **850 / 2000 (42.5%)** |
| after | **0 / 2000** |

The fix is three lines: number the block containing state 0 first.
`test_rule_automata.py` now sweeps random machines and asserts both behaviour
preservation and idempotence, so this cannot come back quietly.

## Blast radius

`minimize` is called by the task sampler, `known_rule`, the identification
search, `product_rule`, `factorize`, and `reconstruct` -- that is, by
everything. **Ten of the eighteen sampled rules changed digest**, so records
written before this fix describe a different task set.

Everything measured post-fix is better, in some cases by a lot:

| Measurement | before | after |
| --- | ---: | ---: |
| Composition, cost ratio against no library | 0.878 | **0.141** |
| Composition, tasks identified | 9/18 | **18/18** |
| Induced counter programs, 28x16 feedback | 15/18 | **18/18** |
| Induced counter programs, 112x16 feedback | 18/18, 14 exact | **18/18, 18 exact** |
| Identification from one episode | 10 exact | **11 exact** |

## The accumulation result, adversarially checked

The composition curriculum was re-run over four independently sampled
primitive pools. Cost is labelled steps on composites, against a control that
induces every task from scratch.

| Primitive pool | library arm | control |
| --- | --- | --- |
| 8000 | 18/18, ratio **0.141** | 16/18 |
| 8500 | 17/17, ratio **0.123** | 17/17 |
| 9000 | 18/18, ratio **0.097** | 13/18 |
| 9500 | 20/20, ratio **0.087** | 17/20 |

**Seven to eleven times cheaper, identifying every task, on every pool.** The
library learns four primitives at 112 labelled steps each and then answers
every composite at 112 steps -- one rung of the ladder -- while the control
needs up to 1792 and fails outright on several.

This is the accumulation curve the interpreter decision named as its own
falsifier, finally bending the right way, and it is now robust to the sampler
rather than a single lucky draw.

## Retraction

**The factorisation result recorded in `FACTORING.md` does not survive the
fix.** Hartmanis-Stearns decomposition with fitted output tables was measured
an hour earlier at 11/18 against the combiner library's 9/18, and read as the
mechanism that made composition work without an experimenter's list. Post-fix,
on all four pools, it is *worse* than simply enumerating `and`, `or` and `xor`:

| Pool | combiners | factoring |
| --- | ---: | ---: |
| 8000 | 0.141 | 0.283 |
| 8500 | 0.123 | 0.170 |
| 9000 | 0.097 | 0.208 |
| 9500 | 0.087 | 0.342 |

The apparent advantage was an artifact of mis-minimised machines. Factoring is
retained -- it is exact, it reconstructs correctly, and it may matter on
distributions where three combiners do not suffice -- but the claim that it
beat the simpler mechanism is withdrawn.

## What the probes still break

The fix did not make the stack robust. These remain open and are recorded
rather than smoothed over.

**Noise.** `build_tree` raised `ValueError` on the first contradicted prefix,
which at 2% label noise means the learner crashed instead of losing a label.
Cells now hold counts, majority wins, and the tree reports its disagreement
rate (0.0039 at 2% noise, 0.0089 at 5%). But the exact search still demands
perfect consistency, so under noise it either abstains or returns an overfit:
an in-class two-state probe at 0.5% noise comes back as an **11-state machine
at chance accuracy**. A violation-budget search and an ALERGIA-style
statistical merger were both tried; neither is good enough to adopt, and both
are kept in `robust_induction.py` with the measurements that condemn them.

**Out-of-class targets.** A running-majority rule -- press while one symbol
has occurred more often than another -- is not finite-state at any size. The
inducer does not abstain on it. It returns a **12-state machine at 0.518
accuracy**, which is a confident wrong answer rather than a refusal. Nothing
in the stack detects that its hypothesis class cannot contain the target.

**Five- and six-state rules** remain unidentified from a single long episode,
as the identification record already says.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_rule_automata.py -q
```

The random-machine sweep is the regression guard. The composition sweep across
primitive pools is in this record's `seed_sweep.txt`.
