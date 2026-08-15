# Sampled-rule baseline (2026-08-15)

Status: **diagnostic**. Development seed 41, already consumed. Nothing was
admitted, no holdout was spent, and `AgentBrain.bank` was byte identical
afterwards. This is a baseline for later work to beat, not a claim.

## What was measured

The searcher the leases use, unchanged, run against rules sampled from
`rule_automata` — the general class of finite-state rules over the symbol
stream — at 448 steps, alongside the four hand-written rules on the same axis.

| Rules | Solved |
| --- | ---: |
| Hand-written (`current_symbol`, `onset`, `changed`, `n_back-1`) | **4 / 4** |
| Sampled, 1 state | 2 / 3 |
| Sampled, 2 states | 0 / 3 |
| Sampled, 3 states | 0 / 3 |
| Sampled, 4 states | 0 / 3 |
| Sampled, 5 states | 0 / 3 |
| Sampled, 6 states | 0 / 3 |

Mean accuracy gain over a never-press constant policy, on sampled rules of two
states or more: **+0.049**. Five of those fifteen rules scored *below* the
constant policy.

## What it means

The four hand-written rules are solved completely; the class they belong to is
not. The gap is not explained by difficulty as the system's own history would
measure it: `onset` is a **2-state** rule and needed a whole lease, while
`changed` and `n_back-1` are **4-state** rules solved by a single retrieve or
invert. Sampled 2-state rules, no harder by that measure, are not solved at
all.

So the searcher is not weak in proportion to complexity. It is tuned to four
particular rules — which is exactly what the seed-holdout protocol could never
detect, because holding out seeds re-samples episodes of a rule already seen.

Three candidate explanations remain open and the diagnostic does not separate
them:

- **inexpressible.** With `max_history = 4` and 20 trainable numbers, most
  finite-state rules may have no representation at this controller geometry.
- **unsearchable.** The grammar's five operators and hand-written ordering may
  simply not reach them.
- **under-acquired.** Each proposal gets one acquire lifetime, the same budget
  the leases use. That sufficed for rules whose structure matches an operator;
  it may not for rules that need a template the acquire rule never forms.

The third was the cheapest to rule out, and it is ruled out. Repeating four of
the failures with eight acquire lifetimes instead of one moved nothing: two
rules were unchanged (`0.663`, `0.580`) and two got *worse* (`0.795` to
`0.558`, `0.641` to `0.542`), because more reward-weighted averaging pulls the
template toward the mean event rather than toward a state-dependent one. More
experience does not help, which is itself evidence that the missing thing is
structural rather than statistical.

Telling these apart is the point of the audit's O6, and `program_search.py`
already warns that an inexpressible target looks exactly like slow search.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.rule_baseline
```

Roughly 30 seconds. Sampling is seeded, so the rule population is fixed.
