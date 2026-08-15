# Rule expressiveness: inexpressible, or merely unfound? (2026-08-15)

Status: **diagnostic**. Development seed 41, already consumed. Nothing
admitted, `AgentBrain.bank` byte identical, no holdout spent.

The sampled-rule baseline could not say whether the searcher's 0/15 on
multi-state rules meant the rules had no representation or merely were not
found. This answers it by enumeration instead of by searching harder, against
two independent ceilings.

- **machine ceiling** — every program the current geometry can hold, executed
  directly with search bypassed: each of the four temporal addresses, with and
  without inverted intention; every prototype template the acquire rule could
  converge on (the mean of each non-empty symbol subset, exhaustive at this
  alphabet); and every AND of the two. 98 programs per rule.
- **window ceiling** — the best accuracy *any* policy could reach seeing only
  the last `w` symbols, i.e. the Bayes-optimal windowed predictor. It bounds
  every architecture with that much memory, this one included.

Validation: the enumeration recovers the known answer for all four
hand-written rules at `1.000` — `current_symbol` by `prototype (0,)`, `changed`
by `invert address 0`, `n_back-1` by `address 0`, and `onset` by
`and(invert address 0, prototype (0,))`, which is exactly the program the onset
lease selected. The window ceiling returns exactly `1.000` at each rule's
theoretically required window and below it otherwise.

## Results

| | Mean over 18 sampled rules |
| --- | ---: |
| Memoryless (`w=1`) ceiling | 0.789 |
| **Current program family ceiling** | **0.786** |
| Window-5 ceiling | 0.931 |
| Search actually achieved | 0.680 |

| Question | Answer |
| --- | ---: |
| Rules blocked by memory (`w=5` ceiling below 0.8) | **0 / 18** |
| Rules the current family can express | 7 / 18 |
| Of those, found by search | **2 / 7** |
| Rules within window reach but beyond the family | 11 / 18 |

## What this settles

**It is not the controller geometry.** Every sampled rule has a window-5
ceiling of at least `0.830`; a policy seeing the last five symbols could solve
all of them. `max_history = 4` is not the binding constraint, and a blueprint
change to widen memory would not address this failure. That was the hypothesis
this diagnostic was built to test, and it is refuted.

**The family's whole use of memory is worth −0.003.** The mean ceiling of every
program the machine can hold is *below* the memoryless ceiling on these rules.
Despite carrying a four-tick history, the family can only ever test equality
against one lagged symbol, so on rules whose state is not "what was the symbol
k steps ago", the history buys nothing. The hand-written four are precisely the
rules for which that one operation is the answer.

**Search leaves most of its own space unused.** Seven sampled rules are
expressible by programs the machine already supports; the searcher found two.
Its mean shortfall below its own family's ceiling is `0.106`. That is a
proposer failure, established independently of the rules the family cannot
express at all.

## What it points at

Two separate deficits, both real, neither fixed by a bigger controller:

1. **a richer program family** — 11 of 18 rules need a program whose output
   depends on accumulated state rather than on one lagged comparison. The
   substrate for this already exists elsewhere in the repo: `control_flow.py`
   is a two-counter machine. It is not wired to this controller.
2. **a proposer** — worth 5 rules immediately, before any new operator, and
   its value grows with the family it searches.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.rule_expressiveness
```

About five minutes: 22 rules by 98 programs of 448 steps, plus 40,000-symbol
probes for the window ceilings.
