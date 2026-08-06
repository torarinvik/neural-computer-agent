# Lower-rate retained decoder prior — 2026-08-06

Status: rejected as a promoted continual-learning strategy.

This follow-up kept the retained head decoder initialization but reduced the
learning rate from `1e-3` to `3e-4`, testing whether slower adaptation would
preserve the prior's early alignment. The head program and shared controller
were frozen, downstream raw events were hidden, and no examples were replayed.

| arm | consumer accuracy | blank accuracy | zero-head accuracy | reward-shuffled accuracy | stable consumer bits |
| --- | ---: | ---: | ---: | ---: | ---: |
| `3e-4`, medium | 0.6719 | 0.6719 | 0.6641 | 0.3984 | none |

The lower rate suppressed useful consumer adaptation rather than stabilizing
it: the consumer did not beat blank, the head ablation was not meaningfully
causal, and no stable mastery prefix appeared. The direction is rejected
without a full rung. The raw pilot and medium report retain the required
accounting and control evidence.
