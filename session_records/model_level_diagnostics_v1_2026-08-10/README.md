# Scoring the models directly: the search was never the defect (F106)

Every games finding from F100 on was inferred from reward alone. These score the
models themselves.

## Transition model
per-slot 0.8154, exact-state 0.5842 (held-out worlds, 40000-update runs).
Mediocre; not the binding defect.

## Outcome model
balanced accuracy 0.4312 (chance 0.3333). Per-class recall: cost 0.4672,
**nothing 0.0000**, food 0.6575. F103's class-balanced loss did not fix F102's
degeneracy — it inverted it. F102's model always said "nothing"; this one never
does.

## Twin discrimination — the decisive measurement

Same (state, action) batch, scored with the CORRECT entry and with the INVERTED
TWIN's (identical rendering, opposite rewards):

| | value |
| --- | ---: |
| label agreement with twin entry | 0.9998 |
| mean abs P(food) gap | 0.0000 |

The outcome model predicts identically under an entry and its exact inverse. No
search over such a model could distinguish the twins — so every behavioural
failure in F100-F105 was downstream of this.

## Sub-fork: the reader isn't encoding the bit either

Cosine between a world's entry and its inverse's: **0.9855** (8000-update run,
6 worlds, range 0.980-0.992).

## This is F58

"With few goals, ignoring the goal channel is competitive, and under isolation
it is OPTIMAL — so the plant learns an unconditional habit and never reads its
instruction." The two halves collapse together: the outcome model finds the
twin-average, which leaves the reader no gradient, which leaves the model
nothing to read.

Fix: F58's phase-1 IGNORANCE OBJECTIVE, already built, never applied here.
Testable prediction: twin agreement must fall well below 0.9998 and entry cosine
well below 0.9855 BEFORE any behavioural claim — the model numbers move first,
or the behavioural number means nothing.
