# Disjoint-dynamics families: nesting refuted, top-down structure validated (F71-F74)

The instrument F67-F70 lacked. Four families sharing no surface — `line`
(bounded position), `dial` (counters mod 8), `toggle` (bits, XOR masks),
`perm` (adjacent swaps) — so a downward cost curve cannot be explained by one
family nesting inside another.

Probe: `schema_family.py` (copy included). 5 seeds x {dense, slot} x
{real, --scramble}. Cost = updates actually spent to reach 0.98 exhaustive
dynamics accuracy. Behaviour derived by BFS in the learned model.

## F71 — F67-F70 measured nesting

Retention after learning all four families sequentially (dense, 5 seeds):
line 0.138 (chance 0.125), dial 0.029 (chance 0.002), toggle 0.080, perm 0.997.
The model forgets as completely as any policy. The reacher ladder's rungs all
share one state space and AGREE on shared inputs, so nothing was ever
contradicted there.

## F72 — no schema transfer; the scramble control kills it

| arm | cold | warm | saving |
| --- | ---: | ---: | ---: |
| real families | 660 | 590 | 70 |
| scrambled | 1445 | 1340 | 105 |

The control saves MORE. Slot accuracy on an unseen family stays below the
trivial copy-forward rule (toggle 0.329 vs copy 0.694): the dense model never
learns "copy slot i forward" as a rule, only six unrelated per-slot mappings.

## F73 — slot-symmetric structure pays, causally

| dynamics | dense cold | slot cold | speedup |
| --- | ---: | ---: | ---: |
| real | 660 | 280 | 2.36x |
| scrambled | 1445 | 1405 | 1.03x |

Per-seed totals do not overlap (dense 675/675/600/675/675, slot
275/300/250/275/300). The advantage disappears when the structure is removed,
so it is the architecture matching what the tasks share — not capacity, not
optimisation, not warm-up.

## F74 — content in weights still fails, and sharing worsens it

| arch | cold | warm | warm - cold |
| --- | ---: | ---: | ---: |
| dense | 660 | 590 | -70 |
| slot | 280 | 380 | +100 |

Retention at chance for both. Weight sharing raises structural transfer and
interference together, because it is the same weights doing both jobs.

## What this establishes

STRUCTURE belongs in the plant's weights (worth 2.36x, measured). CONTENT
must live in the external bank (in weights it is erased). That split is the
project's founding architecture, now derived from measurement rather than
asserted.

Next, prediction recorded in advance: freeze the structure-pretrained slot
plant and hold per-family dynamics in the bank. Expect flat retention, cost
below 280, no negative transfer. If retention still collapses, the bank
interface is leaking content into weights.
