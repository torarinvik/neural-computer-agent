# Span-11 append from the fourth-slot parent — 2026-08-03

The promoted fourth-slot complement checkpoint was used as the parent for a
new span-eleven successor slot. The learner received mixed forward/reverse
sequence outcomes, opaque attempted actions, and no sequence labels. Span
nine and span ten were rehearsed, with residual/gate/logit penalties of 0.03.

## Results

| Target lifetimes | Span 9 Δ | Span 10 Δ | Span 11 parent | Span 11 child | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 256 | −6.63 pp | −4.78 pp | 73.71% | 72.12% | reject |
| 1,024 | −3.53 pp | −4.96 pp | 73.64% | 75.79% | reject |

The 256-lifetime arm actively harmed old skills and slightly reduced span 11.
The 1,024-lifetime arm raised span 11 by 2.15 points, but the retention
violations were too large for promotion. Blank/reset controls remained near
chance, so this is not a rendering or reset artifact; it is an interference
and credit-assignment failure.

These results close the immediate span-11 append fork from this parent. The
earlier input probe still shows that useful next-action information is present
at the slot input, but outcome-only training does not reliably bind it without
damaging older spans. Do not scale this branch blindly. The next high-ROI
direction is a more explicit, task-agnostic credit mechanism or a smaller
intermediate primitive, with the same old-skill retention gates.
