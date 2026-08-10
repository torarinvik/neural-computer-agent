# The ignorance weight: threshold, optimum, and decoupling (F108)

2 seeds each, 40000 updates. Model gate reported before behaviour (F106's rule).

| weight | twin agree | food gap | entry cosine | outcome bal | held-out | entry effect | % headroom |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.9998 | 0.0000 | 0.9855 | 0.4312 | -0.0466 | +0.0005 | 0.2% |
| 0.1 | 0.9931 | 0.0009 | 0.9821 | 0.4175 | -0.0472 | -0.0015 | -0.7% |
| 0.25 | 0.7029 | 0.0912 | 0.6972 | 0.4234 | -0.0324 | +0.0250 | 11.0% |
| 0.5 | 0.5343 | 0.1473 | 0.7119 | 0.4305 | -0.0217 | +0.0499 | 22.0% |
| 1.0 | 0.3364 | 0.1753 | 0.4940 | 0.4474 | -0.0285 | +0.0417 | 18.4% |
| 2.0 | 0.6838 | 0.0863 | 0.3804 | 0.4321 | -0.0362 | +0.0230 | 10.1% |

references: best context-free policy -0.0318, oracle +0.1954, headroom 0.2272

## Three structures

1. **Threshold** — 0.1 does nothing (twin agreement 0.9931). The collapse is
   stable against small pressure; ~0.25 breaks it.
2. **Optimum at 0.5**, inverted-U not plateau: 11.0 -> 22.0 -> 18.4 -> 10.1%.
3. **Decoupling past the optimum.** At 1.0 the MODEL discriminates most
   (agreement 0.3364, best outcome accuracy 0.4474) yet scores below 0.5. At 2.0
   the READER emits the most distinct entries (cosine 0.3804) while the model's
   discrimination falls back. The halves stop moving together.

Maximum discrimination is not maximum benefit. The quantity to tune is the
AGREEMENT between reader and model, not the separation of either — a statement
only available because F106 measured them separately.

## Remaining 78%

Not diagnosed. Ordinary candidates: transition model 0.5842 exact, outcome model
0.4474 balanced at best. Beam search over models that inaccurate has a low
ceiling regardless of how well the entry is read.
