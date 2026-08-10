# Bind-once in the games (F143)

Probe 243, 3 seeds. F135 carried across: the entry is reduced to one
vector added to the state token instead of attended over at every
rollout step. Binding is the only change from F141.

    held-out +0.0980 / +0.1334 / +0.1373, pooled +0.1229
    oracle-value target +0.1234 | full oracle +0.1954 | floor -0.0318
    entry effect +0.297 / +0.348 / +0.358 (largest measured)
    inverted top=food 0.333 / 0.188 / 0.375 (F141: 0.417/0.042/0.000)

The polarity asymmetry open since F112 is closed, on every seed.
68.1% of floor-to-full-oracle; the remaining 31.9% is F110's
search/dynamics residual, which a better entry cannot touch.
