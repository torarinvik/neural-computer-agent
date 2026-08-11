# Literature mechanisms: two nulls (F148)

Probe 248, 2 seeds each, matched at 40k updates.

    baseline (F144)      0.7993 / 0.8447   exact 0.3158 / 0.3543
    semi-amortization    0.7957 / 0.8364   exact 0.2879 / 0.3147
    codebook K=256       0.7854 / 0.8212   exact 0.2576 / 0.4076
    longer training      0.8520 / 0.8889   exact 0.3612 / 0.4845

Both mechanisms null; stranger gaps unchanged, so refinement is not
collapsing either. The codebook's apparent exact-match advantage on
one seed reversed on the other — retracted.

Only more reader training moved the number. The saturation pair
decides whether the amortization framing survives.
