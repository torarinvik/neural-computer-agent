# World-count cliff (F132)

Probe 232. Iterated + oracle entry, depth<=4: 1 world 1.0000, 4 worlds
0.0547, 16 0.0623, 64 0.0570, 256 0.0548 (chance 0.0435). A cliff at
1->4, flat thereafter. The break is at needing CONTEXT-supplied
parameters at all, not at scale.

With F131 (depth 1 reads, depth 4 does not): conditioned execution
fails as soon as depth exceeds one. F121/F125 work because a single
world puts the pieces in weights — so they show compositional
execution, not bank-fed composition.
