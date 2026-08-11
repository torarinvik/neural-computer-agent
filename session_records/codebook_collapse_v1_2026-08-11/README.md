# Codebook collapse (F146)

Probe 246. VQ discrete entry, K in {64, 256}, 2 seeds. All arms at
chance with stranger identical to own — and K=64 and K=256 gave
BYTE-IDENTICAL results, which cannot happen unless K is irrelevant.

Instrumented: 1 of 64 codes used, claiming 40/40 worlds. The entry was
constant, not discrete. Classic VQ codebook collapse: one code wins
every assignment at initialisation and the losers never get gradient.

Fixed by periodic dead-code restart (re-seed unused codes onto recent
reader outputs every 500 updates). Verified 24 distinct codes across
40 worlds afterwards. The discrete-bottleneck hypothesis remains
UNTESTED.
