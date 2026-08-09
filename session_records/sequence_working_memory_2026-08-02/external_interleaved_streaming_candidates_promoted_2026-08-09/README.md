# Interleaved streaming factual candidates — promoted bounded mechanism

Across seeds `1901` and `1902`, two novel affine dynamics streams were
presented in alternating four-row windows before either candidate was
promoted. The router isolated both candidates, consumed `64` rows per
candidate exactly once, retained zero raw provisional rows, and selected the
affine sufficient-statistics family through held-out verification from a
mixed affine/random-feature candidate set.

All held-out errors were below `1e-6`: the largest was
`8.07e-14` for seed `1901` and `6.77e-14` for seed `1902`. Shuffled-next-state
controls reached `37.49` and `13.99` and were rejected. A full-capacity
control refused the second unverified stream without modifying the committed
source slot. Controller freezing, source retention, and exact persistence
passed in both seeds. A deliberately ambiguous window, equally explained by
the two provisional models, was refused with no candidate update in both
seeds.

This promotes bounded interleaved replay-free factual-candidate isolation and
verifier-gated model-family selection. It does not establish arbitrary
nonlinear one-pass learning, unrestricted growth, or general continual
learning.
