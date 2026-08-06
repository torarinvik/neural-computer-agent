# Replicated append-only grammar-shift acquisition (2026-08-06)

Status: replicated promoted bounded no-replay grammar-shift growth.

The append-only route chain acquired composition ID `6`, a three-primitive
program (`reverse -> complement -> rotate`), after three protected rows. The
base route and all earlier extensions stayed frozen; the new extension was
trained only from fresh outcomes for the new composition. The same protocol
was run with seeds `69316` and `69317`.

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| artifact 0 behavior | 0.9102 | 0.9688 |
| artifact 1 behavior | 0.8555 | 0.9414 |
| artifact 2 behavior | 0.9453 | 0.9609 |
| grammar-shift artifact 6 behavior | 1.0000 | 0.9844 |
| causal route accuracy | 1.0000 | 1.0000 |
| candidate-key permutation accuracy | 1.0000 | 1.0000 |
| cold-start old-route accuracy | 1.0000 | 1.0000 |
| stage-specific shuffled control samples | 0.0000 | 0.0000 |

All artifact rows were protected. Reload, corruption rejection, frozen-core,
and zero-replay gates passed in both runs. This promotes replicated
append-only acquisition across a grammar shift to a longer computation. It
remains bounded continual external growth: the grammar, artifact blueprint,
and append-only memory capacity are finite, so this is not yet general
continual learning, unrestricted memory growth, or open-ended program
induction.
