# Prototype route memory under an untrained nonlinear representation (2026-08-10)

This audit adds bounded slot-local route memory to the replay-free nonlinear
external transition experiment. Each logical slot owns up to four normalized
opaque trajectory prototypes. Verified matches may merge or append a prototype
only to that slot; no shared route scorer is updated and no raw transition row
is retained. Proposals remain non-authoritative and factual transition-model
verification remains the acceptance gate.

The experiment was run with:

```text
.venv/bin/python -m experiments.external_learned_nonlinear_open_world.train \
  --seed <seed> --prototype-route-memory --route-memory-prototypes 4 \
  --match-tolerance 0.01 --report-out report_seed<seed>.json
```

## Result

| seed | held-out quality | route proposal matches factual winner | revisit matches | exact persistence | promoted |
| ---: | :---: | :---: | :---: | :---: | :---: |
| 82601 | pass | no | 0/6 | yes | no |
| 82602 | fail | no | 0/6 | yes | no |
| 82603 | pass | no | 0/6 | yes | no |

The memory itself grew only in external slot state and restored exactly. It
did not recover identity because the frozen, untrained context representation
produced highly similar prototypes for distinct nonlinear regimes. This is
not a memory-retention failure; it is an information-formation failure.

## Interpretation

The prototype boundary is retained as useful infrastructure: it gives the
system protected, independently versioned route state that can grow and later
be compressed under a verifier. The current route result is rejected as a
capability gain. Adding more prototypes or changing the cosine floor would
not create separability absent from the representation. The next experiment
must provide a representation-stable or meta-learned initialization while
keeping prototype writes isolated and verifier-gated.
