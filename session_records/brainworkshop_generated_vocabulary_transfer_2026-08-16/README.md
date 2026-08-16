# Generated temporal vocabulary transfer (2026-08-16)

Status: **development diagnostic; not a holdout and not a promotion**.
This is the next consolidation step after the cross-world operator test. It
replaces a hand-written relation list with candidates generated from the event
stream itself, then checks that a candidate survives fresh verification before
it can become an external artifact.

## Method

Each stream contains two separately bound opaque event channels and one scalar
verifier outcome per time step. The generator constructs unary persistence and
change predicates, pairwise equality/same-change/same-delta predicates, and
boolean compositions. It does not receive the hidden rule name. Candidates are
ranked by description bits plus prediction-error bits. The selected candidate
is quarantined and must be exact on independent verification streams before
admission.

The held-out transfer task uses a fresh symbol offset in each world, so the
candidate must reuse the relational operation rather than a symbol coordinate.

| arm | inherited object | target evidence cost |
| --- | --- | ---: |
| retained | source candidate after fresh verification | 64 bits to stable threshold |
| fresh | target discovery plus fresh verification | 256 bits |
| irrelevant | candidate for a different hidden rule | no stable prefix |
| corrupted | negated retained candidate | no stable prefix |

## Development result

Three replicates generated **77 candidates** each. The MDL selector chose the
hidden `same_delta` operation with zero training and zero fresh-verification
errors in every source run. On the target streams:

| | stable bits (three replicates) |
| --- | ---: |
| retained generated predicate | **64, 64, 64** |
| fresh candidate search | 256, 256, 256 |
| irrelevant artifact | none |
| corrupted artifact | none |

The transfer ratio against a fresh learner is **0.25**. Every artifact and
candidate is content-digested; no curated bank or production controller is
modified. Accounting is serialized separately for unique verifier bits,
logical lifetimes, optimizer updates, replay, latency, wall time, stable bits,
and retention.

This is evidence for the vocabulary-discovery boundary, not for an open-ended
semantic language. The candidate generator's *operator family* (equality,
change, persistence, relative transition, and boolean composition) is still a
declared research primitive. The remaining work is to derive that family from
learned event algebra and then run it on rendered amodal streams.

## Decision and next step

Keep the generated-candidate plus quarantine/verification boundary. Do not
promote this predicate artifact or call the result a holdout. The next
promotion-safe step is a preregistered rendered-stream holdout with an
irrelevant vocabulary and reward-shuffled/missing-evidence controls.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.generated_vocabulary_transfer
```

The three-replicate diagnostic takes well under a minute.
