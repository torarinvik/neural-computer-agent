# Outcome-only learned temporal offset growth

This promotion tests the next bottleneck after storage: can an external file
learn which relative history address is useful from scalar verifier outcomes?

## Protocol

- Freeze the canonical controller and learned event frontend.
- Train an external n-back-4 file, then freeze it.
- Train a fresh n-back-5 file with an opaque categorical offset policy over
  offsets 1–8 and a learned output readout.
- Append learned event tensors to `ExternalTemporalHistoryMemory`; the file
  selects an offset and receives only the selected event plus an explicit
  presence mask.
- Credit the selected output logit with attempted-outcome BCE and the offset
  with scalar policy credit.
- Test wrong offsets, missing history, shuffled outcomes, old-file retention,
  frozen-core digests, and zero replay.

No family name, n-back depth, target bit, correct action, or physical memory
address enters the learner.

## Results

| Measure | Seed 17 | Seed 18 |
| --- | ---: | ---: |
| Old n-back-4 accuracy before growth | 1.0000 | 1.0000 |
| New n-back-5 accuracy | 1.0000 | 0.9132 |
| Learned n-back-5 offset mode | 5 on 8/8 lifetimes | 5 on 8/8 lifetimes |
| Old-file retention after growth | 1.0000 | 1.0000 |
| Wrong-offset maximum | 0.6493 | 0.6840 |
| Missing-history maximum | 0.7917 | 0.7917 |
| Shuffled-outcome maximum | 0.6597 | 0.4028 |
| Replayed examples | 0 | 0 |

All promotion gates passed on both seeds, including frozen controller/frontend,
unchanged old file, stable fresh retention, offset selection, wrong-offset
rejection, missing-history rejection below the 0.80 mastery threshold,
shuffled-outcome rejection, and zero replay.

Each seed used 311,296 primary verifier bits, 147,456 matched-control bits,
14,336 audit bits, 32,768 primary logical lifetimes, 16,384 control logical
lifetimes, 1,024 primary optimizer updates, 512 control optimizer updates,
and zero replayed examples.

## Claim boundary

This promotes a reusable scalar-credit mechanism for discovering one global
relative temporal offset in an isolated external file while preserving an old
file. It does not establish query-conditioned addressing, multiple competing
memory items, learned content search, compression, unbounded computation, or
general continual learning. The next bottleneck is conditional addressing:
the address policy must choose among multiple useful offsets or content keys
based on learned context, while retaining prior files without replay.

Raw reports are `seed-17.json` and `seed-18.json`.
