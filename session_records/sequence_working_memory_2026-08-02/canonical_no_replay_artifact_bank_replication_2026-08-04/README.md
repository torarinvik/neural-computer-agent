# Canonical no-replay artifact-bank retention replica — 2026-08-04

Independent seed `69317` replicated the promoted artifact-bank mechanism.
The parent acquired span 2 once; later span-3 and span-4 artifacts were
trained independently over the frozen parent with no old-example replay.

- route accuracy: `100%` for spans 2, 3, and 4;
- final selected accuracy: `100%`, `100%`, and `90.625%`;
- stable span-4 threshold: `12,288` verifier bits;
- blank-sequence control: `50.0%`;
- replayed examples: `0`;
- frozen parent core: exact bit identity.

This confirms a narrow CPU/filesystem-style retention mechanism: later
learning appends an executable file and updates routing state, while earlier
files remain intact. It does not claim arbitrary lifelong learning or a
general learned address-discovery system.

Harness: `experiments/working_memory_continuous/canonical_no_replay_artifact_bank.py`.
