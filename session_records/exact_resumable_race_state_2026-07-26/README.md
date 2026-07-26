# Exact resumable population-race state

Date: 2026-07-26

## Purpose

Successive halving saves compute only if a winning clone can continue from its
screening prefix. Replaying the prefix preserves unique experience but wastes
wall time and candidate compute.

## State boundary

The trainer now supports:

- `--prefix-state-out`: save controller residual weights, physical disk banks,
  row/query state, latent strategy memory, context encoder/optimizer, both
  explicit random generators, reward context, intervention state, trace,
  counters, and audit accounting;
- `--resume-prefix-state`: validate the behavioral configuration and continue
  at the exact next scheduled physical round.

Prefix states remain non-graduating. Resuming does not relax any parity,
persistence, retention, or final capability gate.

## Exactness audits

A six-round smoke audit compared:

1. uninterrupted six rounds;
2. three rounds saved to disk;
3. the saved state resumed through round six.

Final payloads matched recursively and bit-for-bit across every tensor,
physical memory row, strategy-memory statistic, encoder parameter, RNG state,
trace row, gate, and accounting field.

The promoted audit used the real strong configuration: four banks, 18 rounds
per phase, four strategy slots, sixteen direction proposals, and a four-seed
read-only shadow audit. An uninterrupted 54-round run was compared with an
18-round saved prefix resumed through round 54. Final states and scientific
reports again matched recursively bit-for-bit, including 81.9% old-return
target accuracy.

## Verdict

Promoted. Exact state continuation is safe for the planned two-stage race.
The next experiment can stop acquisition-screen losers at round 18, resume
only selected candidates to round 42, and resume only the six-round return
winner to round 54.
