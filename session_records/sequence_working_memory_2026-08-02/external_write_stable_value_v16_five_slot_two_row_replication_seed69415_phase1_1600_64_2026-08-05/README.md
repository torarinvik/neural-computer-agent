# Stable controller value path — five-slot, two-row replication

Status: promoted narrow five-slot retention rung.

Seed 69415 initially failed parent acquisition at 800 phase-1 steps. Extending
only the phase-1 curriculum to 1,600 requested steps stabilized the parent
after 1,152 effective updates and preserved the same architecture and
training controls.

- target-first: `0.978`
- target-last: `0.987`
- intact: `0.983`
- mastered-parent retention: `0.992`
- unseen-token minimum: `0.969`
- stable bits to threshold: `55,296`
- replayed examples: `0`
- wall time: `97.06s`

All causal, corruption, clear-memory, missing-evidence, parent-retention, and
order-symmetry gates passed. The matched persistent audit is recorded in the
`...phase1_1600_persistent_64...` directory.
