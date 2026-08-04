# Balanced-order retention population rejection

This record evaluates the corrected outcome-only retention protocol on the
current v23 runtime. Training uses a stable latest-event memory address,
latest-prior cosine context, parent-stability gating, 256-step alternating
target-first/target-last warmup, balanced-order retention updates, held-out
validation selection, and missing-write/query-cue controls.

The population is rejected. Seed 17 stayed at chance. Seed 18 learned a strong
target-first but chance target-last policy. Seed 19 passed both order checks,
but one seed cannot establish a reusable capability. The order asymmetry is
consistent with a write-position shortcut rather than learned cue-conditioned
retention. The reward-shuffled control remains near chance.

The next controlled intervention was limited to the existing generic
write-cost regularizer. It did not change the controller interface or expose
verifier labels, but it produced a target-first shortcut in the parent-stable
mini-rung and is not promoted.

The per-run reports are the four JSON files in this directory. The full
accounting is in `sample_efficiency_ledger.json`. No checkpoint or weight from
this population is promoted.
