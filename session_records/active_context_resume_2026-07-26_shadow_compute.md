# Resume: sample-efficient shadow compute allocation

## Current verified state

Branch: `work`

The latest completed experiment is a passive compute allocator that learns
whether one extra external-memory read is worthwhile. It is trained only from
four controller-created read statistics, a uniformly logged attempted
read/no-read action, exact propensity `0.5`, a normalized read cost `0.01`,
and the attempted action's scalar verifier outcome.

No correct compute action or unattempted outcome enters training.

The direct inverse-propensity advantage objective passed and replicated with a
105-parameter head at 720 unique verifier bits:

- seed 7424: 69.0% choice accuracy, 59.7% oracle-gap capture;
- seed 7425: 70.6% choice accuracy, 60.2% oracle-gap capture.

Both reached the primary stable threshold at the first measured 120-bit
prefix. Reward/feature/zero-evidence controls failed; evidence shuffling made
utility worse than always-read. Old skills and persistence passed.

The inherited 49-parameter gate is still stronger and must not be overwritten.
Its matched audit is saved at:

`session_records/shadow_compute_allocation_2026-07-26/inherited_gate_audit_7424_7425.json`

## Latest GPU result

The exact width-8/24-update experiment ran on the RTX PRO 6000 Blackwell
instance at `47.156.154.165`. It completed in 0.45 seconds and failed only the
choice-accuracy gate:

- `62.70%` choice accuracy versus the required `65%`;
- `35.73%` oracle-gap capture;
- +`0.11238` utility over always-read;
- every causal, retention, persistence, and latency control passed.

The report was copied off the nonpersistent instance with matching SHA-256
`939743ea736787afc1bd1af1dfc6abc184c3f1bf32cee82a9da1a79c3fe40858`.

Do not replicate or further scale width 8. Retain the replicated
105-parameter width-16 blueprint and the stronger inherited production gate.

## Next scientific frontier

Design a new shadow compute decision with a genuinely novel optional operation,
then compare:

1. inherited 105-parameter advantage blueprint;
2. same architecture with reset weights;
3. inherited production read gate where interface-compatible;
4. fixed always/never-compute controls.

The primary measurement is stable unique verifier bits to a verified utility
threshold, with retention and latency as hard gates. This is the next test of
whether the new objective transfers and compounds rather than merely relearns
read/no-read.

## Important files

- `experiments/unified_cognitive_controller/train_shadow_compute_advantage.py`
- `experiments/unified_cognitive_controller/train_shadow_compute_critic.py`
- `experiments/unified_cognitive_controller/audit_inherited_compute_gate.py`
- `session_records/shadow_compute_allocation_2026-07-26/README.md`
- `experiments/forward_transfer_attention/SAMPLE_EFFICIENCY_LEDGER.md`

No training process is running.
