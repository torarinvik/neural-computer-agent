# Opaque learned consolidation

This audit trains the canonical memory-side consolidation policy from scalar
rewrite utility. The policy sees only learned keys, learned values, strength,
and relative age. It proposes a permutation-equivariant pair and one of three
mechanical operations; an immutable snapshot transaction is adopted only when
the independent verifier and retention gate pass.

The pressure test starts with eight opaque rows arranged as four noisy latent
duplicates. A successful consolidation merges one pair at a time until four
rows remain. The report compares the learned policy with an untrained policy,
a reward-shuffled control, candidate permutation, value corruption, held-out
retention, and zero-replay accounting.

```bash
PYTHONPATH=src uv run python -m experiments.opaque_consolidation_amodal.train \
  --updates 512 --audit-count 512 --seed 69316 \
  --report-out /tmp/opaque-consolidation-seed69316.json
```

This is a memory-side learned consolidation result, not arbitrary program
induction or general continual learning. The verifier remains the authority
for adoption, and the frozen controller is outside the policy boundary.
Both canonical seeds pass the registered gates; archived reports and accounting
are in
`session_records/sequence_working_memory_2026-08-02/opaque_consolidation_v1_2026-08-06/`.
