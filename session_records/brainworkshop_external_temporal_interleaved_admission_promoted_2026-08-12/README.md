# Promoted interleaved replay-free capability admission

This promotion interleaves fresh temporal-route learning with external memory
pressure. Six new opaque query/address routes are learned from fresh scalar
counterfactual probes and admitted immediately into a three-slot active cache.
The long-term episodic archive grows to seven records; prior routes are
revisited between admissions and cold routes return through copy-on-write,
verifier-gated reactivation.

The controller, learned event encoder, and acquired capability file are frozen
after source acquisition. The learner sees learned event tensors, opaque
actions, and scalar outcomes only. Query symbols, depths, route identities,
positions, semantic names, and old training streams remain outside the learner
boundary.

Seeds `17`, `18`, and `19` pass all `20/20` gates. Each performs 14
replacements and 14 genuine learned-policy victim choices. All fresh
admissions and reactivations are accepted; the protected source is never
evicted; active, archive, and 20%-related-key route accuracy is `1.0000`.
Held-out victim-policy accuracy is `0.9336`, `0.9297`, and `0.9199`; the
reward-shuffled controls are `0.3027`, `0.2070`, and `0.3418`.

Per seed: `171,776` unique verifier bits, `48,128` counterfactual-arm bits,
capacity-retention bits of `25,344`, `25,920`, and `25,728`, `3,000` policy
updates, seven archive records, three active slots, and zero replayed
examples.

This promotes bounded replay-free interleaved capability admission under
capacity pressure. It does not establish unrestricted memory growth, learned
compression, arbitrary new computation, or general continual learning. Raw
reports are `seed-17.json`, `seed-18.json`, and `seed-19.json`.
