# Promoted replay-free temporal capacity pressure

This promotion pressure-tests a five-route externally acquired temporal bank
with a two-slot active cache. A memory-side policy chooses opaque victims from
learned signatures and generic reliability/age telemetry. Archived routes are
reactivated with copy-on-write retention verification; the mastered source
route is protected after an outcome-only stable prefix.

The controller, learned event encoder, and acquired capability file are frozen
throughout. The learner receives learned event tensors, opaque keys, and scalar
verifier outcomes. It does not receive query depth, route position, semantic
names, task IDs, or replayed old streams.

Seeds `17`, `18`, and `19` pass all `17/17` gates. The archive retains all five
routes while the two-slot cache performs 11 replacements and 3 active no-op
probes. All reactivations are accepted, the protected source is never evicted,
and every final active route remains at accuracy `1.0000`. The generic victim
policy reaches held-out accuracy `0.9414`, `0.9590`, and `0.9727`; the
reward-shuffled controls reach `0.5723`, `0.5723`, and `0.4980`.

Per seed: `169,600` unique verifier bits, `12,224` capacity-retention verifier
bits, `3,000` policy verifier bits, `3,080` unique logical lifetimes, `4,024`
optimizer updates, five archive records, two active slots, 11 replacements,
and zero replayed examples.

This promotes bounded replay-free capacity pressure and verifier-gated
reactivation. It does not establish unrestricted memory growth, learned
compression, arbitrary new computation, or general continual learning. The
complete raw reports are `seed-17.json`, `seed-18.json`, and `seed-19.json`.
