# Action-conditioned challenger — pre-registration

## Motivation

Active disagreement allocation made the hard seed produce a proposal, but
fresh confirmation rejected it. Private audit showed the scalar-advantage
challenger improved utility by only `2.7` points and largely collapsed toward
one action. The attempted-advantage target is high variance because it derives
an action difference from one centered scalar outcome.

## Candidate

Replace only the challenger learner with a two-output action-value head:

`Q(generic_memory_evidence, ordinary_read)` and
`Q(generic_memory_evidence, requery)`.

Each randomized attempted outcome trains only the value corresponding to the
action actually taken. The policy emits the latent concept represented by
`Q(requery) - Q(ordinary_read)`. There are no correct-action labels,
unattempted outcomes, task identities, or game-state hooks.

The incumbent remains immutable until a proposal and a disjoint fresh
confirmation both have positive lower-95% bounds. Active pool multiplier four,
720 verifier bits, 2,880 unlabeled contexts, and every other setting remain
unchanged.

## Hard-stream gate

Seed 7973 passes only if:

1. mastered incumbent receives no promotion and retains utility;
2. the action-value challenger proposes by 480 bits;
3. fresh confirmation promotes it by 720 bits;
4. audited utility improves by at least `0.03`;
5. retention and exact persistence pass.

Only a complete pass permits unchanged seeds 7981 and 7982. Both must pass
before durable skill integration.
