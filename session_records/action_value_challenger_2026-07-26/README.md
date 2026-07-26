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

## Initial result

- hard seed 7973: proposed at 240 bits, confirmed at 480, utility `+7.98`
  points, mastered incumbent unchanged;
- fresh seed 7982: proposed at 480, confirmed at 720, utility `+4.47` points,
  mastered incumbent unchanged;
- fresh seed 7981: safe but no positive proposal within 720 bits.

The learner is substantially stronger but not yet consistent enough for
persistent integration.

## Fixed-basis consistency fork

The action-value head's hidden basis was previously initialized from each run
seed, coupling environment randomness to an optimizer lottery. The next
candidate fixes that basis with task-agnostic seed `424242`; its output remains
zero-initialized, so no policy or task knowledge is injected.

First replay seeds 7973, 7981, and 7982 as a stability diagnostic. If all three
protect mastery and confirm a gap promotion by 720 bits, run untouched seeds
7991 and 7992. Only five-of-five passes authorize persistent integration.

## Fixed-basis result and width fork

The fixed 16-unit basis remained safe but produced no positive proposal on any
of seeds 7973, 7981, or 7982. That closes basis stabilization alone.

The learner is only about one hundred parameters and may be under-capacity for
stable action-value regression. The next gradual candidate increases only its
hidden width from 16 to 64 (still fewer than 500 trainable parameters), keeps
fixed seed `424242`, and preserves every data, budget, active-selection, and
confirmation setting.

It must pass the same three diagnostic seeds before untouched seeds 7991 and
7992 are allowed. Five-of-five passes remain required for persistence.
