# Outcome-only context-dependent relevance

This rung tests whether the canonical controller can select relevant event
content when candidate identity changes per episode. Event `a` carries a high
bit and a context tag. Events `b` and `c` carry independent low bits; exactly
one candidate's tag agrees with `a`, and that candidate supplies the target
low bit. The relevant candidate is hidden and randomized. All raw frontends
are frozen before outcome training.

The trainer uses an exactly balanced hidden-assignment curriculum within every
batch. The controller receives no assignment, target, correct action, or
semantic label—only opaque event tokens, sampled-action propensity, and scalar
verifier reward.

Across seeds 17, 18, and 19, clean reward is `0.9995`, `0.8730`, and `0.9995`;
the two forced assignments, candidate swap, and stream-order controls all pass
the `0.80` gate. Cross-episode candidate shuffling is `0.5093`, `0.5142`, and
`0.5137`, while action and intention interventions remain near chance. A
reward-shuffled seed reaches only `0.2534` clean reward and does not promote.

This promotes narrow synthetic context-dependent relevance through the
canonical event boundary. It does not qualify broad natural cross-modal
grounding, arbitrary contradiction resolution, or general missing-stream
inference.
