# Ordered external state pressure test

This record compares the historical pooled event-window state with the
opt-in `ordered_payload_and_presence_v1` external state contract. Both arms
use the same rendered Brain Workshop online-discovery harness, frozen
controller/frontend/decoder, zero replay, held-out one-step and recursive
rollout gates, source-slot retention, matched fresh challenger, and opaque
goal-fragment gate.

The ordered arm preserves bounded learned event-token order and empty
positions. With factual routing tolerance `0.005`, it improves raw candidate
admission from `3/8` to `5/8` and complete end-to-end promotion from `3/8` to
`4/8`. This is a mechanistic improvement, not a promoted continual-learning
capability: three of eight ordered runs still fail the complete gate.

The retained architectural lesson is that an external factual memory needs a
state representation that can preserve temporal order when order is part of
the learned state. The unresolved implementation bottleneck is calibrating
route acceptance from generic verifier residuals rather than a fixed absolute
threshold.
