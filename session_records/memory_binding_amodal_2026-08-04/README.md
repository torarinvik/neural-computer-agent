# Outcome-only two-slot binding audit

This audit tested the next memory bottleneck after scalar recall: two opaque
slot outcomes stored in a two-row memory, queried after recurrent-state reset,
with four independent batch scopes. The fixed-write setting isolated content
binding and scope isolation from the separate learned-retention question.

The backend boundary passed its direct contract tests: identical keys in two
scopes retrieve different values, transactions preserve gradients, and scoped
snapshots round-trip. The controller qualification did not pass. At 128
updates, seeds 17, 18, and 19 reached intact recall `0.5234`, `0.7266`, and
`0.5625`; swapped-slot recall was `0.5313`, `0.8359`, and `0.5625`. The three
seed population is therefore rejected: the controller has not learned stable
content-key binding under two-row interference.

This rejects only the two-slot learned-binding claim. It does not invalidate
the scalar outcome-recall promotion or the implementation-level batch-scope
contract. The next attempt should reduce the curriculum jump or add a direct
key/value binding diagnostic before retraining a full controller.
