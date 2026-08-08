# External basis reuse probe — 2026-08-08

This probe tests whether a mastered external computation slot can be reused
by a fresh opaque instruction without replaying prior examples or updating the
slot. The first `rotate` instruction is learned with one appended basis slot;
the second instruction is trained from fresh outcomes while that slot is
frozen. A matched fresh learner is trained for the same reuse budget.

| seed | reused stable bits | fresh stable bits | reused accuracy | fresh accuracy |
| --- | ---: | ---: | ---: | ---: |
| 69316 | 4,096 | 12,288 | 0.9531 | 0.8555 |
| 69317 | 4,096 | 8,192 | 0.9805 | 0.9375 |

Both seeds passed basis admission, first and reused mastery, first-capability
retention, frozen-slot digest, shuffled-outcome, missing-evidence, exact
reload, frozen-parent, and zero-replay gates. This promotes bounded reuse of
mastered external computation, not general continual learning or unrestricted
new computation.

## Distinct-operation follow-up

The same frozen `rotate` basis was reused for a fresh `global_parity`
instruction. All safety and retention controls passed, but strict positive
transfer did not replicate:

| seed | reused stable bits | fresh stable bits | result |
| --- | ---: | ---: | --- |
| 69316 | 16,384 | 8,192 | rejected: slower than fresh |
| 69317 | 4,096 | 8,192 | positive transfer |

The longer rung therefore rejects the cross-operator promotion gate. The
remaining bottleneck is learned compatibility/routing across distinct
primitive families, not slot persistence or same-family reuse.

The efficiency-aware admission rerun makes the correct route explicit:

| seed | correctness decision | efficiency decision | reason |
| --- | --- | --- | --- |
| 69316 | reuse | grow | reused cost 16,384 exceeded fresh 8,192 |
| 69317 | reuse | reuse | reused cost 4,096 beat fresh 8,192 |

This is a policy result, not yet a full grow-and-retrain experiment: the next
audit must actually append and train the new slot on the `grow` branch.
