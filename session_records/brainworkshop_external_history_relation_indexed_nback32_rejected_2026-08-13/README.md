# Relation-aware indexed n-back-32 reader (2026-08-13)

Status: **rejected**.

This experiment added an opt-in generic relation encoder over each relative-age
history slot and the current event. The encoder used shared weights across age
slots and second-order opaque features (slot value, current value, difference,
product, and presence). It did not alter the controller or frontend.

The n-back-32 smoke rung reached `0.78125` on both held-out lifetimes, exactly
the four-symbol majority baseline, with a training tail around `0.63–0.69`.
The flat `history_indexed` reader produced the same result at this budget, so
the relation branch showed no causal learning signal. Missing/corrupted,
action-shuffled, depth-shift, and reward-shuffled controls remained below the
mastery threshold, but that does not rescue the absent fresh-task gain.

The branch is retained as an explicitly opt-in diagnostic ABI for future
comparison; it is not promoted, not a canonical default, and not evidence for
learned comparison or general continual learning. The next successful gain
came from curriculum plus isolated append-only external files, archived at
`session_records/brainworkshop_append_only_nback32_depth_promoted_2026-08-13/`.
