# Outcome-only contextual contradiction arbitration

This promotion tests a single canonical controller on three simultaneous
standardized event streams: a context event and two candidate events that
always contradict. The hidden rule is stable but private: context `0` trusts
candidate `b`, while context `1` trusts candidate `c`. The learner receives
only frozen independent frontend events, sampled-action propensities, and the
deterministic scalar verifier outcome.

Across seeds 17, 18, and 19, clean rewards were `0.9995`, `1.0000`, and
`1.0000`. Both forced context values and stream-order permutations passed at
approximately `1.0`. Shuffling the hidden context-to-candidate assignment
collapsed to `0.4956`, `0.5044`, and `0.5068`; inverting the visible context
collapsed to `0.0005`, `0.0000`, and `0.0000`. Action and intention
interventions stayed near the two-action chance level. A reward-shuffled
seed-17 negative control remained at chance and did not promote.

This promotes narrow context-to-source contradiction arbitration. It does not
promote natural multimodal grounding, arbitrary temporal trust reversal, or a
general contradiction solver. The sequential temporal follow-up remains an
explicit rejected rung because its post-reversal learning was not stable
across seeds.
