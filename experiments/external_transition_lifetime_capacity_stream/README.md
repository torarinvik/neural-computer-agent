# Long alternating lifetime pressure stream

This audit keeps three independently learned factual transition models in a
bounded four-slot bank while repeatedly replacing a disposable fourth model.
The recurring models are accessed between pressure events; the disposable
model is intentionally left stale. Each learned lifetime proposal is checked
by a held-out retention probe for all recurring models. A rejected proposal is
not silently counted as success: a verifier-authorized fallback removes only
the disposable model so the stream can continue.

The controller is frozen and never constructed. The external bank owns
usage, age, and prediction-error telemetry. The policy receives one verifier
bit per pressure event and no transition replay. Promotion requires both
seeds to preserve every recurring capability at every measured prefix and to
improve learned admission over random eviction; recency is reported as a
strong safety baseline rather than treated as a learning claim.

The promoted run archived under
`session_records/sequence_working_memory_2026-08-02/external_transition_lifetime_capacity_stream_promoted_2026-08-09/`
preserved all recurring capabilities for both seeds. Learned safe admission
was `1.000`/`1.000`, matching recency and beating random `0.500` because this
fixture makes the disposable model stale by construction. The result is a
retention and lifecycle-integrity promotion, not evidence that the learned
policy is superior to recency.
