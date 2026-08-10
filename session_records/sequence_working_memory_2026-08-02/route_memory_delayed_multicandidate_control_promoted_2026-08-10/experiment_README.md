# Delayed evidence with multiple high-similarity candidates

This experiment keeps one evolving external route-memory store. Each latent
identity arrives as a full first observation, two unrelated but highly
similar distractors, and a delayed partial observation. The planner must grow
or admit the observations, then consolidate the true pair after the delayed
evidence arrives. A private copy-on-write verifier rejects every other pair
and checks retention of all previously accepted evidence.

The stream alternates generic evidence-mask patterns and reverses coordinates
midway. Promotion requires persistent prefix retention, transfer to an unseen
mask pattern, zero false-consolidation commits, atomic rejection, a frozen
controller, and zero replay. The claim is deliberately bounded: this tests
delayed verifier-safe capacity maintenance, not arbitrary semantic identity
or general continual learning.
