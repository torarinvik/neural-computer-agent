# Promoted: delayed evidence with multiple high-similarity candidates

Each seed used one evolving external route-memory store. Every latent identity
arrived as a full first observation, two unrelated but highly similar
distractors, and a delayed partial observation. The planner had to grow or
admit the observations, then consolidate the true pair. A private
copy-on-write verifier rejected every other pair and checked retention of all
previously accepted evidence.

Across seeds `86201`–`86204`, all 12 delayed cycles completed, including a
coordinate reversal after cycle 6. Each run committed 12 compressions and 36
growth transactions, with minimum sampled-prefix and full-final retention of
`1.0`. The trained planner made 95–284 false-consolidation proposals per seed
and committed none; every rejection was atomic. It transferred to all six
unseen-pattern evaluation cycles, while fresh controls completed 0–3. The
reward-shuffled controls required 2,395–3,000 attempts versus 198–387 for
clean training. Each run used one verifier utility and one optimizer update
per attempt, zero replay, and a frozen controller.

This promotes delayed verifier-safe bounded capacity maintenance with
multi-candidate distractor control and generic pattern transfer. It does not
establish arbitrary semantic identity, learned verifier design, unrestricted
memory growth, or general continual learning.
