# Promoted: verifier-safe false-consolidation control

Each fresh route-memory state contained one true redundant pair and one
unrelated pair with higher raw key cosine. The true pair shared a generic
evidence-mask pattern; the distractor did not. All rows were protected, so the
planner had to choose consolidation or growth. A copy-on-write retention
verifier rejected every non-target pair and checked that rejected proposals
left the live memory unchanged.

Across seeds `86101`–`86104`, online planner utility improved from
`0.07`–`0.26` to `0.98`–`1.0` in the measured windows. The trained planner
achieved `1.0` utility on both training mask patterns and on an unseen third
pattern. It made `320` false-consolidation proposals during training and
committed zero of them; all rejected proposals were atomic. Fresh unseen
controls scored `0.0`–`0.17`. Each run used 1,200 unique utilities, 1,200
optimizer updates, zero replay, and a frozen controller.

This promotes verifier-safe bounded consolidation control and transfer across
generic evidence patterns. It does not establish arbitrary semantic
equivalence, learned verifier design, unrestricted memory growth, or general
continual learning.
