# Promoted: persistent route-memory stream under interference and cost

Each seed ran one evolving route-memory store for 300 events, reversing the
event schedule halfway through. Mastered anchors were retained while
unprotected distractors were admitted and evicted; pressure events required
verifier-gated growth. The planner updated online from one scalar utility per
event and never replayed an old example.

Across seeds `85801`–`85804`, all gates passed. Stable utility ended at `1.0`,
trained streams committed `290`–`297` of 300 transactions, growth occurred
72–74 times, and both sampled-prefix and full-final retention remained `1.0`.
Fresh planners committed zero transactions in three controls and fewer than
the trained planner in the remaining control. The controller remained frozen;
each run used 300 unique utilities and 300 policy updates.

This promotes persistent bounded memory maintenance under interference,
reversal, and growth cost. It does not establish consolidation selection in
every persistent stream, unrestricted memory growth, universal policy
composition, autonomous verifier design, or general continual learning.
