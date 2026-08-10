# Promoted: replay-free mixed-action capacity-policy learning

The planner first learned a bounded consolidation regime, then received a
balanced stream of admission, eviction, consolidation, and growth states
without replaying the pretraining examples. Every episode used a fresh opaque
memory bank and one scalar verifier utility for the proposed action and, when
needed, its selector.

Across seeds `85601`–`85604`, all promotion gates passed. Mixed online utility
rose from `0.900`–`0.910` to `0.975`–`0.995`; eviction selector utility rose
from `0.64`–`0.70` to `0.90`–`0.98`. Held-out utility was at least `0.905`
for every action and `1.0` for admission, consolidation, and growth. The
previous consolidation skill remained at `1.0` after mixed training. The
trained policy beat a fresh policy on the learnable action families, while
growth was retained as a deliberately trivial all-protected control.

Each seed used 2,000 unique verifier utilities, 2,000 optimizer updates, zero
replay, and a frozen controller. This promotes sequential learning and
retention for one bounded capacity-policy family; it does not establish a
universal policy, autonomous verifier design, unbounded memory, or general
continual learning.
