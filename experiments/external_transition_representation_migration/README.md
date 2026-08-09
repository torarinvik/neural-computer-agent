# Verified external representation-space migration

This audit pressure-tests the replacement boundary exposed by the exported
session: a controller or frontend may be replaced without changing tensor
widths, but equal width does not imply equal meaning.

Each seed constructs a factual external bank, creates a copy-on-write
candidate in new state/intention spaces, and verifies it against held-out
transition observations. The unchanged candidate is accepted; a candidate
with one changed factual model is rejected. The old planner is also rejected
against the new bank until a planner carrying the matching space IDs is used.

This promotes an explicit compatibility gate and verified migration contract,
not arbitrary representation alignment or general continual learning.
