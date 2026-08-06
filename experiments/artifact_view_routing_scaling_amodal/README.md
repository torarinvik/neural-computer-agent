# Four-view opaque routing scaling

This audit scales the promoted one-row/two-view contract to four
independently acquired procedures at the same sequence span. The procedures
are `forward`, `reverse`, `complement`, and `complement_reverse` over span
four, so the controlled difficulty change is the number of executable views.

An opaque permutation-equivariant router learns view selection from fresh
controller query tensors, random opaque candidate keys, attempted view pairs,
and scalar outcomes. It does not receive operation names, task IDs, or correct
unattempted choices. The query includes the controller’s learned memory-query
representation after the first query event, which is the earliest point at
which the procedure cue exists.

The promotion gates include four-view permutation equivariance, reward-
shuffled routing, wrong-view causal behavior, persistent reload, checksum
corruption, frozen-core equality, and no replay. This is still a bounded
scaling result, not general continual learning or unrestricted program growth.
