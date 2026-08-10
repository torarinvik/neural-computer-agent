# Dynamic-regime factual versioning

Three seeds (`96001`, `96002`, `96003`) passed a reversal-cycle pressure test
for the online external factual memory. One opaque stream first admitted
regime A, then supplied two contradictory rows for regime B. The first
contradiction remained uncommitted; the second allocated B at a new address.

All seeds then returned to A and B. Both versions were routed exactly from
the retained facts, without new writes or address growth. Serialization and
restore preserved both routes. The controller remained byte-stable and had
zero optimizer updates; replayed examples were zero. Fresh and corrupted
memory controls failed factual retrieval as expected.

This promotes bounded same-stream factual versioning and reactivation. It does
not establish arbitrary regime discovery, unbounded memory growth, learned
compression, or general continual learning.
