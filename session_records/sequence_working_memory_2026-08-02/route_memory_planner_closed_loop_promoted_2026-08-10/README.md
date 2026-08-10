# Promoted: closed-loop route-memory capacity-policy learning

The opaque capacity planner was connected to real
`ExternalTransitionRouteMemory` transactions. Its proposals now select the
actual replacement row, consolidation pair, or growth operation; every
mutation is copy-on-write and verifier-gated. The pair selector uses generic
coordinate-invariant relations rather than raw feature coordinates.

The stream contained distractor prototypes and two reversed redundancy
patterns. Across seeds `85701`–`85704`, trained planners achieved `1.0` held-out
utility for admission, eviction, consolidation, and growth in both patterns.
The earlier consolidation skill remained at `1.0`, and mixed online utility
remained above `0.98` in the stable tail. The trained policy's aggregate gain
over a fresh planner on the learnable action families was at least `0.333` in
both patterns. Each seed committed 1,598–1,599 of 1,600 verifier-gated mixed
transactions, used 2,000 unique utilities, zero replay, and a frozen
controller.

This promotes closed-loop bounded capacity-policy learning and relational
selector transfer. It does not establish unbounded memory, universal policy
composition, autonomous verifier design, or general continual learning. The
next pressure test is a persistent single-memory stream with nonstationary
interference, reversal, and explicit capacity costs rather than independent
copy-on-write states.
