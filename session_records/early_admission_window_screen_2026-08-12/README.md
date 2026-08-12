# Early provisional-admission window screen

Candidate overlap retention exposed a second bottleneck: waiting for the full
six-row admission window can discard the first useful mismatch prefix before
an isolated candidate exists. The existing router was therefore screened at
an explicit three-row admission window, with all promotion, recursive,
retention, source-preservation, and fresh-challenger gates unchanged.

On exact seeds `80–103`, same-cue active discovery improved from `45/72` to
`53/72` complete gates across n-back-3/4/5. The matched passive arm improved
from `38/72` to `45/72`. The different-cue n-back-5 arm was not uniformly
better (`12/24` active versus `13/24` at the conservative window), so the
three-row setting is retained as an opt-in hard-regime experiment rather than
a global default. The old six-row default also remains required by the
canonical seed-93 smoke contract.

All three-row same-cue active runs kept the controller unchanged and source
slot byte-stable (`72/72`), replayed zero examples, and consumed `2,148`
transition rows once. This is a promoted experiment-level evidence-budget
configuration, not a claim of universal continual learning. The next step is
to learn the admission decision from factual surprise/context stability rather
than selecting a fixed row count.
