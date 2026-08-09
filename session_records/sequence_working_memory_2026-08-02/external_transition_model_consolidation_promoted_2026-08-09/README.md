# Promoted behavior-verified transition-model consolidation

Two seeds (`70111`, `70112`) test safe physical consolidation. Equivalent
source/duplicate transition models were verified on held-out opaque source and
target transition bundles, then made to share one parameter object while all
three context keys and indices remained addressable. Physical model count fell
from three to two, with zero consolidation optimizer updates and unchanged
held-out factual loss.

A source model and a disjoint target model were then presented to the same
consolidation boundary. Their held-out prediction differences were `0.174` and
`0.126`; both were rejected without mutation. Frozen-controller,
pre/post-retention, wrong-context, alias-persistence, and distinct-function
controls passed.

This promotes verified parameter sharing only. It does not claim semantic
merging, arbitrary compression, or equivalence beyond the supplied held-out
evidence.
