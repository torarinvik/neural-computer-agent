# Replay-free learned reliability and delay

This three-seed pressure test separates immutable factual memory from two
plastic external policies. `ExternalTransitionEvidenceStatistics` learns from
scalar verifier outcomes and can veto a committed route, while
`EventWaitStatistics` learns whether incomplete timestamped evidence should be
held or released. Neither component receives raw modality formats, task
labels, or historical model rows.

The factual source model is first fit and then held fixed. A low-error
corrupted revisit is deliberately inside the factual match tolerance, so a
fresh gate-disabled control routes it. The learned gate rejects it, preserves
the source slot, and routes a later clean reversal back to the original slot.
All three seeds also pass exact persistence and frozen-controller checks.

The learned wait probabilities were `0.999665` for a delayed incomplete
window and `0.000335` for a fast-absence window. Each seed consumed 128
reliability outcomes and 128 wait outcomes once, with zero replay and zero
controller updates.

This promotes a bounded replay-free reliability/delay boundary, not learned
multimodal grounding, unrestricted memory growth, arbitrary new computation,
or general continual learning.
