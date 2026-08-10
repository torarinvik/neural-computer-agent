# Online interleaved reliability and delay

This three-seed pressure test interleaves two factual transition streams. The
reliability statistics are updated only after each route from a scalar
verifier outcome; there is no separate calibration fixture. After four clean
outcomes, the learned committed-slot gate vetoes a low-error corruption in
stream B without staging a new candidate. Clean observations from both streams
then reverse back to their original opaque slots, while genuinely high-error
novel evidence remains eligible for candidate formation. A fresh gate-disabled
control matches the same corruption, establishing that the veto is causal.

Incomplete evidence is updated in the same run. Wait statistics learn a
`0.999665` probability for delayed incomplete evidence and `0.000335` for fast
absence. Across all seeds, the controller and factual bank remain byte-stable,
router/statistics persistence is exact, and replay is zero.

This promotes bounded online interleaved reliability/delay state, not learned
multimodal grounding, unrestricted memory growth, arbitrary new computation,
or general continual learning.
