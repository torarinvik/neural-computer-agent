# Mixed-domain explicit-modulus audit — rejected promotion

This two-seed in-repository audit is a diagnostic result for the corrected
recipe ABI. It uses random programs only, six slots with domains
`(2, 2, 8, 8, 8, 8)`, no replay, and an instruction encoding that exposes the
arithmetic modulus.

At the 1,500-update rung, the explicit-modulus arms did not meet the stable
`>=0.9` threshold on all required measures. The parallel target was especially
unstable: it rose above `0.98` on one seed at update `300`, then fell below
`0.4` by update `1,500`; the other seed showed a similar non-monotonic curve.
Old-basis length-two and length-four execution also remained below threshold
on at least one seed. This is a rejected promotion, not evidence against the
explicit-modulus ABI; it indicates that the current learner/training mixture
does not yet retain the new structural target while learning mixed domains.

The deterministic boundary is strong and separate: the former global modulus
of eight matches family increments at `[0.5, 0.5, 1.0, 1.0, 1.0, 1.0]`, while
explicit per-slot moduli match at `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`.

Reports were generated from `experiments/recipe_expressibility/audit.py` at
seeds `70421` and `70422` with `1,500` updates, batch size `64`, and evaluation
every `300` updates. The next experiment must isolate modulus learning from
parallel-composition retention rather than widening the architecture again.
