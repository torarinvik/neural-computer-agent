# Four-view opaque routing scaling — 2026-08-05

Status: promoted bounded four-view learned-routing boundary.

The system acquired four independent span-4 procedures—`forward`, `reverse`,
`complement`, and `complement_reverse`—then compacted them into one physical
artifact row with four opaque views. The view keys are independent random
opaque storage identities; the router sees only controller-produced query
representations after the first query event, candidate keys, attempted view
pairs, and scalar verifier outcomes.

The promoted router is the permutation-equivariant joint opaque scorer trained
with paired counterfactual credit. Across seeds 69316 and 69317:

- route accuracy: `1.000/0.969`
- candidate-permutation accuracy: `1.000/0.969`
- reward-shuffled route accuracy: `0.215/0.250`
- all four procedures mastered: minimum `0.762/0.809`
- wrong-view causal behavior: passed for both seeds
- one physical row/four views: passed
- reload, exact candidates, checksum rejection, and frozen core: passed
- replayed examples: `0`

The first four-view attempts exposed two real bottlenecks. Context-derived
addresses collided in the bounded store, so addresses were separated into
opaque storage identities. The factorized router and direct attempted-outcome
loss were sample-inefficient at four views; the joint scorer with paired
counterfactual credit resolved that scaling failure.

This promotes four-view routing of already-acquired procedures. It does not
yet prove unrestricted memory growth, arbitrary procedure induction, or
general continual learning without replay.
