# Near-boundary recovery and explicit quarantine saturation

This three-seed audit tests the failure boundary that remained after the
long-horizon lifecycle result. A frozen factored slot receives deliberately
near-tolerance drift: the evidence is close enough to resemble a valid match,
but replay-free verifier outcomes teach the separate reliability state to
veto it.

The first two vetoes are retained in bounded quarantine. The third veto is
rejected explicitly when quarantine is full (`status="reliability_veto"` and
`quarantine_accepted=false`), rather than being collapsed into ambiguous
state. Later verifier reversal resolves the retained evidence and the same
drift routes back to the original slot without candidate staging or fact-bank
mutation. Persistence and frozen-core controls pass for all three seeds.

This promotes explicit bounded overflow accounting and near-boundary recovery;
it does not establish unrestricted memory growth, arbitrary new computation,
or general continual learning.
