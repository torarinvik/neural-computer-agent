# Stable controller value path — five-slot reward-shuffled control

Status: rejected causal control.

The five-slot/two-row protocol was run with verifier outcomes shuffled before
the learner saw them.

- intact: `0.476`
- target-first/last: `0.514`/`0.503`
- mastered-parent retention: `0.473`
- unseen-token minimum: `0.488`
- parent stable: `false`
- replayed examples: `0`

The result remains at chance and fails the capability gates, supporting the
claim that the ordinary five-slot result depends on the outcome signal rather
than the intervention scaffolding alone.
