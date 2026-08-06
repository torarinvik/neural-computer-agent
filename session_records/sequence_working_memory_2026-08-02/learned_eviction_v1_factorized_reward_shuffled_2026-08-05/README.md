# Learned opaque-row eviction — reward-shuffled control

Status: rejected causal control.

Verifier outcomes were shuffled during both parent acquisition and eviction
training. The frozen parent never stabilized and the eviction policy remained
at chance:

- balanced recall: `0.526`
- target-first/last: `0.501`/`0.505`
- random target-first: `0.504`
- replayed examples: `0`
- parent stable: `false`

This control rejects the explanation that paired row interventions or the
candidate-row interface alone create the learned capability.
