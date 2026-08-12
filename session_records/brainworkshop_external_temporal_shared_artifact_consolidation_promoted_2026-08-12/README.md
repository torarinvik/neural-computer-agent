# Promoted temporal shared-artifact consolidation

This promotion composes four learned opaque temporal routes with the
replaceable `ExecutableArtifactMemory` backend. Two distinct route artifacts
share a basis but retain separate residual and address views. A generic
`OpaqueConsolidationPolicy` selects that compositional pair from opaque
candidate tensors; an independent verifier then authorizes a copy-on-write
rewrite from four physical rows to three.

The policy controls use address-scrubbed candidate views. This removes an
accidental shortcut in the first audit, where learned temporal address
geometry allowed the reward-shuffled policy to select the target pair. The
deployed transactional streams still use the learned route keys, and the
verifier checks the exact route keys, both retained views, reload, corruption,
and rejection non-mutation.

Seeds `17`, `18`, and `19` pass all gates. Learned shared-pair selection is
`1.0000` across all 24 physical permutations. Address-scrubbed
reward-shuffled controls are `0.0000`, `0.6667`, and `0.1667`; untrained
controls are `0.1667` on every seed. Both forward and reversed physical
orders accept the rewrite, preserve the two distinct route behaviors and the
nearby source alias, save one row, and reduce the serialized artifact bytes
from `10,212` to `8,275`.

Per seed: `4,352` temporal route-verifier bits, `24,576` counterfactual
temporal-verifier bits, `28,928` combined unique temporal bits, `1,344,000`
policy bits, `1,344,000` shuffled-policy bits, `24` shared-view retention
bits, `48,024` policy logical lifetimes, `3,000` optimizer updates, and zero
replayed examples.

This promotes narrow learned shared-view external consolidation for distinct
temporal capabilities. It does not qualify arbitrary semantic compression,
unrestricted memory growth, arbitrary new computation, or general continual
learning. Raw reports are `seed-17.json`, `seed-18.json`, and `seed-19.json`.
