# Learned opaque-row eviction — seed 69415

Status: promoted replicated learned-eviction rung.

The same frozen-controller protocol was repeated with an independent seed.
Parent acquisition used fresh opaque event tokens; eviction training used
target-first and target-middle paired counterfactual factors.

- held-out balanced recall: `0.963`
- target-first recall: `0.912`
- target-last recall: `0.999`
- strength-eviction target-first baseline: `0.512`
- random target-first control: `0.756`
- clear-memory/corruption controls: `0.515`/`0.511`
- persistent reload/recovery: `0.911`/`0.875`
- checksum corruption rejected: `true`
- replayed examples: `0`

The controller remained frozen during eviction learning. This is a narrow
replication of learned utility-based eviction, not general continual learning
or arbitrary new computation.
