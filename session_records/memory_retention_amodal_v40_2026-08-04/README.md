# Longer retention stress test rejection

The first longer 2,048-update retention run used the v39 counterfactual
write-utility protocol without parent rehearsal or parent-aware checkpoint
selection. Retention itself was excellent (`0.998` intact, `0.995` reversed,
`0.996` target-first, `0.995` target-last; clear-memory `0.521`), but the
mastered single-event primitive fell to `0.738` and unseen-token retention was
`0.699`.

This run is rejected as a promotion because it violates the required
retention-on-mastered-primitives control. The failure motivated v41's explicit
parent rehearsal and joint validation selector. The raw report is
`seed_19.json`; no checkpoint or capability claim is promoted from v40.
