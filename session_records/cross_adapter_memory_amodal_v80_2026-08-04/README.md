# v80 promoted capacity-one cross-adapter memory

v80 closes the v79 fresh-token generalization bottleneck. The counterfactual
write-credit intervention now uses the same randomized opaque-token blocks as
base acquisition, reusing each block for four episodes. The intervention stays
trainer-only: the deployed controller receives only learned event tensors,
opaque attempted actions, and scalar verifier outcomes.

The rung also retains the v79 event-lifecycle correction: action preview does
not advance state, and the outcome-bearing payload is inserted exactly once.
Memory capacity is one, the cue arrives before three rows, and the target row
is randomized. Stable-content prior binding and an explicit write threshold
remain enabled.

Seeds 17, 18, and 19 all pass the promotion gate. Writer recall is
`0.935/0.998/0.998`; reader recall is `0.971/0.996/0.999`; fresh-reader
minimum recall is `0.991/0.996/0.944`; and fresh aligned-vs-raw minimum gain is
`+0.494/+0.466/+0.421`. Clear, corrupt, swapped-slot, and random-action
controls remain near chance. Persistent reload/recovery are
`0.980/1.000/1.000`, and checksum corruption is rejected for every seed.

The reward-shuffled control remains at chance (`0.504` writer and reader, with
negative fresh aligned-vs-raw gain), supporting causal use of verifier
outcomes. This promotes a narrow synthetic outcome-only capacity-one,
cross-adapter retrieval capability with randomized target position. It does
not establish natural-modality grounding or general episodic memory.

Exact summaries and required accounting are in `reports.json` and
`sample_efficiency_ledger.json`.
