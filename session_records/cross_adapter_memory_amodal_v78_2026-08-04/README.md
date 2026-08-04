# v78 random-position cross-adapter memory qualification

v78 removes the recency shortcut from the promoted v77 cross-adapter rung.
The controller receives an ordinary opaque cue before a three-event sequence,
but the cued event is presented at a random position. Memory capacity remains
two rows, the reader adapter is aligned from paired unlabeled event
consistency, and the controller and memory are frozen during reader training.
The optional counterfactual leave-one-out write intervention is used only as
training-time outcome credit; it exposes no target index, verifier bit, or
semantic label to the deployed controller.

Across seeds 17, 18, and 19, writer recall is `0.993–0.998`, reader recall is
`0.990–0.998`, the fresh-reader population minimum is `0.988–0.998`, and the
fresh aligned-reader gain over the raw reader is at least `+0.458` for every
seed. Swapped-slot populations remain at or below `0.512`; clear and corrupt
memory remain near chance; and persistent reload, checksum rejection, and
recovery pass for every seed.

The cue-removal diagnostic remains `0.819–0.838`, while cue-swapping is
`0.719–0.831`. This is deliberately not claimed as cue-conditioned utility:
the evidence suggests the learned controller can compress or route the
outcome sequence without depending strictly on the cue at inference. A
longer fresh no-cue training arm remains below the population promotion gate.
The capacity-one pilot also remains below threshold (`~0.65`), so this is a
bounded two-row result rather than arbitrary episodic compression.

The reward-shuffled control is at chance and is not promoted. The qualified
claim is synthetic outcome-only cross-adapter retrieval with randomized target
position, not natural-modality grounding or general episodic memory.

Per-seed reports, the reward-shuffled control, and the sample-efficiency ledger
are in this directory.
