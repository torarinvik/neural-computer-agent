# v77 persistent bounded cross-adapter retrieval qualification

v77 promotes a narrow, outcome-only cross-adapter memory boundary across three
seeds. A writer stores three opaque event/outcome rows in a capacity-two
memory; a separately trained reader adapter is aligned from paired unlabeled
event consistency; and the controller and memory remain frozen during reader
training. The run also round-trips memory through an atomic persistent
snapshot and rejects a checksum-corrupted snapshot.

The protocol includes an opaque target cue and presents the cued row last.
The canonical memory contract now separates strict write collision matching
(`0.95`) from learned-IR read matching (`0.75`): a near-miss read returns no
value instead of the nearest occupied row. This prevents bounded-memory
swapped-slot hallucination while retaining aligned reader retrieval.

All seeds pass fresh-token writer/reader minima, positive reader-vs-raw gain
for every population pair, swapped-slot, clear, corruption, and persistent
reload/recovery controls. The reward-shuffled control remains at chance.
This promotes synthetic outcome-only cross-adapter retrieval with bounded
three-slot/two-row interference; it does not claim natural-modality grounding,
arbitrary raw frontend alignment, or general episodic memory.

The compact per-seed records and accounting ledger are in this directory.
