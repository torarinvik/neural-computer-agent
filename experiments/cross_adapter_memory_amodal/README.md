# Outcome-only cross-adapter memory retrieval

This rung tests the memory contract's cross-adapter boundary. A writer event
adapter stores opaque outcome rows. A separate reader adapter sees a fixed
orthogonal transformation of the same latent events and must align them back
into the shared neural IR. The controller and memory are frozen during reader
alignment; only the generic reader adapter is trainable.

Reader alignment uses paired opaque event consistency, not verifier-private
targets, actions, slot IDs, or semantic labels. Scalar verifier outcomes then
test whether the aligned reader can retrieve and use writer-side memory. The
raw-reader control is retained, and the gate uses fresh-token aligned-vs-raw
gain rather than a fixed raw score alone.

The v75 two-row rung first qualified the neural-IR adapter boundary. The v77
three-seed persistent rung now qualifies three-row retrieval with an opaque
target cue and cued-row-last presentation. Fresh aligned-reader minima are
`0.996`, `0.997`, and `1.000`; aligned-vs-raw mean gains are `0.496`, `0.505`,
and `0.515`; swapped-slot maxima are `0.511`, `0.509`, and `0.520`; and persistent reload,
recovery, and checksum rejection pass for every seed. This qualifies
synthetic outcome-only cross-adapter retrieval, not natural-language
grounding, arbitrary raw-modality alignment, or general episodic memory.

The three-slot/two-row bounded-interference variant now passes after the
canonical memory backend separated strict write collision matching (`0.95`)
from learned-IR read matching (`0.75`), causing near-miss reads to return no
value. Evidence is in
`session_records/cross_adapter_memory_amodal_v77_2026-08-04/`.

The v78 follow-up removes the remaining recency shortcut: the opaque target
cue arrives first, but the target row is randomized within the three-row
sequence. A generic counterfactual leave-one-out write intervention supplies
trainer-only outcome credit. Across seeds 17, 18, and 19, fresh-reader minima
are `0.991`, `0.988`, and `0.998`; fresh aligned-vs-raw gain minima are
`0.476`, `0.458`, and `0.482`; and persistent reload, recovery, checksum
rejection, clear-memory, corruption, and swapped-row controls pass. Cue
removal and cue swapping remain diagnostics rather than a cue-conditioned
selection claim. Evidence is in
`session_records/cross_adapter_memory_amodal_v78_2026-08-04/`.

Run a short rung with:

```bash
PYTHONPATH=src uv run python -m experiments.cross_adapter_memory_amodal.train \
  --base-steps 512 --alignment-steps 512 --adapter-steps 512 \
  --slot-count 3 --memory-capacity 2 --persistent-memory-audit \
  --randomize-event-tokens --random-orthogonal-basis --target-cue \
  --randomize-slot-order --writer-intervention-steps 512 \
  --seed 17 --report-out /tmp/cross-adapter.json
```
