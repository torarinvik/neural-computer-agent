# Stable controller value path — three-slot, two-row bank

Status: promoted narrow two-row retention and persistence rung.

The same controller-native stable value path and isolated external writer were
tested with two durable rows rather than one replacement row.

- target-first: `0.963`
- target-last: `0.940`
- intact: `0.947`
- mastered-parent retention: `0.980`
- unseen-token minimum: `0.945`
- persistent reload: `0.965`
- checksum corruption: rejected
- recovery: `0.938`
- replayed examples: `0`

The two-row result is not evidence for arbitrary episodic memory; it is the
next bounded bank rung after the replicated one-row mechanism.
