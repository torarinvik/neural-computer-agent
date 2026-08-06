# Stable value path with persistent reload — seed 17

Status: promoted narrow persistence audit.

The persistent backend used the same isolated external writer as retention.
Reload recall was `0.965`; checksum corruption was rejected; recovery recall
was `0.938`. The underlying three-slot retention result remained target-first
`0.963`, target-last `0.940`, with zero replayed examples.

This qualifies persistence of the narrow synthetic memory boundary, not
general durable episodic memory.
