# Long alternating lifetime pressure stream — promoted

This two-seed audit kept three recurring independently learned factual
transition models alive in a bounded four-slot external bank while repeatedly
replacing a disposable fourth model. Each seed ran `600` training and `240`
held-out pressure events. The verifier tested held-out behavior for every
recurring model after each copy-on-write proposal; rejected learned proposals
used a verifier-authorized disposable-only fallback so the stream could
continue.

Both seeds preserved every recurring capability at every measured prefix and
restored the lifetime policy exactly. Held-out learned safe admission was
`1.000`/`1.000`; the random reference is `0.500`, while recency also reaches
`1.000` on this deliberately stale-disposable stream. Seed 1702 had five
training misses before converging, and all misses preserved the recurring
capabilities through the fallback verifier.

The result promotes long bounded retention under capacity pressure, not a
claim that the learned policy has surpassed a strong recency heuristic. The
controller remained frozen, no old transition examples were replayed, and the
verifier remained authoritative. Unrestricted memory growth, learned
consolidation/compression, and general continual learning remain open.
