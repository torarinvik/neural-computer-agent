# Real external-register basis acquisition — 2026-08-08

Three opaque source primitives (`rotate`, `global_parity`, `complement`) were
trained into independent external basis slots. Their fresh verifier outcome
matrix was used to update the compatibility prior once, then a held-out
`prefix_parity` acquisition was routed through the live register scheduler.

Both seeds produced distinct source outcome rows, preserved fresh-verifier
admission as the authority, and correctly found no passing existing basis for
the unseen target. The target therefore requested growth rather than being
incorrectly reused. No replayed examples were used.

This promotes real multi-slot opaque acquisition and no-false-admission
behavior. It does not yet demonstrate positive transfer to a genuinely new
primitive; the correct result here is verified growth.
