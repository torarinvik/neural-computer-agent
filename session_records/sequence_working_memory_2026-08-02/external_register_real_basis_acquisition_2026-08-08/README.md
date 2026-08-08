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

## Growth execution follow-up — rejected

The selected slot-3 growth branch was then trained on held-out `prefix_parity`.
Both seeds reached high final target accuracy (`0.9766` and `0.9453`) and
retained all source capabilities, with the old basis digests unchanged.
However, neither reached a stable-prefix threshold, and shuffled-outcome
controls remained above the rejection floor (`0.9922` and `0.9531`). The
growth result is therefore rejected for promotion. The next bottleneck is
causal credit/verification dependence in new-slot acquisition, not retention
or append-only capacity.

The causal follow-up switched only new-slot training to `attempted_bce`, so
the optimizer received delivered scalar outcomes rather than verifier-private
correct-action utilities. Shuffled-training controls then collapsed to
`0.4766` and `0.5000`, confirming causal dependence. Normal target accuracy
remained `0.9375` and `0.9063`, with source retention intact, but stable-prefix
promotion still failed. The corrected result remains rejected for stability,
while the credit-path repair itself is retained.
