# End-to-end online active probe resolution — promoted

This three-seed audit exercises the complete request–execute–observe seam.
An ambiguous opaque window requests a read-only probe from the online router;
the caller executes the returned intention in a hidden regime; the noisy
consequence is submitted to the ordinary factual router. The fixture uses
eight opaque one-hot intention candidates, one informative candidate, two
hidden regimes, outcome noise standard deviation `0.1`, and factual routing
tolerance `0.5` with margin `0.05`. A random-intention control uses the same
factual resolution rule.

| metric | seed 83201 | seed 83202 | seed 83203 |
| --- | ---: | ---: | ---: |
| active resolution rate | 1.000 | 1.000 | 1.000 |
| active routing accuracy | 1.000 | 1.000 | 1.000 |
| random resolution rate | 0.145 | 0.117 | 0.137 |
| random routing accuracy | 0.145 | 0.117 | 0.137 |
| controller updates | 0 | 0 | 0 |
| old-regime replay | 0 | 0 | 0 |
| exact persistence | true | true | true |

All seeds pass the integrated gate. The controller and factual bank remain
unchanged; the router receives no regime label, and the active probe is
selected solely from opaque model disagreement.

Claim boundary: this qualifies a bounded synthetic online probe integration.
It does not establish learned probe selection, multimodal probe usefulness,
arbitrary irreversible environments, or general continual learning.

Reports are protected by `SHA256SUMS`.
