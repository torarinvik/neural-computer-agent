# Long alternating nonlinear address shift — promoted

This three-seed rung tests the remaining learned-address bottleneck. A
context encoder was trained on the two source regimes and then frozen. Each
novel target received an isolated copy-on-write address adapter that learned
from its current evidence windows while remaining separated from committed
keys. The associated nonlinear factual model was promoted only after held-out
prediction and retention verification.

The stream acquired four nonlinear targets from `32/64` presented rows each,
then revisited every target after intervening regimes. All revisits routed to
existing slots; no duplicate target slots were created.

| seed | target held-out MSEs C/D/E/F | final address version | corruption |
| ---: | --- | :---: | :---: |
| 82101 | 6.23e-5 / 3.09e-4 / 3.54e-4 / 1.41e-3 | 20 | rejected |
| 82102 | 3.71e-4 / 6.97e-4 / 1.06e-3 / 2.65e-4 | 20 | rejected |
| 82103 | 3.44e-4 / 6.78e-4 / 3.96e-4 / 3.94e-3 | 20 | rejected |

All gates passed: source and target factual slots retained their verified
behavior, historical keys were unchanged by address updates, all repeated
streams matched existing slots, corrupted evidence staged but failed held-out
verification without a bank write, the controller remained frozen, raw
candidate rows were not retained, and persistence was exact. Replayed
examples and old-regime replay were zero.

Claim boundary: bounded long-horizon nonlinear factual-memory routing with
copy-on-write learned address versions. This is not unrestricted memory
growth, arbitrary new computation, or general continual learning; the model
basis and capacity are finite, and the address learner is still driven by a
router-detected factual mismatch rather than autonomous open-world discovery.
