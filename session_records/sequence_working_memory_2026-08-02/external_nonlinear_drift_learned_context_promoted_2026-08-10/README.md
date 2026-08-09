# Learned-context nonlinear drift — promoted

This three-seed rung extends the partial affine-drift result to nonlinear
transition memory. A permutation-invariant `ExternalTransitionContextEncoder`
was trained only on two source bundles, frozen, and used to form opaque keys
for two later nonlinear drift regimes. The target regimes arrived one row at
a time; only `32` of `64` available rows were presented and each staged
evidence window was consumed once by replay-free random-feature sufficient
statistics.

| seed | target-C held-out MSE | target-D held-out MSE | source return | corruption |
| ---: | ---: | ---: | :---: | :---: |
| 82001 | 1.31e-4 | 3.57e-4 | pass | rejected |
| 82002 | 6.70e-4 | 3.27e-4 | pass | rejected |
| 82003 | 1.51e-3 | 9.83e-5 | pass | rejected |

All gates passed: source and target factual slots were promoted and retained,
the controller stayed frozen, source re-routing selected the original slot,
corrupted evidence staged but failed held-out verification without a bank
write, raw provisional rows were not retained, and router persistence was
exact. The context encoder used `400` external optimizer updates before the
target stream and no target examples were replayed during model adaptation.

Claim boundary: bounded replay-free nonlinear drift retention with a
source-trained permutation-invariant learned address. This is not unrestricted
memory growth, arbitrary new computation, or general continual learning.
