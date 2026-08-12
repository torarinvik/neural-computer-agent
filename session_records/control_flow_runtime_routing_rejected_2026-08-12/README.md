# Outcome-only multi-file control-flow routing — rejected promotion

This audit exercised the canonical `ControlFlowProgramAmodalRuntime` with two
protected external counter-machine files and the generic
`ExternalOutcomeProgramRouter`. The controller, intention codec, and external
files were frozen. The router received only opaque intention features, exact
logged propensities, and one delayed scalar verifier outcome per fresh training
episode. Per-file counter state, route state, checksummed reload, and corrupted
state rejection were all exercised without replay or controller optimizer
updates.

The four-seed forward/reversed-file audit is rejected as a learned promotion.
Six of eight verifier arms reached `1.0000` held-out accuracy, but both seed-20
arms remained at `0.5000`. The reward-shuffled null also reached `0.0000` or
`1.0000` in several symmetric arms, so it is not a clean causal null. The
result qualifies the ABI and state-isolation seam only; it does not promote
seed-stable learned routing, arbitrary new computation, unrestricted memory
growth, or general continual learning.

The next experiment should remove the controller-feedback distribution shift
from the route-feature audit or evaluate under the same delayed-feedback
protocol, then require a paired null that cannot pass through static route
collapse. The full machine-readable report is
`report_seed-17-20.json`.

Accounting for the verifier arms: `8,000` unique training verifier bits,
`8,000` logical training lifetimes, zero replayed examples, zero controller
optimizer updates, and `9.545` seconds of measured training wall time. The
stable training prefix ranged from `1` to `999` episodes in passing arms and
was not seed-stable.
