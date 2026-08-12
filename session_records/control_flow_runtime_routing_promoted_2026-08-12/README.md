# Outcome-only multi-file control-flow routing — promoted narrow result

This audit exercised the canonical `ControlFlowProgramAmodalRuntime` with two
protected external counter-machine files and the generic
`ExternalOutcomeProgramRouter`. The controller, intention codec, and external
files were frozen. The router received only opaque controller intention
features, exact logged propensities, and one delayed scalar verifier outcome
per fresh training episode through the optional route-only feedback channel.
The controller received quiet feedback in both training and evaluation, so the
router did not create a controller-feature distribution shift.

Across four seeds and both forward/reversed physical-file orders, all eight
verifier arms reached `1.0000` held-out accuracy. The paired
reward-shuffled order-permutation null was exactly `0.5000` for every seed.
Per-arm shuffled scores sometimes collapsed to `0.0000` or `1.0000`; retaining
the paired permutation as an explicit gate prevents those symmetric policies
from being misread as causal success.

The result promotes bounded outcome-only routing among two generic external
files with isolated per-file state, frozen controller/codec, zero replay, and
zero controller optimizer updates. It does not establish learned codec
adaptation, arbitrary new computation, unrestricted memory growth, or general
continual learning.

The full machine-readable report is `report_seed-17-20.json`.
