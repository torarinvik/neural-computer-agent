# Replay-free nonlinear factored drift retention

This archive records the promoted five-seed pressure test for the external
factored residual boundary. A nonlinear shared base was trained on `40`
opaque source rows and frozen. The external memory then received a target
regime through `20/24` online rows, followed by `6` rows of nonlinear drift;
four independent rows were held out for the target and four for drift.

The promoted residual family was
`random_feature_sufficient_statistics_v1`: a fixed nonlinear feature map plus
normal-equation sufficient statistics. It consumed each online row once and
used zero residual optimizer updates. The controller, shared base, and
context encoder were frozen. All seeds (`81021` through `81025`) passed:

- source, target, and drift promotion on independent held-out evidence;
- prior-target retention after the drift update;
- alternating opaque routing `[0, 1, 0, 1, 0, 1]` and persistence;
- corrupted drift rejection without committed-model mutation;
- frozen-component digest checks; and
- drift held-out MSE below the frozen-base-only control.

Across the five seeds, mean drift held-out MSE was `0.001094` for the
replay-free learner versus `0.012400` for the frozen-base-only control. Mean
retained target MSE after drift was `0.009989`. A fresh learner was measured
with actual held-out loss, but this fixture makes no positive-transfer claim.

This promotes bounded smooth nonlinear drift retention. It does not establish
unrestricted memory growth, automatic context formation, arbitrary learned
computation, or general continual learning.

The optimizer-based nonlinear MLP variant was explored and rejected because
sparse no-replay updates were not reliably better than the frozen-base control
across fresh seeds. That negative result is retained in the experiment
README and is part of the rationale for the sufficient-statistics backend.
