# Variable deliberation short rung

This record covers the first sub-minute rung for the single-controller
`WAIT`/`THINK`/`COMMIT` execution plane. Each episode exposes a high-bit event;
the low-bit partner is either present immediately or arrives one tick later.
The controller receives only opaque event tensors and the scalar verifier
outcome.

The rung was run for 256 optimizer updates at seeds 17, 18, and 19. It is
rejected as a learned capability promotion: all three policies stayed at
100% immediate `COMMIT`, and adaptive utility did not consistently beat fixed
waiting or fixed thinking. The result validates the runtime path and exposes
the next implementation bottleneck—execution-policy exploration and credit
assignment—not a learned compute-allocation gain.
