# Generated composition pipeline medium rejection (2026-08-06)

Status: rejected; no weights or artifact promoted.

The two-program serial `ExternalCapabilityPipeline` was trained for 256
updates on the verifier-private six-composition workload. It reached only
`0.5742` held-out behavior and never reached a stable `0.75` prefix. The
parent was stable and unchanged; replay was zero.

Simple serial stacking is therefore insufficient: the pipeline has no learned
external binding for which primitive occupies each composition position. The
next repair is an explicit external composition router over independently
stateful program slots.
