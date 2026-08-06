# Generated composition medium rejection (2026-08-06)

Status: rejected; no weights or artifact promoted.

The monolithic `ExternalCapabilityProgram` was trained for 256 updates on
the verifier-private workload of six sampled two-primitive compositions. The
parent reached a stable prefix and remained frozen, but the new capability
reached only `0.5508` held-out behavior and never reached a stable `0.75`
prefix. Replay was zero.

This is a mechanism boundary, not an under-budget pilot: individual fixed
procedures already pass at this budget, while a distribution of compositions
does not. More optimizer tuning is therefore not the next move. The next
implementation is an explicitly compositional external stack with
independently learned primitive programs and a learned composition route.
