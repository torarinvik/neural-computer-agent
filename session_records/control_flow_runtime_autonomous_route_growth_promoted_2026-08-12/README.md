# Runtime-owned route credit — narrow promotion

This three-seed audit validates runtime-owned reachability of a newly admitted
external control-flow file. The runtime stores the previous opaque route query
and selected file in `ControlFlowRuntimeState` v2. The next cycle consumes an
explicit scalar route outcome, updates a copy of the external context-route
evidence, and selects the next file without a caller-supplied slot override.
Nonzero exploration returns the exact selected propensity.

The positive arm interleaves two contexts after admitting a second file, then
reverses one context without replaying the other. Seeds 17, 18, and 19 all
reached the new file, retained the original context binding, recovered the
reversal, preserved the source file and frozen controller, and passed state /
evidence reload and corruption controls. The shuffled-feedback null failed to
master the two bindings on all three seeds.

Accounting charged 400 verifier lifetimes per arm and seed: 2,400 total across
the three positive and three shuffled arms. Controller optimizer updates and
replayed examples were zero.

This promotes bounded external route reachability and delayed scalar credit,
not arbitrary program induction, unrestricted memory growth, or general
continual learning. Reproduce with:

`experiments/recipe_expressibility/control_flow_runtime_autonomous_route_growth.py`
