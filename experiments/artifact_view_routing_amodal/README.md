# Outcome-only executable-view routing

This pressure test removes the caller’s view-selection shortcut from routed
artifact compaction. A frozen parent acquires two independent growth artifacts
and stores them in one verified row with two opaque aliases/views. A separate
`FactorizedOpaqueAddressRouter` then learns which view to activate using only
controller-produced query tensors, opaque candidate keys, attempted view IDs,
and scalar verifier outcomes.

The router never receives a span label, task ID, correct unattempted choice,
or semantic meaning for a key coordinate. The test includes paired
counterfactual credit, candidate permutation, reward-shuffled, wrong-view,
persistent reload, and checksum-corruption controls.

This qualifies learned routing of two already-acquired procedures. It does not
yet establish route discovery for arbitrary tasks, unbounded executable
program growth, or general continual learning.
