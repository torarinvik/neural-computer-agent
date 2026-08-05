# Canonical growth-register working-memory pressure test — 2026-08-04

This is the promoted canonical-controller follow-up to the archived
producer→consumer experiment. A rendered Brain Workshop-style sequence is
encoded into amodal events. The parent learns forward reproduction; then its
controller, frontend, and decoder are frozen while two generic growth slots
are trained only from sampled opaque actions and scalar verifier outcomes.

Slot zero is the acquired producer. Slot one is a recurrent prior-only
consumer: its input width is exactly the producer register width, so it has no
raw event or controller-state bypass. Both tensor artifacts are persisted and
reloaded through `ExecutableArtifactMemory`, then remapped into the two
canonical growth namespaces.

Primary seed `69204` reached `85.94%` composed accuracy versus `50.52%`
parent; the independent seed `69205` reached `98.18%` versus `42.71%` parent.
Producer-zeroed, prior-read-ablated, blank-sequence, reward-shuffled,
artifact-reload, and frozen-core controls passed in both runs. The consumer-
only replica scored `25%`, which is retained as a no-bypass result because it
does not exceed its parent.

This promotes only the narrow canonical producer-register → prior-only-
consumer mechanism. It does not establish arbitrary program induction,
long-span mastery, or general cognition.

The executable harness is
`experiments/working_memory_continuous/canonical_growth_pressure_test.py`.
