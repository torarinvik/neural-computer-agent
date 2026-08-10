# Context-isolated replay-free reliability

This three-seed audit tests whether learned reliability state can remain local
to an opaque factual slot. Four slots receive identical near-tolerance drift
errors, but verifier outcomes alternate negative and positive by slot.

The contextual sufficient-statistics gate vetoes only the negative slots and
routes the positive slots back to their original stable IDs. A matched global
error-bin gate is included as a control and over-vetoes the positive slot,
showing the failure caused by pooling evidence across regimes. The fact bank
is unchanged by routing, contextual state persists exactly, and the controller,
base, and context encoder remain frozen with zero replay.

This promotes context-isolated reliability as a bounded continual-memory
primitive. It does not establish learned raw-modality context formation,
unrestricted memory growth, or general continual learning.
