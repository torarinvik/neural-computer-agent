# External capability composition

`train.py` pressure-tests the controller-as-CPU / memory-as-files boundary.
It acquires `complement4` and `reverse4` as separate external recurrent
programs, freezes them, serially composes them through
`ExternalCapabilityPipeline`, and trains only a fresh decoder on the novel
`complement_reverse4` target.

The harness includes blank-pipeline, fully fresh trainable-pipeline,
reward-shuffled, zeroed-program, exact reload, corruption, frozen-core, and
zero-replay controls. It is intentionally stricter than a side-by-side bank
test: the result must show that the programs are useful in a new composition.

The 2026-08-06 replicated run is retained as a rejected general-composition
diagnostic in
`session_records/sequence_working_memory_2026-08-02/external_capability_composition_rejected_v1_2026-08-06/`.
The pipeline beat the blank control on both seeds, but the first primitive was
not causal on one seed. The first audit's fresh-pipeline arm was invalidated by
a `no_grad` scope that blocked its program gradients, so its transfer result is
not used. The result therefore does not establish arbitrary program induction
or positive transfer against a fresh learner. `audit_event_visibility.py`
rehydrates a persisted pipeline and removes raw events from downstream
programs; the seed-69317 result drops from `0.8828` to `0.5195`, exposing the
current shortcut rather than hiding it.
