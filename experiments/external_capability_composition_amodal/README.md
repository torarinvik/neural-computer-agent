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

The corrected harness also exposes the generic stable-prefix candidate
selector. On the corrected reports it selects the composed candidate for seed
69316 (`2,048` bits) and the fresh candidate for seed 69317 (`6,144` bits),
which is the intended safe fallback while inherited transfer remains
seed-sensitive.

`train_intermediate_consumer.py` is the stricter follow-up. It trains a
downstream consumer while downstream events are hidden, so the consumer must
use the prior opaque intention. On seed 69316 it reaches `0.8633` versus
`0.5586` for a blank pipeline and both program ablations are causal, but the
fresh head-only learner reaches mastery in `6,144` bits versus `16,384` for
the inherited consumer. The result is retained as an intermediate-only
capability diagnostic, not positive transfer.

The consumer audit accepts `--pipeline-warmup-updates` as a replay-free
training control. During warm-up, only the output decoder learns; the consumer
is evaluated without a gradient and is enabled for the remaining updates.
This tests whether separating output calibration from new memory-side
computation improves inherited sample efficiency without changing the frozen
controller or opaque pipeline contract. A nonzero warm-up is an experiment,
not an architectural requirement.
