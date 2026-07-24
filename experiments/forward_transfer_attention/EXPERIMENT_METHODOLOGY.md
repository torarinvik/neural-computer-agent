# Probe-gated capability development

## Current frontier

The first complete temporal loop is solved for the current sensory controller.
The demonstrated rule is decodable from raw memory, a factorized learned
readout composes it with query-event facts, and the integrated agent passes
fresh counterfactual, memory-corruption, and old-skill retention audits. This
milestone is supervised-bootstrapped, not reward-only. The active frontier is
reusable event-indexed memory and measured transfer to a new composition task.
A generic content-addressed reader now passes replicated, lifetime-disjoint
diagnostics and transfers across two visual surfaces. In the full four-color
composition task it raises frozen behavior from roughly 20% to 34--38%, but
few-shot temporal-rule use remains weak. Full compositional transfer is
therefore not yet solved.

## Permanent decision rules

1. Probe both sides of an interface before changing either side.
2. Counterfactual re-rendering, fresh-seed replication, memory corruption, and
   retention are mandatory promotion gates.
3. A negative below the measured ignition scale is bounded and cannot support
   an architecture verdict.
4. Use cached frozen-controller representations for readout/composition work.
   Fresh rollout is reserved for cache construction and final audits.
5. Use geometric scale rungs. Start at the measured threshold, double only
   while loss/train-fit indicators remain alive, and stop after two consecutive
   scale increases without movement.
6. Compare final accuracy with ingredient arithmetic. When final performance
   matches the product of ingredient accuracies, improve the ingredients rather
   than the composer.
7. A run is decision-complete only when it saves provenance, per-example
   predictions, counterfactual and corruption controls, ingredient metrics, and
   per-primitive retention.
8. Supervised bootstrap, reward-only discovery, and verifier-free improvement
   are distinct claims and must remain explicitly labeled.

## Event-indexed memory principle

Recurrent state is a useful processor but a leaky long-term carrier. Facts from
different moments should be archived as event-indexed sensory/recurrent
snapshots and bound across events. This is a first-class architectural
principle for future temporal, counting, tracing, search, and compositional
tasks—not a task-specific patch.

The implementation uses two distinct agent-owned stores: compact active memory
for current rules and an immutable sensory event archive for later
content-addressed retrieval. Raw event rows must not be reinserted into compact
memory merely to make them accessible; doing so changes the consolidation
distribution and damages prior skills.

A reusable primitive should produce candidates or facts for a downstream
composer, not overwrite the final answer globally. Any new reader must pass
both its own causal controls and a no-op integration test before behavioral
training.

## Efficient experiment shape

Cheap, wide exploration happens against a large immutable cache. A
pre-registered sweep chooses at most one candidate. Only that candidate receives
fresh replay and the deliberately slow adversarial audit. Cloud GPU time is
reserved for cache extraction, training, and fresh sensory replay; cached
linear/MLP analysis may run locally.

The north-star metric remains verified reusable capability gained per unit of
experience and compute. The temporal loop now passes its behavioral and
retention gates. Novelty weighting, learning-progress curriculum, and
self-generated value signals remain deferred until the next composition task
measures whether the acquired primitive actually reduces future learning cost.
The immediate next measurement is compositional support-rule decodability at
the raw-write boundary; architecture work resumes only if that probe identifies
where the rule disappears.
