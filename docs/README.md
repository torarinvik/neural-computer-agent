# Project documentation map

## Normative documents

1. [`AMODAL_N_TO_M_ARCHITECTURE.md`](AMODAL_N_TO_M_ARCHITECTURE.md) defines the
   target system, module boundaries, terminology, current implementation gap,
   migration order, and audit requirements. It is the source of truth for all
   architecture questions.
2. [`../AGENTS.md`](../AGENTS.md) defines repository operating and scientific
   rules for contributors and automated agents.
3. [`../README.md`](../README.md) summarizes the project, current audited
   frontier, and links to evidence.
4. [`FAST_ITERATION.md`](FAST_ITERATION.md) defines campaign-sizing practice.
5. [`PROMOTION_FIREWALL.md`](PROMOTION_FIREWALL.md) defines the machine-checkable
   evidence bundle and one-use holdout guard for promotion claims.

The canonical implementation is in `../src/neural_computer/`. Experiment
directories may retain historical trainers and compatibility readers, but new
runtime code must import the production package rather than defining agent
interfaces inside `experiments/`.

## Supporting design documents

- [`../experiments/forward_transfer_attention/AMODAL_CONCEPT_SPACE_DESIGN.md`](../experiments/forward_transfer_attention/AMODAL_CONCEPT_SPACE_DESIGN.md)
  explains the representational motivation for learned modality-independent
  concepts and intentions.
- [`../experiments/archive/unified_cognitive_controller/SCRIPT_ALIGNMENT_LADDER.md`](../experiments/archive/unified_cognitive_controller/SCRIPT_ALIGNMENT_LADDER.md)
  turns the target architecture into auditable migration gates.
- [`../experiments/archive/unified_cognitive_controller/CONTINUAL_LEARNING_DECISIONS.md`](../experiments/archive/unified_cognitive_controller/CONTINUAL_LEARNING_DECISIONS.md)
  records continual-learning methods compatible with the target.

## Historical evidence

Experiment READMEs, result reports, handoffs, and session records describe the
exact systems tested at the time. They are evidence, not alternative normative
architectures. Do not silently rewrite historical results to match the current
goal; add a scope notice or link to the canonical specification instead.

When wording conflicts, the canonical architecture document controls the
target, while the historical file controls what that particular experiment
actually implemented and demonstrated.
