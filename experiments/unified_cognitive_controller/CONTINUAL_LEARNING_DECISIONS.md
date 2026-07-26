# Continual-learning decisions

This project optimizes verified reusable capability per interaction, not
benchmark accuracy alone. Methods must remain compatible with one controller,
unknown task boundaries, sensory-only inputs, opaque attempted actions, scalar
outcomes, and no hand-authored semantic targets.

## Adopt now

### Experience replay as the baseline

Replay is the strongest broadly reliable continual-learning baseline in the
survey literature, and it already solved a measured forgetting failure here.
Our deterministic generators can recreate prior sensory experiences, so old
raw examples do not need to be stored.

The replay ratio is an experimental variable. In the persistent-memory atom,
changing from one new-memory update plus two rehearsal updates to two
new-memory updates plus two rehearsal updates improved recall at the same total
update budget while preserving both old gates.

### Fixed-budget accounting

Every result records or should record:

- unique logical contexts and verifier outcomes;
- replayed experiences and optimizer updates;
- model parameters, active-memory rows, and disk bytes;
- wall time and device;
- forward transfer, backward retention, and causal controls.

Search compute and matched-fresh controls stay separate from final-agent
experience.

### Future-shifted and task-free evaluation

Audits use disjoint seeds and renderer surfaces. The learner receives no task
identity. Capacity and appearance are varied after training to measure genuine
forward transfer rather than iid memorization.

## Test later, only against replay

### Dark/self-distillation replay

DER-style replay is philosophically compatible when targets are the
controller's own earlier logits or latents. It may reduce the number of full
old-environment rollouts. It earns implementation only if matched-budget
experiments beat plain generated replay on retention plus forward transfer.

### Interference-aware replay

MIR-style selection may help when the primitive library becomes large. The
verifier can select old sensory lifetimes with the greatest measured
interference without revealing task IDs to the controller. It is unnecessary
while cheap balanced replay preserves all gates.

### Lightweight functional or parameter regularization

Output distillation, EWC, SI, or gradient projection are secondary controls,
not defaults. They add pressure toward stability but may reduce the plasticity
and phase-transition learning this project prioritizes. Test them only when
replay cost becomes a demonstrated bottleneck.

## Reject for the current architecture

- task-specific heads, masks, prompts, or test-time task IDs;
- a new network column or expert for each primitive;
- uncounted pretrained backbones or hidden external data;
- model growth reported without total parameter and memory cost;
- supervised labels, rule IDs, correct unattempted actions, or semantic
  solution traces;
- average accuracy without retention, transfer, and resource accounting.

## Next continual-learning question

The controller now learns both sparse write/skip and adaptive read/no-read
decisions, plus bounded replacement at capacity 4, and passes replicated
physical disk, corruption, and retention gates. Zero-shot capacity 8 was
behaviorally close but failed causal eviction gates. The next high-value rung
is therefore capacity 5 with capacity-4 rehearsal, changing only one
difficulty axis. After that replicates, replace the deterministic
oldest-is-obsolete distribution with a noisy frequency/recency utility stream.
Consolidation and merging follow only after replacement survives both changes.
