# Fixed-capacity reliability transfer — pre-registration

## Question

After sequentially consolidating capacity-four and capacity-five experience,
does that lineage learn a changed retrieval-reliability regime faster than its
capacity-four ancestor and a matched reset learner?

Capacity remains five. The only target change is write/retrieval threshold
`0.5 → 0.6`, which alters occupancy and stored-item reliability statistics.
The learner is not told the threshold or capacity. The controller, sensory
stream, four generic evidence values, attempted-action objective, optimizer,
read cost, and model size remain unchanged.

## Lineage and accounting

1. Reconstruct the passing seed-7842 capacity-five phase from the curated
   capacity-four checkpoint.
2. Save the resulting capacity-five checkpoint with its 720 source bits and
   12 source updates attached to provenance.
3. On the threshold-0.6 target, race:
   - consolidated capacity-five lineage;
   - capacity-four ancestor;
   - matched reset;
   - reward-shuffled, feature-shuffled, and missing-evidence controls.

The target race uses 720 fresh learner-visible outcomes, 12 updates, zero
replay, and 2,040 held-out contexts/4,080 private counterfactual audit bits.

## Gates

The consolidated lineage must stably reach:

- at least `65%` compute-choice accuracy;
- at least `+0.05` utility over the strongest fixed action;
- at least `20%` oracle-gap capture;
- all three thresholds earlier than both ancestor and reset.

Every causal control must cost at least `0.02` utility. Gradients,
serialization, binary mapping, and four-rule retention must pass. Seed 7851 is
the discovery run; only a complete pass permits unchanged seed-7852
replication. Failure earns no additional training budget.

Transfer distance increases scientific value only after these verified gates
pass; novelty itself produces no reward.

## Threshold 0.6 result and frontier diagnostic

The consolidated lineage was stronger at threshold 0.6 (`83.4%` choice
accuracy versus `80.8%` for its ancestor), but both were already stably
mastered at zero new bits. It therefore failed the pre-registered
bits-to-threshold advantage and is not replicated.

This rung was too easy rather than too hard. A zero-training private diagnostic
now measures thresholds `0.60, 0.65, 0.70, 0.75, 0.80` on 2,040 contexts each.
It selects the smallest measured threshold where the capacity-five lineage
passes the fixed mastery gates and the capacity-four ancestor does not. The
probe uses no learner-visible outcomes or optimizer updates; its 20,400
private counterfactual verifier bits are reported. The selected threshold must
still pass a fresh pre-registered training race and replication.
