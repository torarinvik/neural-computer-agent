# Shared compute-value prerequisite — pre-registration

## Question

Can one small latent value trunk learn two physically different optional
operations—external-memory read and one recurrent thought—without losing the
source operation, and can inherited read experience reduce stable
bits-to-threshold on thought versus an identical reset trunk?

This is a prerequisite test, not yet the held-out third-operation claim.

## Architecture and information boundary

The model has one 4→16 generic trunk and two one-neuron operation adapters.
The operation adapter identifies the available physical interface, not the
semantic cognitive task. The learner sees only generic controller statistics,
its uniformly attempted action, exact propensity `0.5`, scalar verified
outcome, and compute cost. It never sees the unattempted result, correct
compute action, help/harm label, correct answer, or semantic task identity.

The inherited arm begins with the replicated read allocator's trunk and read
adapter; its thought adapter is zero. The matched-reset arm begins with the
same architecture fully reset. Both receive exactly the same fresh read and
thought experience. Reward-shuffled and missing-evidence controls share the
inherited initialization.

## Sub-minute budget and gates

- 12 joint updates;
- 60 fresh read and 60 privately balanced thought lifetimes per update;
- 720 learner-visible bits per operation, 1,440 total, zero replay;
- metrics every 120 bits per operation;
- all private screening and counterfactual audit bits reported separately.

The inherited shared model must attain on thought:

1. at least `60%` choice accuracy;
2. at least `+0.08` utility over the strongest fixed action;
3. at least `15%` oracle-gap capture;
4. a stable crossing strictly earlier than matched reset.

It must also retain at least `+0.05` read utility over the fixed policy, lose at
least `0.05` thought utility under both causal controls, preserve the mastered
controller tasks, keep live gradients, and reload exactly. A pass permits one
unchanged replication. A failure receives no larger budget.
