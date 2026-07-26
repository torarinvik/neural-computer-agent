# Explicit-cost shared-trunk re-query — pre-registration

The established trunk-transfer baseline learns second-ranked re-query in 120
target bits. This experiment must do strictly better.

The architecture preserves the learned four-statistic evidence pathway exactly,
adds normalized compute cost as a fifth generic input, and uses separate
one-neuron read and re-query adapters. It receives no task identity, correct
action, unattempted outcome, or correct answer.

Before re-query, the shared trunk receives 720 fresh attempted-read outcomes
across costs `0.01, 0.08, 0.16, 0.24`. A cost-shuffled source control sees the
same outcomes with mismatched cost inputs. The re-query adapter remains zero
during this phase.

The target phase uses 720 fresh re-query outcomes and compares:

- explicit-cost source-trained trunk;
- cost-shuffled source trunk;
- inherited ancestor without variable-cost source training;
- fully reset learner;
- reward-, feature-, and missing-evidence controls.

Pass requires stable target mastery before 120 bits and strictly earlier than
all three learning baselines, plus the usual choice, utility, oracle-gap,
causal evidence, gradient, persistence, and retention gates. Seed 7901 is the
sub-minute discovery run. Only a complete pass permits unchanged seed 7902
replication.
