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

## Seed 7901 result and source-use diagnostic

The architecture failed: cost-aware, cost-shuffled, ancestor, and reset arms
all first reached stable target mastery at 600 bits. The established trunk
baseline is 120 bits, so no replication or scaling is allowed.

The cost-aware and cost-shuffled source arms were behaviorally identical.
Before rejecting explicit cost itself, the unchanged seed is rerun once with
diagnostic-only telemetry: L2 norm of the new cost-input column and the mean
change in read-advantage prediction between normalized cost 0 and 1. This
determines whether the source phase learned any cost dependence. It cannot
promote the failed configuration.

The diagnostic confirmed weak but real cost dependence:

- correct-cost source: cost-column L2 `0.0808`, prediction change `0.0140`
  across normalized cost 0→1;
- cost-shuffled source: L2 `0.0718`, prediction change `0.0099`.

Thus the new input was not dead, but 720 source outcomes produced only a small
cost effect and no target acceleration. Scaling is not justified because all
intact target learners—including reset—crossed only at 600 bits, far worse
than the established 120-bit inherited-trunk baseline.

The explicit-cost source branch is closed. The next higher-ROI curriculum is
operation-aligned: consolidate the successful 120-bit re-query trunk, then
introduce progressively harder re-query regimes before attempting another
physical-operation bridge.
