# Curated checkpoints

## Color primitive compounder

`color_primitive_compounder_bits16_seed1901.pt` is the first replicated
cross-attribute compounding checkpoint.

- SHA-256:
  `698cf1a56914ff733e808c189043caea66e0355b1da1742bbcd78fd9be2f156f`
- Size: approximately 5.5 MB
- Learner-visible training signals: rendered RGB events, attempted opaque
  binary answers, and scalar verifier outcomes
- Atom acquisition: 64 target-color outcomes and 24 effect-color outcomes
- New relation acquisition: stable causal mastery from 16 outcomes
- Exact unacquired architecture control: no mastery through 64 outcomes
- Blind disjoint replication: 100% normal and counterfactual accuracy with
  100% causal flips
- Missing-target and missing-effect audits: chance
- Retained position ladder: 8/8/24 outcomes

The checkpoint contains two compact vision branches, their reward-trained
primitive heads, and the 16-outcome relation head. The two-branch design is an
honest response to measured negative transfer: position-trained vision helps
effect-color learning but suppresses target-color learning. Unifying or
compressing these branches without losing the transfer gain remains open.

## Unified controller lineage

`unified_compound_rehearsal_seed2505.pt` is the first one-controller
few-shot-and-retention milestone.

- SHA-256:
  `8e346fd2863f595fd0ee96b5f4a8353cae48e96e7739ad0febd11345a60f9099`
- Size: approximately 1.2 MB; 298,252 parameters
- Learner-visible signals: rendered RGB, attempted opaque action, scalar
  verified outcome, and the controller's own latent active state
- One-support hidden bijection: 99.98% blind normal and 99.95% reversed
- Retained two-support four-function task: 100% blind normal and reversed
- Counterfactual prediction flips: 99.93% and 100%
- Shuffled-feedback, active-state-reset, and blank-vision controls prevent
  reward-history, static-action, and renderer-shortcut explanations
- No semantic task IDs, correct-action labels, unattempted-action labels, or
  within-lifetime weight updates

The three preceding checkpoints preserve the exact gradual lineage:

- `unified_constant_action_v1_seed2501.pt`: active-state memory atom;
- `unified_visible_identity_from_constant_seed2501.pt`: visual-action atom;
- `unified_four_rule_support2_from_visual_seed2502.pt`: audited two-support
  memory × perception parent.

The final rung used balanced rehearsal with different evidence schedules,
preserving the broader parent while acquiring the harder one-support mapping.
Persistent disk memory exists only as an inactive tested interface and is not
part of this capability claim.

## Persistent-memory ladder

`unified_persistent_capacity16_bridge_seed4006.pt` is the current unified
controller milestone.

- SHA-256:
  `7ae568ea2007241dcc167764bd91ba4e4c19bb95431c968d42e0f4c6f766a215`
- Capacity-16 trained recall: 91.21%
- Zero-shot capacity-32 recall: 87.48%
- Capacity-32 retrieval top-1: 73.82%
- Capacity-64 frontier: 81.69%, honestly rejected
- Empty, shuffled, and corrupted memory controls: large causal degradation
- Private-rule reversal and paired prediction-flip controls: passed
- Original one-support and four-rule behavioral retention: passed
- Serializable disk read reproduces the hard latent-memory read

The exact gradual parents are:

- `unified_persistent_capacity2_ratio2_seed4004.pt`, SHA-256
  `2be66cd8b13f4eee86a80bd2c0369a7cb138d1b83f489cef5e962ab0d086a34c`;
- `unified_persistent_capacity8_bridge_seed4005.pt`, SHA-256
  `90d4a28c8c855190fc58f4f536a927ef3548229768a9bd5ca6117655fa24528c`.

The memory rows contain controller-created opaque vectors, not semantic
records. The current rung uses store-all admission; it does not yet claim
learned selective writes or consolidation.
