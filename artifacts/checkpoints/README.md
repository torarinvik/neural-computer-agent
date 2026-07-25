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
