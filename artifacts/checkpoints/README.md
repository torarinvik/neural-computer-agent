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
