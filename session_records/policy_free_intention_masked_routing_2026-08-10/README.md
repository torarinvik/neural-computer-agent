# Explicit masked-context external routing — 2026-08-10

This is a promotion-quality pressure test for the new external-learning mask
contract. The controller and state adapter are frozen. External intention
memory receives explicit observed-value and observation-mask channels; routing
keys receive the same opaque value/mask representation. Source, successor, and
reversal regimes use complementary partial views, with delayed feedback,
noisy reversal, copy-on-write growth, negative-transfer rollback, shuffled
outcome/action controls, missing-evidence no-op, corruption, persistence, and
zero-replay accounting.

Both seeds preserve the frozen controller, explicit mask features, protected
cells, exact persistence, and all causal controls. The frontier is rejected:
only one seed beats the matched-fresh successor on update count, so arbitrary
mask-distribution transfer is not promoted. The result qualifies the mask ABI
as an information-preservation mechanism, not as general continual learning.

The next experiment should use overlapping mask curricula and then introduce
mask changes gradually. A later promotion must require every seed to pass
stable-prefix retention and positive transfer under the same accounting.

Reproduce one seed from the repository root:

```bash
.venv/bin/python -m experiments.policy_free_intention_routing.train \
  --seed 85301 \
  --masked-context \
  --report-out /tmp/policy-free-intention-masked-routing.json
```
