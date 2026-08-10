# Verifier-selected positive transfer — 2026-08-10

This audit is the complementary case to the negative-transfer challenger. A
mastered adaptive seven-stage evidence sequence is given an unseen mask and
an unseen target that is a nearby continuation of the mastered successor.
The same isolated outcome-only challenger compares a copied protected cell
with fresh state; promotion requires transfer to win in both the warm and
matched-fresh branches across all seeds.

Transfer wins every branch with a material probe margin. It is selected in all
six decisions, and every selected branch passes novel mastery, held-out
retention, source immutability, exact persistence, frozen-core, missing-
evidence, corruption, shuffled-outcome, and post-reversal controls. The
positive result is about the verifier's branch decision: matched warm
acquisition is faster in two of three seeds, so this does not yet establish a
universal warm-over-fresh speedup.

| seed | warm transfer/fresh probe | fresh transfer/fresh probe | warm/fresh continuation | warm/fresh score |
| ---: | ---: | ---: | ---: | ---: |
| 85301 | 0.9967 / 0.5736 | 0.9337 / 0.5736 | 1 / 8 | 0.9967 / 0.9815 |
| 85302 | 0.9241 / 0.6033 | 0.9925 / 0.6033 | 13 / 1 | 0.9881 / 0.9884 |
| 85303 | 0.9871 / 0.7926 | 0.8870 / 0.7926 | 1 / 15 | 0.9807 / 0.9774 |

Reproduce from the repository root:

```bash
.venv/bin/python experiments/policy_free_intention_routing/positive_transfer_challenger.py \
  --seed 85301 \
  --report-out /tmp/policy-free-intention-positive-transfer.json
```

This promotes bounded verifier-selected positive transfer for one nearby
unseen evidence combination and target. Broad positive transfer, arbitrary
new computation, unrestricted growth, compression, and general continual
learning remain unqualified.
