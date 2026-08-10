# Factored factual transition memory — promoted

This three-seed audit validates a frozen shared factual base plus an external
context-addressed residual store. Four genuinely disjoint transition regimes
arrive sequentially. Source regimes receive complete evidence; target regimes
receive only verifier-private target-covering rows. The planner derives
behavior from base-plus-residual predictions and receives learned opaque
context tensors, never regime labels.

All seeds mastered every planner goal, retained every earlier regime after each
append, kept the base and context encoder unchanged, used zero residual
optimizer updates, and restored the complete component exactly. The residual
store consumed 40 transition rows once per seed.

| seed | base updates | residual rows | all-regime mastery | exact persistence |
| ---: | ---: | ---: | ---: | ---: |
| 82701 | 35 | 40 | 1.000 | true |
| 82702 | 38 | 40 | 1.000 | true |
| 82703 | 37 | 40 | 1.000 | true |

Claim boundary: this promotes a bounded factored factual-memory boundary under
partial evidence. It does not establish automatic context formation, arbitrary
missingness, unbounded residual growth, compression, or general continual
learning. The context encoder is pretrained, the residual lookup is exact
match, and the target mask is verifier-selected so measured goals remain
solvable.

Reports are protected by SHA256SUMS.
