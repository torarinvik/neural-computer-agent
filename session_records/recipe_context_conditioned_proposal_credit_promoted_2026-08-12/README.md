# Context-conditioned recipe proposal credit — promoted narrow result

This audit tests a memory-side proposal policy for the generic recipe
sequence search. The policy receives opaque context and candidate identities
plus aggregate scalar verifier quality. It does not receive task labels or
verifier rows, and it does not update the controller.

Across seeds `17` and `18`, two contradictory order-sensitive recipes were
acquired in separate contexts, the policy was persisted and reloaded, and both
recipes were reacquired in new lifetimes without replay. Held-out accuracy was
`1.0000` for every acquired and reacquired target. Warm proposal counts were
lower than fresh controls in all four comparisons. An unseen context stayed
unbiased, the exploration floor remained active, each trained context favored
its own candidate, and shuffled feedback was rejected.

Run with:

```text
PYTHONPATH=. uv run python experiments/recipe_expressibility/context_conditioned_proposal_credit.py --report-out report.json
```

The archived summary is metadata-only; raw verifier rows are not retained.

Claim boundary: bounded replay-free contextual reuse of exact whole-candidate
proposal credit. This is not general continual learning, factorized
instruction/position transfer, arbitrary program induction, or unrestricted
memory growth.
