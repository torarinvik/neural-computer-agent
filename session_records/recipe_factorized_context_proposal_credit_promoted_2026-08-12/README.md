# Factorized context-conditioned proposal credit — promoted narrow result

This audit tests whether an external memory can transfer scalar proposal
credit over a reusable instruction and insertion position to a different
parent program. The controller remains frozen. The learner sees only opaque
recipe candidates, parent/context keys, generic factor descriptors, and
deterministic scalar verifier outcomes.

Across four seeds, the source parent changed from `INC(0,m=2)` to
`DEC(0,m=2)`, so the target whole-program digest changed, while the opaque
`CINC(1|0,m=8)` insertion at position one was unchanged. Factorized warm
transfer reached the target in one proposal on every seed, versus `10--23`
for fresh controls and `10--16` for the prior whole-candidate policy. All
held-out targets reached `1.0000`. A contradictory context learned `CDEC`
locally without changing the original context's `CINC` preference.

The audit also passed protected-file retention, policy and file checksum
reload, exploration-floor, shuffled-feedback rejection, zero-replay, and
zero-controller-update controls. Raw verifier rows are not retained.

Run with:

```text
PYTHONPATH=. uv run python experiments/recipe_expressibility/factorized_context_proposal_credit.py --report-out report.json --seeds 17 18 19 20
```

Claim boundary: bounded factorized instruction/position transfer and local
reversal routing. This does not establish reusable multi-step composition,
unrestricted memory growth, or general continual learning.
