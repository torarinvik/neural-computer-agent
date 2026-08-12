# Non-commuting recursive external composition — promoted bounded result

This audit strengthens the recursive CPU/files pressure test with a dependent
four-file chain over mixed domains `(2, 4, 8)`:

- `INC(slot0, m=2)`;
- `CINC(slot1 | slot0, m=4)`;
- `CINC(slot2 | slot1, m=8)`;
- `CDEC(slot0 | slot2, m=2)`.

Each later file reads a value changed by an earlier file. Reordering the
depth-four chain is therefore behaviorally meaningful rather than merely a
different serialization of commuting operations. The external memory and
generic interpreter remain frozen during acquisition; only aggregate scalar
verifier outcomes reach the optional external proposal policy.

Across four seeds, depth-two, depth-three, and depth-four files reached
`1.0000` held-out accuracy, every protected atomic file retained `1.0000`, and
recursive provenance, exact reload, missing-evidence no-op, shuffled-feedback
rejection, and zero-replay/controller-update gates passed. The wrong depth-four
order scored `0.0625` on every seed, giving a substantially stronger causal
order control than the earlier chain.

The orientation-invariant structural policy is retained as an architectural
option, but its warm/fresh proposal ratios were `0.8750`, `0.8421`, `3.0000`,
and `1.1111`. This is not promoted as a reliable sample-efficiency gain. The
promoted claim is bounded replay-free recursive external computation with
non-commuting order dependence, not arbitrary program induction or general
continual learning.

Run with:

```text
PYTHONPATH=. uv run python experiments/recipe_expressibility/verified_recursive_composition_growth.py --report-out report.json --seeds 17 18 19 20 --variant noncommuting_chain --policy-profile orientation_invariant
```
