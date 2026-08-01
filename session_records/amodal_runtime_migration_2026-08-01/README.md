# Amodal runtime migration: intention-only checkpoint

## Question

Can the current five-capability controller stop emitting a learned residual in
the two-action decoder's coordinate system without losing any behavior?

## Method

The parent action adapter emits a two-dimensional residual `r`, while the
external decoder has weight matrix `W` with shape `2 × 24`. The migration
computes the minimum-norm right inverse:

```text
P = inverse(W W^T) W
```

and folds `r P` into the action adapter's final layer. The migrated adapter now
emits a 24-dimensional intention residual. Applying the unchanged decoder gives
`(r P) W^T ≈ r`. This is a checkpoint transformation, not training: it consumes
no sensory examples, labels, outcomes, verifier bits, or optimizer updates.

## Gradual evidence ladder

1. Synthetic controllers: under `1e-6` maximum logit drift and no decision
   changes.
2. 64-lifetime smell test: parent and candidate reports exactly identical. Both
   shared the same expected small-sample four-rule miss.
3. 512-lifetime test: parent and candidate both passed all five gates and their
   complete result objects were identical.
4. Four paired 512-lifetime rollouts: zero action flips over 12,288 decisions;
   maximum absolute logit difference `5.7220458984375e-6`.
5. Full 4,096-lifetime audit: all five gates passed.

## Adversarial and causal controls

The full audit includes reversal, shuffled feedback, active-state reset,
missing visual evidence, persistent-memory removal/corruption, prediction-flip,
and complete working-memory reset controls. Binary mapping, four-rule mapping,
cross-appearance relation, persistent memory, and span-two working memory all
passed from one immutable checkpoint.

## Result and boundary

The promoted checkpoint's two-coordinate compatibility suffix is structurally
zero. The learned residual now lives in the base intention space, and the
external decoder alone translates it to actions. This clears the active
protocol-specific migration debt, but does not yet prove multiple decoders or
variable-cardinality input streams.

Artifacts:

- `repertoire_amodal_intention_audit_seed124005.json`
- `artifacts/checkpoints/unified_repertoire_span2_amodal_intention_seed122005.pt`
- SHA-256 `9eea7ab479cb8450737f040b76495cc5ec737e970cdc165af2446873e530cd6c`
