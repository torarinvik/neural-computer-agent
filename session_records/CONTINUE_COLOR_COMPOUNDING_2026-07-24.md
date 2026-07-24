# Continuation record — color-primitive compounding — 2026-07-24

## Resume point

- Repository: `neural-computer-agent`
- Branch: `main`
- Capability commit before this continuation note:
  `9b89dc3c190e3cbbe66a78fa893b23c21780271c`
- Curated checkpoint:
  `artifacts/checkpoints/color_primitive_compounder_bits16_seed1901.pt`
- Checkpoint SHA-256:
  `698cf1a56914ff733e808c189043caea66e0355b1da1742bbcd78fd9be2f156f`

This is the first replicated cross-attribute compounding result. The model
acquired target color from 64 verified reward bits and effect color from 24.
With both continuous primitives available, a new same/different relation
reached stable causal mastery from 16 new reward bits on both the selected
stream and a fully disjoint blind stream. Neither-acquired and single-atom
controls failed through 64 bits, establishing a measured transfer-ratio lower
bound of 4x.

On the blind stream, normal accuracy, two true pixel-rerender
counterfactual accuracies, and both prediction-flip rates were 100%. Removing
either causal ingredient returned behavior to chance; exact complement
controls scored 0%; stratified shuffled-outcome controls had a 55.1% median
and no causal pass. The inherited core stayed bit-identical, and its earlier
position curriculum retained its 8/8/24-bit learning curve.

## Claim boundary

This is strong evidence of reward-only acquisition and causal reuse of two
continuous visual primitives inside a shared event structure. It is not yet a
general amodal concept space. The objects use deliberately salient fixed
colors, the binary answer interface and event structure are shared, and the
checkpoint retains two vision branches because the inherited position branch
helped effect-color learning but suppressed target-color learning.

No semantic labels or unattempted-action labels were used to train the
accepted capability. Verifier-private facts were used only for evaluation,
counterfactual generation, and audit controls.

## Evidence to read first

1. `experiments/forward_transfer_attention/SAMPLE_EFFICIENCY_LEDGER.md`
2. `experiments/forward_transfer_attention/ROBUST_SAMPLE_EFFICIENCY_STRATEGY.md`
3. `experiments/forward_transfer_attention/reports/core263_color_primitive_compounding_selected_parent.json`
4. `experiments/forward_transfer_attention/reports/core263_color_primitive_compounding_blind_replication.json`
5. `experiments/forward_transfer_attention/reports/core263_color_primitive_initialization_race.json`
6. `artifacts/checkpoints/README.md`

## Restore and verify

The sibling file `neural-computer-agent-latest.bundle` is a complete Git
bundle. Restore it from its parent directory with:

```bash
git clone neural-computer-agent-latest.bundle neural-computer-agent
cd neural-computer-agent
```

Set up and verify:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
./scripts/verify_curated_artifacts.sh
.venv/bin/python -m pytest -q
```

Reproduce the selected-parent audit:

```bash
.venv/bin/python -m experiments.forward_transfer_attention.audit_color_primitive_compounding \
  --checkpoint artifacts/candidates/core263_parent.pt \
  --report /tmp/color_compounding_selected.json \
  --checkpoint-out /tmp/color_compounder.pt \
  --target-bits 64 \
  --effect-bits 24 \
  --relation-bits 64 \
  --test-lifetimes 256 \
  --atom-updates 128 \
  --relation-updates 128 \
  --batch-size 32 \
  --seed 1901 \
  --target-initialization-seed 2741 \
  --effect-initialization-seed 1943 \
  --relation-initialization-seed 2805
```

For the blind replay, use the same command with seed `2003`, data offset
`2000000`, and a different report path. The committed reports are the
authoritative results; do not overwrite them casually.

## Next highest-ROI ladder

Keep the sub-minute -> 3-minute -> 10-minute escalation rule. Advance only
when blind causal evidence improves.

1. Shrink the salient color cues gradually while keeping event structure
   fixed. Measure the smallest cue size that preserves the 16-bit curve.
2. Vary color pairs and palettes with logical-lifetime-disjoint splits. Require
   true palette counterfactuals, missing-cause controls, shuffled outcomes, and
   retention before promotion.
3. Test whether the two vision branches can be distilled or compressed
   without worsening the stable bits-to-gate. Keep the dual-branch checkpoint
   as the immutable control.
4. Only after those pass, bridge color to a new attribute such as shape. This
   is the next serious test of whether the learned primitives are becoming
   modality/appearance-independent rather than remaining color-specific.

Every race must share environmental experience across clones, count search
compute separately, select on one stream, and confirm the exact parent on a
blind stream. Catastrophic-forgetting gates and adversarial causal audits
remain mandatory.
