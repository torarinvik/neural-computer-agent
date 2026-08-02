# Brain Workshop 4-back retention-aware continuation

This session tested the next protected-plasticity frontier after the verified
1-back → 4-back compounding rung. The controller was initialized from the same
inherited `nback3_rehearsal1_2_depth3` parent family used by the earlier ladder,
then trained on 4-back with the unchanged three-stream amodal bus:

```text
--modalities vision,audio,text
--target-modalities text
--factorized-output --factorized-reward
--external-history --external-history-depth 4
--per-stream-external-history
--external-memory-adapter-width 64
--per-stream-intention-adapter-width 64
```

The new implementation adds `--rehearsal-weights`, a verifier-side vector
matching `--rehearsal-n-backs`. The tested policy used weights **2, 0.5, 0.5**
for 1-, 2-, and 3-back. This is not a semantic label exposed to the controller;
it only changes how much private rehearsal loss is mixed into the optimizer.

## Result

| seed | 4-back at 256 updates | 4-back after continuation | 1-back retention | reset control | time-shuffle |
|---|---:|---:|---:|---:|---:|
| 47408 | **80.42%** | — | **94.29%** (parent 93.96%) | 49.12% | 51.56% |
| 47409 | 66.99% | **77.00%** at 320 updates | **94.61%** (parent 94.32%) | 49.07% | 51.78% |

All percentages above are eligible exact accuracy after the n-back warm-up,
unless noted otherwise. The checkpoint retention audits use a frozen
controller and a one-update no-op continuation; history reset returns to
chance. Both seeds therefore pass the 2-point 1-back retention gate and the
causal controls. The continuation is the important sample-efficiency result:
we spent another 64 updates only after a valid 256-update run had crossed the
65% acquisition gate but remained below the desired mastery band.

The fixed weighting is not universally better than uniform rehearsal: seed
47408 improved over its uniform-weight comparison, while seed47409 initially
lagged it and needed continuation. The promoted strategy is consequently
**gated continuation with retention audits**, not “always use these weights.”
Uniform rehearsal remains the robust baseline; per-rung weights are an
experimentally supported control for protecting a mastered rung when the
learning curve justifies it.

## Artifacts

- `nback4_rehearsal123_w2_05_05_seed47408_256_inherited.json` and
  `nback1_retention_targeted_inherited_after_nback4_seed47408_256.json`.
- `nback4_rehearsal123_w2_05_05_seed47409_256_inherited.json`,
  `nback4_rehearsal123_w2_05_05_seed47409_64_continuation.json`, and
  `nback1_retention_targeted_inherited_after_nback4_seed47409_320.json`.
- Checkpoints with matching names under `artifacts/checkpoints/`.

Several earlier files in this directory are deliberately retained as negative
controls. Some were fresh rather than inherited, and others omitted the
factorized output or 64-wide RAM adapters; they must not be compared with the
compounding runs. This provenance distinction was caught before promotion.

## Next frontier

The system now has a verified, causal **learn → check → continue** loop for a
harder cognitive primitive while retaining the mastered one. The next step is
to make the continuation decision itself generic and verifier-driven: estimate
bits-to-threshold from the held-out score, continue only when progress is
positive, and reject runs whose retention margin falls below the gate. A third
seed and a matched uniform-rehearsal continuation are the next replication
controls before claiming a universally improved rehearsal policy.
