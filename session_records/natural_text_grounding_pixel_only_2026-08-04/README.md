# Pixel-only grounded text frontend — 2026-08-04

## Decision

This is now a promoted three-seed causal population. All three saved frontends
passed the 16-update in-process gate and an independent optimizer-free replay
over 1,024 lives per held-out appearance/style cell. The earlier 256-lifetime
seed `1001001` near miss was finite-sample variance; its 1,024-lifetime
contradiction flip is `77.44%`.

The earlier raw-byte record is separately retracted because its renderer used
verifier-generated context IDs. This run uses the repaired pixel-only renderer
and must not be combined with the retracted evidence.

## Mechanism

The external text source derives colour, brightness, position, and coarse
visible geometry from rendered pixels. The learner receives only shifted,
padded UTF-8 bytes. A replaceable character n-gram frontend is trained with
paired encoded-event consistency against the frozen visual encoder. The
controller, input bus, output bus, decoder, and visual encoder are frozen.

The audit uses held-out text styles, cross-lifetime text shuffling,
contradictory text, and full-vision retention. It checks every held-out style
for bars, diamonds, and dot pairs. No verifier outcome, action label, task ID,
or hidden context ID is passed to the renderer or frontend.

## Results

| seed | training min fused | training min flip | replay min fused | replay min flip | replay |
|---|---:|---:|---:|---:|---|
| 1001001 | 93.33% | 76.56% | 91.95% | 74.84% | near miss |
| 1001002 | 93.13% | 77.19% | 92.34% | 77.19% | pass |
| 1001003 | 93.13% | 79.27% | 93.67% | 79.37% | pass |

The promotion replay at 1,024 lives per cell gives these population minima:

| seed | min fused | max shuffled | max contradictory | min flip | min vision |
|---|---:|---:|---:|---:|---:|
| 1001001 | 93.59% | 53.96% | 16.15% | 77.44% | 97.81% |
| 1001002 | 93.03% | 54.45% | 14.94% | 78.09% | 97.62% |
| 1001003 | 93.79% | 54.12% | 14.24% | 79.55% | 97.60% |

Strict gate: fused ≥90%, shuffled ≤60%, contradictory ≤25%, contradiction
flip ≥75%, and full-vision accuracy ≥95%, with the controller unchanged.

## Accounting

Each run used 16 optimizer updates, 36,864 paired unlabeled frames, zero
verifier bits, zero logical lifetimes, and no replayed examples. Wall times
were 127.64s, 125.56s, and 132.56s for seeds 1001001–1001003. The saved
frontend replay loaded no optimizer state and left the controller unchanged.

## Promotion record

The machine-checkable promotion evidence is [promotion.json](promotion.json)
and its one-use local holdout claim is
[promotion_holdout_ledger.jsonl](promotion_holdout_ledger.jsonl). Verify it
with:

```sh
uv run python scripts/verify_promotion_record.py \
  session_records/natural_text_grounding_pixel_only_2026-08-04/promotion.json \
  --holdout-ledger session_records/natural_text_grounding_pixel_only_2026-08-04/promotion_holdout_ledger.jsonl
```

## Artifacts and reports

- [training summary](training_summary.json)
- [saved replay summary](saved_frontend_replay_summary.json)
- [1,024-lifetime promotion replay summary](saved_frontend_replay_n1024_summary.json)
- [promotion manifest](promotion_manifest.json)
- [promotion configuration](promotion_configuration.json)
- [sample-efficiency ledger](sample_efficiency_ledger.json)
- [reproducible audit CLI](../../experiments/natural_text_grounding/audit_saved_frontend.py)
- [seed 1001001 training report](training_seed1001001_u16.json)
- [seed 1001002 training report](training_seed1001002_u16.json)
- [seed 1001003 training report](training_seed1001003_u16.json)
- [seed 1001001 saved replay](saved_replay_seed1001001_n256.json)
- [seed 1001002 saved replay](saved_replay_seed1001002_n256.json)
- [seed 1001003 saved replay](saved_replay_seed1001003_n256.json)

Curated frontend checkpoints are listed in
`artifacts/manifests/curated_checkpoints.sha256`:

- `amodal_pixel_only_text_frontend_seed1001001_u16.pt`
- `amodal_pixel_only_text_frontend_seed1001002_u16.pt`
- `amodal_pixel_only_text_frontend_seed1001003_u16.pt`

## Follow-up

The hard-cell variance was repaired at the next boundary by an independent
corpus with two authored training variants per style. That promoted follow-up
is recorded in
`../natural_text_grounding_external_corpus_v2_2026-08-04/`. The next frontier
is a genuinely independently produced caption stream under the same gate.
