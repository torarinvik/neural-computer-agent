# Promoted: the staggered six-game battery (2026-08-07)

Six calibrated simple games (choice twins, dual trio, avoid) on one
fixed plant with a disjoint-oracle fragment bank, admitted one context
every 300 updates (3600 total, batch 16, steps 24). Mastery is
knowledge-scored (F15) and judged against per-game SOLO CEILINGS at the
same budget (choice 1.00/1.00, dualAC 1.00, dualAD 0.686, dualBC 0.720,
avoid 0.922).

```bash
uv run python -m experiments.games_amodal.fragment_bank \
  --seed <seed> --suite battery --oracle-selection --oracle-map disjoint \
  --warm-updates 0 --updates 3600 --batch-size 16 --steps 24 \
  --fragments 16 --balance-contexts --stagger-updates 300 --cross-pairs 2 \
  --ignorance-weight 0.5 --ignorance-every 3 --adapt-updates 150
```

## Result (ratio to solo ceiling)

| game | 69316 | 69317 | simultaneous baseline (69316/69317) | 2x-budget control |
| --- | ---: | ---: | ---: | ---: |
| choiceA | 1.00 | 1.00 | 0.47 / 1.00 | 1.00 |
| choiceB | 1.00 | 0.72 | 0.16 / 0.44 | 0.42 |
| dualAC | 1.00 | 0.86 | 1.00 / 1.00 | 1.00 |
| dualAD | **1.16** | **1.38** | 0.93 / 0.92 | 0.99 |
| dualBC | **1.05** | 0.84 | 0.90 / 0.67 | 0.97 |
| avoid1 | 1.00 | 0.95 | 1.00 / 0.98 | 1.00 |

Worst game staggered: 0.72x solo. Worst game simultaneous: 0.16x. The
2x-budget control shows compute alone does not cure the twin collapse
(choiceB 0.42) — arrival order does. Twin cross-feeds stay at 0.00-0.05
both seeds: fragments still specify, not merely help. dualAD finishing
ABOVE its solo ceiling on both seeds (1.16x, 1.38x) is the program's
first replicated super-solo transfer: earlier contexts made a later one
easier.

## Claim boundary

Promoted: six contexts, three rule families, zero replay, staggered
admission as the battery's default protocol (F18); no context collapses
and at least one context shows super-solo transfer, both seeds. Not
promoted: seed-2 softness (choiceB 0.72, dualBC 0.84 — above the 0.7
gate but below seed 1's sweep), learned selection at this scale, and
composition (dualBD holdout remains unsolved by imposed allocation, F16).
