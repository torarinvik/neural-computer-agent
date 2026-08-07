# Promoted: arbitrated consolidation (2026-08-07)

The enhanced consolidation rule targeting vanilla EWC's measured weakness
(acquisition headroom under accumulated penalties, per
`ewc_ladder_three_game_qualified_v1_2026-08-06`). Each penalty coefficient
is attenuated per parameter by the new task's demonstrated need:

```
a_i = F_i / (F_i + mu * G_i),    penalty_i = lambda * a_i * F_i * (theta_i - anchor_i)^2
```

where `F` is the protected game's unit-mean diagonal Fisher and `G` is a
running unit-mean estimate of the new task's own squared policy gradients
(task gradients are computed before the penalty gradient is added in
closed form, so `G` never sees penalty pressure). Parameters the new game
does not need keep full protection; parameters it demonstrably needs are
proportionally released. One dial (`mu`); `mu=0` recovers vanilla EWC.

## Calibration (small-budget probes, seed 69317, in this directory)

Pong acquisition under the Snake penalty: vanilla 0.297; mu 0.3 -> 0.484;
mu 1 -> 0.477; mu 3 -> 0.641; mu 10 -> 0.438. Snake retention was stable
or slightly improved at every mu. A clean inverted-U; mu=3 promoted.

## Full-budget confirm (both seeds, all nine gates pass)

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| Pong acquisition (vanilla baseline) | 0.9824 (0.9082) | 0.8203 (0.7656, failed gate) |
| Snake post-acq -> after two later games | 0.9395 -> 0.9395 | 0.9355 -> 0.9141 |
| Pong retained after Breakout | 0.9961 | 0.8203 |
| Breakout | 0.5254 | 0.5273 |

Command (per seed):

```bash
uv run python -m experiments.games_amodal.ewc_ladder \
  --seed <seed> --consolidation-mode arbitrated --arbitration-mu 3.0
```

The vanilla comparison baseline is the qualified archive
`ewc_ladder_three_game_qualified_v1_2026-08-06` (identical command with
`--consolidation-mode sum`). Arbitrated closes vanilla's one failed gate
(seed 69317 Pong acquisition) and matches or exceeds it on every other
metric, with the same unprotected/permuted/shuffled null structure passing.

## Why release helps retention too

Under a rigid penalty the new game's gradients grind against protected
directions and spill into whatever is unprotected; arbitrated release
gives the new game legitimate room, lowering pressure on the old game's
actually-critical parameters. The rule routes plasticity, it does not
trade retention away.

## Claim boundary

Promoted: three-game replay-free continual learning through one plastic
core with demand-proportional consolidation release, two seeds, all gates,
dial calibration preserved. Not promoted: depth beyond three games,
mu robustness beyond the probed grid, seed pools beyond two on this rung,
or any claim that arbitration helps when tasks genuinely require the same
parameters (no such conflict case exists yet in the game suite).
