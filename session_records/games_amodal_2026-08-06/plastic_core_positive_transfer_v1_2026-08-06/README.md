# Measured: positive transfer through the consolidated plastic core (2026-08-06)

First reliable positive-transfer evidence in this program. The fresh-core
baseline (this directory) trains Pong from scratch with the exact budget,
seeds, and architecture used by the promoted EWC rung's protected
condition; the comparison target is the archived
ewc_consolidation_plastic_core_v1_2026-08-06 reports.

| Pong acquisition | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| Snake-trained plastic core + EWC | eval 0.8125, mean-train 0.5642 | eval 0.9688, mean-train 0.6832 |
| fresh core (this baseline) | eval 0.7051, mean-train 0.4921 | eval 0.6641, mean-train 0.2252 |

The consolidated core wins on both seeds while simultaneously retaining
Snake (0.8438/0.9395 in the EWC archive). Contrast: the frozen-core rung
showed seed-negative transfer (-0.41 on 69316). The difference is
plasticity - Pong reshapes everything Snake does not need instead of
being forced through a fixed alien map.

Claim boundary: this is a cross-run comparison at matched seeds, budgets,
and configurations, not a same-run randomized audit; it involves one game
pair and two seeds. A same-run harness with a shuffled-reward null and
more seeds is required before promoting "reliable positive transfer" as a
headline claim. Zero replay in all conditions.
