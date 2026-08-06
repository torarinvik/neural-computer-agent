# Qualified: three-game routing with Breakout (2026-08-06)

The promoted routing mechanism extends structurally to three candidate
slots: candidate permutation invariance 1.0000, outcome-shuffled null at
chance, frozen slots unchanged, zero replay, and the visually distinct
game (Snake) routes at 1.0000 on both seeds with 0.94-0.97 routed mastery.

Not promoted: (a) the Breakout slot plateaus at ~0.54 routed mastery on
both seeds and did not respond to a 700-update budget escalation - a slot
acquisition limit of the standalone policy on the harder compound game,
not a routing failure; (b) Pong and Breakout (ball-and-paddle siblings)
confuse the router at 0.69-0.90 accuracy even with 5-frame queries. The
confusion is similarity-structured, which is evidence the router reads
real visual structure from opaque events; games that confuse the router
are the games that should share bank fragments in the compositional-
transfer rung.

Stopping rule applied: one matched escalation
(report_seed*_budget_control.json = 400/256/3-frame baseline), then
qualify. Next designs: stronger slot policies for compound games and
key separation trained on sibling contrast, not more budget.
