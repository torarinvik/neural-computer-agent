# Rejected: common four-key action space for both slots (2026-08-06)

Routing accuracy was already >= 0.977 here, but forcing the Pong slot onto
four clamped keys biased exploration and capped its routed mastery at
0.752/0.760 on seed 69316 despite a 700-update matched budget escalation.
See game_routing_native_actions_v1_2026-08-06/README.md for the promoted
repair. Preserved as the acquisition-space control.
