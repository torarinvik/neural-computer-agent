# pair_dynamics_v1 (F249)

Probe: experiments/games_amodal/probes/pair_dynamics.py
Seeds: 69316 1234 4242 555 31337 2718 (6).

2x2 factorial {per-slot base-ISA fit vs joint pair fit over generic
relational motions} x {nearest-rank encoding vs identity-stable
continuity tracking}, held-out group exact-match on the F245 trio.
Identity tracking converts mover fits 0.38-0.45 -> 0.72-0.85; the
pair vocabulary adds nothing under tracking (second refused
primitive witness in a row). Also exposed: per_slot_search modulus-
overfit tie-break (INC mod5 train 0.99 / held 0.15); intercept
fallers are fatal when missed. Limitation: the all-slots-present
filter starved worlds with structurally empty planes (avoid2_delayed3,
control cells are None). See F249 in docs/MEMORY_BANK_DESIGN.md.
