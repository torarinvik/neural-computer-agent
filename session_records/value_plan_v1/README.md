# value_plan_v1 (F247)

Probe: experiments/games_amodal/probes/value_plan.py
Seeds: 69316 1234 4242 555 31337 2718 (6), interpreter check 1.0 all.

Deployable VALUE-PLAN: depth-d search over composed bank programs
(plant-executed), learned per-action reward head + H=8-step value
head, ridge-fit on random-rollout returns over the generic
relational basis. Arms: random, vplan d1-d4, one policy-iteration
round (d4), shuffled-weights binding control, privileged-dynamics
truedyn_d2 localization arm.

Headline: on the solved control world the planner is depth-monotone
and policy iteration reaches +1.30 vs the privileged depth-4 truth
ceiling of +0.41 (shuffled control collapses to random). On the
F245 witness worlds vplan ~ random while truedyn recovers most of
the certified gap with the same value head: failure localized to
the learned bank-dynamics layer (relative-motion conditionals are
inexpressible in the current ISA). See F247 in
docs/MEMORY_BANK_DESIGN.md.
