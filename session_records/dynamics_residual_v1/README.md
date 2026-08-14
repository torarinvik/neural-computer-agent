# dynamics_residual_v1 (F248)

Probe: experiments/games_amodal/probes/dynamics_residual.py
Seeds: 69316 1234 4242 555 31337 2718 (6).

Per-slot program fits (base ISA, 32 vs 256 train examples; extended
ISA with TOWARD/AWAY, 256) scored by held-out exact-match per slot,
on the F245 witness worlds + control. TOWARD witness refused:
ext256 == base256 on every trio world; residual is cross-axis
coupling (pursuer larger-gap-axis rule) + nearest-rank identity
switching, concentrated in mover slots (avatar slots 1.00
everywhere). See F248 in docs/MEMORY_BANK_DESIGN.md.
