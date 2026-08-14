# aliasing_capacity_v1 (F255)

Probe: experiments/games_amodal/probes/aliasing_capacity.py (v2)
Seeds: 69316 1234 4242 555 31337 2718 (6). 12,288 samples/world.

Distillation audit: matched kNN on tracked-8 vs privileged full
state. Core exonerated (gap ~0 on trio, 0.00 on control); knn_full
itself reproduces the -0.85 stall on the trio while matching the
privileged anchor on the control: the stall is sample complexity of
the compound decision function, not state. See F255 in
docs/MEMORY_BANK_DESIGN.md.
