# Contrastive batch size, replicated (F144)

Probe 244 + replication. Contrastive auxiliary loss at w=1.0, batch 32
worlds, linear binder, 256 worlds, depth<=4.

    held-out per-bit  0.7993 / 0.8447 / 0.6945   mean 0.7795
    exact match       0.3158 / 0.3543 / 0.0794   mean 0.2498
    stranger          0.5388 / 0.6008 / 0.5621
    joint training 0.5283 | privileged ceiling 0.9723 | chance 0.5000

Best non-privileged reader result: 56.6% of the joint-to-ceiling gap,
exact match 26x joint. Held back until the best point replicated, per
the rule earned when F142 made the same claim off one seed and failed.

The 128-batch arm (0.5738) is single-seed and not claimed.
