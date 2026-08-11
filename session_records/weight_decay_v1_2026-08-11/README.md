# Weight decay (F154)

Probe 252, 2 seeds per setting, 40k updates.

    wd=0.00 @  40k : 0.7795 own, 0.2498 exact   (F144, 3 seeds)
    wd=0.01 @  40k : 0.8569 own, 0.3509 exact   <- 2 seeds, 0.8538/0.8599
    wd=0.00 @ 100k : 0.8704 own, 0.4228 exact   (F147, 2 seeds)
    wd=0.10 @  40k : 0.6725 own, 0.0903 exact   <- destructive

Weight decay 0.01 recovers ~85% of the gain from 2.5x more training at
40% of the compute. Interior optimum: 0.1 is worse than none.

First literature-derived mechanism to work (after semi-amortization,
codebook, curriculum and two-phase all measured null). The curves are
smooth at every setting, so it changes the RATE, not the shape — the
intervention is vindicated, the grokking framing that suggested it is
not.
