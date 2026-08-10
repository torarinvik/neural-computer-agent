# Contrastive reader (F139)

Probe 239. Reader pre-trained with InfoNCE over world identity (no
privileged information), frozen, then plant trained to bind its code.
Held-out worlds, per-bit: 0.5646 / 0.6136 / 0.6287 own vs 0.541 /
0.538 / 0.580 stranger.

Best non-privileged scheme measured (beats joint 0.5283 and task-loss
0.4973) but far from distilled 0.9723. Suspected cause: the oracle
entry is a linear projection and the binder is linear, so F135 was
measured under a matched pairing; a contrastive code is arbitrarily
arranged and needs a nonlinear binder.
