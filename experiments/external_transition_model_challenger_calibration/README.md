# Broader disjoint transfer-challenger calibration

This audit addresses the failure mode exposed by the rejected five-seed
disjoint compounding population: a short transfer-vs-fresh probe may choose a
candidate whose eventual acquisition cost is worse.

It uses seven distinct transition tables, two source dynamics, and five target
dynamics. For every target, both the transfer and fresh candidate receive the
same eight-update probe and are then trained independently to the full mastery
gate. The probe winner is compared with the actual full-cost winner; only the
probe-selected candidate is committed to the live external bank. A random
intention floor and byte-stable prior-slot checks remain in the audit.

The promoted three-seed result matched the probe winner to the full-cost winner
on all 15 target comparisons, while every candidate mastered and all prior
slots remained stable. This calibrates a bounded challenger mechanism; it does
not establish unrestricted memory growth or general continual learning.
