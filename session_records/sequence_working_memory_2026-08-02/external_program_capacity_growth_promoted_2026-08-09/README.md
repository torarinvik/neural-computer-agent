# Verifier-gated external program capacity growth — promoted 2026-08-09

This archive records the two-seed pressure test for safe growth of the
memory-side executable address space. Each seed first learned two opaque
routes from one-pass scalar verifier outcomes. A copy-on-write transaction
expanded router capacity from two to three only after a retention probe passed
on both the source and expanded state. The third slot was then activated and
learned from scalar outcomes.

Both seeds retained 100% accuracy on the two mastered routes after the new
route was learned and reached 100% across all three routes. The rejected
growth control left capacity and state unchanged; reward-shuffled feedback
stayed at 66.7%; persistence was exact; and the frozen controller and
plasticity rule digests were unchanged. The runs used zero optimizer updates,
zero replayed examples, and retained zero raw feature rows.

This is a narrow promoted capability: verifier-gated bounded external
address-space growth with retention. It does not establish unrestricted
memory growth, arbitrary program induction, positive transfer against a fresh
learner, or general continual learning.
