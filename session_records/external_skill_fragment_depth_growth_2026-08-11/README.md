# External skill-fragment depth growth

This record captures the first decisive diagnostic for the append-only
trace-conditioned growth boundary. It is not a promoted general continual
learning result.

## What changed

`ExternalSkillFragmentGrowthCombiner` provides one shared trace encoder and
canonical readout plus zero-initialized external residual slots. A slot is
appended for each new composition depth, trained from fresh verifier outcomes,
and protected before the next depth. The parent controller, register
interpreter, and acquired fragment bank stay frozen on the inherited path.

## Evidence

The seed-69316 joint-foundation diagnostic trained all four atomic fragments
against one shared foundation for 128 updates, then trained the depth-2 slot
for 256 updates without replaying atomic examples. Atomic foundation accuracy
was `[0.921875, 0.958333, 0.992188, 0.940104]`. The 12 ordered depth-2
program accuracies were:

`[0.955729, 0.916667, 0.916667, 1.0, 1.0, 0.994792, 1.0, 0.916667, 1.0, 1.0, 0.979167, 0.9375]`.

The minimum was `0.916667`; the final per-target held-out minimum reported by
the shared stage was `0.914063`. This is a strong signal that a protected
external slot can acquire new compositional capacity while the CPU-like
interpreter and controller remain frozen.

## Rejected controls and lessons

- Sequentially acquiring four atomic representations against one shared
  decoder was rejected: retention ended at `rotate=0.7474`,
  `complement=0.3828`, and `prefix_parity=0.5807`.
- A summary-only residual MLP slot was rejected at depth 2; a full
  trace-conditioned segment slot improved the 128-update minimum from about
  `0.61` to `0.75`, then to `0.9167` at 256 updates.
- The direct diagnostic did not yet execute the full depth-3/depth-4 gate,
  replicated seeds, or the complete corruption/reversal/missing-evidence
  audit. Those remain promotion requirements.
- A full depth-1-to-depth-4 run with 128 updates per growth rung was started
  with reduced audit sampling but exceeded 17 minutes while still in the
  matched fresh lineage. It was interrupted before report emission; no
  capability result or accounting claim is inferred from that incomplete run.

## Claim boundary

This establishes an architectural seam and a positive bounded-growth
diagnostic. It does not establish unrestricted memory growth, compression,
arbitrary new computation, or general continual learning.
