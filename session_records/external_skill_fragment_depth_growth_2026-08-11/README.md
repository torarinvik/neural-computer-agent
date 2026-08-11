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

## Cumulative protected-prefix follow-up

The growth combiner is now versioned as
`neural-computer.skill-fragment-growth-combiner.v2`. New instances apply all
protected depth slots cumulatively to deeper traces; exact-depth application
remains an explicit compatibility mode. This makes acquired composition
capacity reusable at later depths rather than silently discarding the prefix.

The matched seed-69316 depth-3 audit used `1,157,568` unique verifier bits,
`182,272` logical lifetimes, `832` optimizer updates, and zero replay. Atomic,
pair, and trained-triple minima were `0.9896`, `0.9688`, and `0.9896`; the
prefix remained frozen and the no-fragment-bypass, missing-evidence,
persistence, frozen-parent, and zero-replay controls passed. Held-out triple
accuracy improved to `[0.5833, 0.6042, 0.4688]` from the prior exact-depth
`[0.5729, 0.5729, 0.3229]`, but the held-out gate still failed. The corrected
maximum-depth wrong-order control reached `0.7083`, below the mastery
threshold after excluding semantically commutative lower-depth pairs.

This is a positive architectural improvement, not a promotion of general
continual learning. The remaining bottleneck is reusable operator algebra:
the new slot can fit fresh depth-3 programs, but it still does not reliably
infer unseen compositions from protected fragments.

## Rejected joint-foundation curriculum

An explicit foundation-depth-2 curriculum was also tested. It jointly updated
the shared interpreter and old prefix while learning pairs, then froze them
before depth-3 growth. Pair and triple fits were strong, but atomic retention
collapsed to `0.5208` minimum after growth (source baseline range
`0.5104–0.7292`). This is a direct no-replay catastrophic-forgetting failure;
the curriculum is rejected. New learning must preserve the admitted prefix
through isolation or a verified external expansion, not update old computation
in place.
