# Shared external composition learner — 2026-08-11

This pressure test asks whether one external learner can reuse one frozen
four-fragment bank across multiple ordered programs. It is stricter than the
promoted multi-target result, which allocated a fresh combiner per target.

The implementation adds a versioned rich execution trace containing learned
instruction codes, transition deltas, and fragment segment lengths. Routing
receipts remain diagnostic memory-side state: `ExternalSkillFragmentLearnerTrace`
physically removes fragment indices and route scores before a combiner sees the
trace. The register interpreter also batches rows with equal executable length,
preserving variable-length semantics while reducing avoidable transport cost.

## Results

| arm | training accuracy | held-out accuracy | stable bits | promoted |
| --- | --- | --- | ---: | --- |
| state-only shared learner | 0.9245 / 0.8984 / 0.8750 | 0.5443 / 0.4792 / 0.4766 | 55,296 | no |
| rich flat trace learner | 0.9141 / 0.9427 / 0.9141 | 0.4583 / 0.4453 / 0.4583 | 36,864 | no |
| rich trace + atomic anchors | 0.8542 / 0.7839 / 0.8932 | 0.5208 / 0.5391 / 0.5781 | none | no |
| rich trace + six training orders, 64 updates | 0.7682 / 0.6563 / 0.7656 / 0.7813 / 0.6641 / 0.5703 | 0.5729 / 0.4349 / 0.6146 | none | no |
| rich segment trace, batched, 128 updates | 0.6536 / 0.9531 / 0.7760 | 0.6276 / 0.5182 / 0.6094 | none | no |

The matched segment run completed in 353.5 seconds after batching, versus
496.5 seconds for the row-by-row transport implementation. The speedup is a
transport improvement, not a capability promotion. None of the shared arms
passed stable-prefix transfer and held-out composition gates. No replayed
examples were used; the parent and acquired bank remained frozen and their
checksums were unchanged. The full reports were generated under `/tmp` during
the audit; the table above is the durable decision record.

## Interpretation

The current bottleneck is not storage growth or routing isolation. It is
learning a target-agnostic composition law from too few ordered examples while
the acquired fragment representations were learned under separate atomic
objectives. Atomic anchor loss made the shared learner worse, so it is rejected
as the default fix. The next experiment should vary composition depth and
provide a curriculum of fresh opaque program orders to one shared learner,
while keeping the bank frozen and charging every verifier bit. Adding one
combiner per target would hide this bottleneck and is not the architecture we
want.

Claim boundary: the repository still promotes bounded frozen-bank composition
with separate target adapters, not universal program induction, unrestricted
memory growth, compression, or general continual learning.
