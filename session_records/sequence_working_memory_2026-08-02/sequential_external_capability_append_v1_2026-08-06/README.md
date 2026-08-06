# Sequential external-capability append — 2026-08-06

Status: promoted replicated two-seed protected-capacity append result.

This audit starts with the three-program parent-conditioned external bank
(`reverse4`, `forward4`, `complement4`) at physical capacity. Each row is
mastered through fresh held-out verifier probes and becomes protected by the
opaque retention ledger. A fresh `rotate4` capability is then acquired in an
isolated recurrent context program plus capability-local decoder.

The append is intentionally attempted before the bank is enlarged. Both seeds
reject the write with the explicit protected-capacity error rather than
evicting an existing capability. `ExecutableArtifactMemory.grow()` copies the
four existing contracts—artifact files, opaque addresses, checksums, and
retention state—into a new capacity-four bank. The new artifact is then
written, its retention is verified, and only a memory-side route extension is
trained on fresh `rotate` queries. The parent controller and the three-row
router remain frozen after the append point.

Both seeds pass every promotion gate:

- protected capacity rejection and preservation during transactional growth;
- stable-prefix mastery and retention for all four external programs;
- old-route, new-route, combined-route, and candidate-permutation accuracy of
  `1.000`;
- balanced reward-shuffled extension control at `0.297` and `0.000`;
- causal separation from the wrong artifact;
- exact bank and route reload, corruption rejection, frozen parent/router
  digests, and zero replay after append.

The promoted selected accuracies are `0.918/0.973/0.902/0.945` for seed
`69316` and `0.898/0.973/0.984/0.988` for seed `69317` in the order
`reverse4/forward4/complement4/rotate4`.

The rejected random-key control is retained beside the reports. Its opaque
router failed to recover the old three-way route (`0.536`), showing that route
keys must be learned address evidence rather than arbitrary random IDs. This
is a useful architectural constraint, not a promoted capability result.

This promotes one sequential protected-capacity append for a frozen processor.
It does not establish repeated open-ended growth, eviction/consolidation under
nonstationarity, route reversal, arbitrary program synthesis, or general
continual learning.
