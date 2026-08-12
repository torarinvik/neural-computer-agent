# Verifier-gated recursive external recipe composition — promoted bounded result

This audit pressure-tests whether the frozen-controller CPU/files boundary can
grow an external recipe through a recursive depth-four chain. Four atomic
files are admitted first over mixed slot domains `(2, 4, 8)`:

- `INC(slot0, m=2)`;
- `CINC(slot1 | slot0, m=4)`;
- `INC(slot2, m=8)`;
- `CDEC(slot2 | slot1, m=8)`.

The external search composes those protected files into depth two, depth three,
and depth four programs. The controller, recipe interpreter, and atomic files
remain frozen. The optional composition policy receives only opaque source
digests, generic recursive shape/depth descriptors, and aggregate scalar
verifier quality. No verifier rows, task labels, correct actions, replay, or
controller optimizer updates are used.

All four seeds passed the structural promotion gates: held-out accuracy was
`1.0000` at every depth, atomic-file retention stayed at `1.0000`, recursive
provenance survived reload, reversed depth-two/depth-four programs were
rejected behaviorally, empty evidence was a no-op, shuffled feedback was
rejected, and memory/policy checksums reloaded exactly.

The warm/fresh proposal diagnostic is deliberately not promoted as a reliable
sample-efficiency gain. Ratios were `0.2917`, `1.2500`, `0.3200`, and `0.3810`
for seeds `17–20`; seed 18 was slower than fresh. The promoted result is the
bounded recursive external-memory capability, not universal transfer.

The arithmetic boundary is explicit per family. In particular, the two-valued
increment uses `m=2`; the old global-modulus-8 interpretation is not used, and
the two-valued toggle is correctly understood as two existing increments.

Run with:

```text
PYTHONPATH=. uv run python experiments/recipe_expressibility/verified_recursive_composition_growth.py --report-out report.json --seeds 17 18 19 20
```

Claim boundary: replay-free, verifier-gated recursive growth through a
depth-four external recipe file. This does not establish arbitrary program
induction, unrestricted memory growth, or general continual learning.
