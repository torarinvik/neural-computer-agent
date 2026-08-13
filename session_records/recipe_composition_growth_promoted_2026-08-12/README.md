# Verifier-gated external recipe composition — promoted bounded result

This audit tests the frozen-controller CPU/files boundary: three protected
one-step recipe files are composed into a verified two-step file, then that
file is composed into a verified three-step file. The recipe interpreter and
controller are unchanged during growth. The mixed domains are explicit:
the first slot uses modulus 2 and the second uses modulus 8. The conditional
operation makes reversed order behaviorally distinguishable.

Across four seeds, both composed files reached `1.0000` on held-out states,
all atomic files retained `1.0000`, and the composed provenance reloaded with
an exact checksum. Reversed-order controls scored `0.0000`; empty evidence was
a no-op; shuffled verifier outcomes never admitted a file. No replayed rows or
controller optimizer updates were used.

The optional factorized proposal policy is reported but not promoted as a
sample-efficiency result: it was no slower than fresh on three of four seeds
and slower on seed 18. The promoted claim is therefore the bounded external
memory/composition boundary, not general learned program induction.

Run with:

```text
PYTHONPATH=. uv run python experiments/recipe_expressibility/verified_composition_growth.py --report-out report.json --seeds 17 18 19 20
```

Claim boundary: replay-free, verifier-gated growth through depth-three serial
composition of protected external files. This does not establish arbitrary
program induction, unrestricted memory growth, or general continual learning.
