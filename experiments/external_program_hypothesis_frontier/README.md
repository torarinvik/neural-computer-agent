# External program hypothesis frontier

This experiment tests the next architectural seam after one-edit candidate
search: a bounded, persistent frontier of opaque multi-step executable files.

The interpreter and controller remain frozen. A memory-side frontier performs
generic replace/insert/delete/swap edits, retains a protected root plus a
bounded set of provisional hypotheses, and consumes only deterministic scalar
verifier outcomes. A verified target is committed through the existing
copy-on-write external-file admission transaction.

The promoted claim is intentionally narrow:

- a useful opaque parent can be composed into a new three-step executable file;
- a random parent requires more verifier exposures;
- the source file remains protected and mastered;
- failed corruption does not mutate memory;
- frontier and file state round-trip exactly;
- the controller and interpreter receive zero optimizer updates.

This is not open-ended program induction, unrestricted memory growth, or
general continual learning. The next pressure test is to replace the synthetic
atom chain with a family of rendered Brain Workshop tasks while retaining the
same opaque frontier and verifier boundary.

Run one seed with:

```bash
uv run python -m experiments.external_program_hypothesis_frontier.train \
  --seed 23001 \
  --report-out /tmp/external_program_hypothesis_frontier_23001.json
```
