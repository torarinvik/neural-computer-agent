# Protected artifact capacity growth

This pressure test makes finite-capacity behavior explicit. A two-row
executable-artifact bank receives fresh successful verifier outcomes until
both opaque capabilities are protected. A third write must refuse eviction
with an explicit protected-capability error. The caller then creates a larger
verified bank, copies the artifacts and retention ledger without mutating the
source, and admits the new artifact only after growth.

This qualifies a safe memory-capacity escape hatch. It does not claim learned
capacity planning, learned compression, arbitrary new skill acquisition, or
general continual learning.

Example:

```bash
uv run python -m experiments.artifact_capacity_growth_amodal.train \
  --report-out /tmp/artifact-capacity-growth-seed69316.json \
  --seed 69316
```
