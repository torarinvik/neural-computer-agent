# Provenance-closed recursive recipe compaction — promoted bounded result

This audit pressure-tests finite capacity for the external recipe-file memory.
Each run admits a protected non-commuting depth-four chain plus three
unreferenced decoy files, creating ten resident files. A copy-on-write
compaction request keeps only the depth-four root. The memory automatically
retains the root's transitive provenance closure and every protected source,
then submits the candidate to an independent behavior verifier.

Across seeds `17–20`, all runs compacted ten files to the seven-file closure,
removed exactly three decoys, retained the root and every protected source at
`1.0000`, reloaded with an exact checksum, and left the source memory
byte-identical. A verifier-rejected compaction was a no-op in every run. The
controller and recipe interpreter were frozen; optimizer updates and replayed
training examples were zero.

This promotes the storage contract needed for capacity pressure: safe
provenance-closed copy-on-write compaction. It does not yet establish learned
eviction economics, learned semantic compression, unrestricted archive growth,
or general continual learning.

Run with:

```text
PYTHONPATH=. uv run python experiments/recipe_expressibility/verified_recipe_compaction.py --report-out report.json --seeds 17 18 19 20
```
