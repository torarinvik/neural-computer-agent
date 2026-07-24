# Artifact policy

`checkpoints/` contains only the current, causally validated milestone models.
They are intentionally small enough to accompany the Git repository.

`manifests/curated_checkpoints.sha256` verifies the included checkpoints.
`manifests/hf_checkpoints.sha256` verifies the same files after they are
packaged under the Hub repository's `checkpoints/` directory.

`manifests/historical_artifacts.sha256` records the hashes and original paths
of older checkpoints, compressed session bundles, and compiled Elisa binaries
that remain in the original `elisa-screenwatch` workspace. Those historical
files total several gigabytes and are excluded from normal clones.

The historical manifest preserves provenance; it is not a download mechanism.
When an older artifact becomes scientifically load-bearing, copy it into
`checkpoints/`, document why it is needed, refresh the curated checksum file,
and force-add it to Git.
