# Artifact policy

`checkpoints/` is reserved for causally validated milestone models. The cleanup
left it empty because none of the old checkpoint files belongs to the retained
architecture frontier.

`manifests/curated_checkpoints.sha256` verifies any included checkpoints. It
contains no checksum entries while the curated checkpoint set is empty.

When a new artifact becomes scientifically load-bearing, place it in
`checkpoints/`, document the causal evidence that promotes it, and refresh the
curated checksum file. Generated caches and disposable checkpoints stay out of
Git.
