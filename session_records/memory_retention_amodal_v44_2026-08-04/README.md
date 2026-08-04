# Persistent-memory retention boundary qualification

v44 repeats the consolidated v43 one-seed rung and adds an end-to-end
persistent-memory audit. The learned retention episode writes through the
disk-backed memory implementation, closes/reopens the backend, and queries
the reloaded store. The reload intact recall is `1.000`, matching the
in-process result.

The same run retains `1.000` mastered-primitive and unseen-token retention and
reproduces the `1.538x` fresh-over-transferred stable-bit ratio (13,312 versus
20,480). Storage uses a temporary atomic snapshot; no disposable memory file
is promoted or added to Git.

This qualifies the persistent-memory interface for the narrow retention
verifier on one seed. Replicate persistence, corruption recovery, and transfer
retention across the v41 population before promoting a checkpoint or claiming
general persistent episodic memory.
