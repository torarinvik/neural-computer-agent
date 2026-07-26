# Persistent safe skills — pre-registration

## Question

Can a safely promoted latent skill be committed to long-term disk memory,
reloaded into a fresh process, and used without damaging or replacing its
verified parent skill?

## Boundary

The store receives only an opaque controller payload, a context key derived
from the four learner-visible memory statistics, the attempted-outcome
promotion lower confidence bound, verifier-bit accounting, and lineage
provenance. It receives no task label, correct action, unattempted outcome, or
semantic skill name.

Commits are permitted only when the promotion lower confidence bound is
strictly positive. Skill files are immutable and content-addressed. Manifest
updates are atomic. Every load recomputes SHA-256 before deserialization.

## Sub-minute gate

Seed 7971 repeats the replicated global-centered safe adaptation and commits
the promoted gap incumbent alongside its already verified parent. The
integration passes only if:

1. the mastered incumbent is not promoted or degraded;
2. the gap learner is promoted from attempted-outcome evidence;
3. both parent and child reload exactly in a fresh store instance;
4. the reloaded child reproduces its pre-save outputs and audited utility;
5. the parent remains byte/hash valid and reproduces its original outputs;
6. corrupting a disposable child copy is detected without blocking parent
   retrieval;
7. prior controller retention checks pass.

Only a full pass permits unchanged replication.
