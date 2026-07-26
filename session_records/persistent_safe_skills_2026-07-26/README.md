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

## Seed 7971 result and fresh-confirmation repair

The storage boundary passed: parent and child reloaded exactly, child
corruption was detected, parent retrieval survived, and controller retention
held. The run nevertheless failed because the mastered challenger received a
false-positive promotion at 480 bits. Private audit showed a small utility
loss (`−0.13` points) and a large decision-pattern change. No persistent-skill
claim is promoted from this seed.

Seed 7972 changes one safety axis. A positive confidence bound now creates a
frozen **proposal**, not a deployment. The proposal must achieve a second
positive lower-95% bound on a fresh attempted-outcome block collected after
the proposal. Only then may it replace the incumbent or enter disk memory.
The proposal and confirmation use disjoint verifier outcomes; thresholds,
learning, task, and hidden-information restrictions remain unchanged.

The repaired integration passes only if the mastered proposal is rejected or
never proposed, the gap learner is independently confirmed and committed,
both skills reload exactly, corruption isolation passes, and retention holds.
Only a complete pass permits seed-7973 replication.

## Result

Seed 7972 passed the complete integration:

- mastered incumbent: no positive proposal and no change;
- gap learner: positive proposal at 480 bits (`lower95=+0.0047`), independently
  confirmed on the next 240 fresh outcomes (`lower95=+0.0018`);
- parent and promoted child reloaded bit-exactly from a fresh store instance;
- child corruption was detected while parent retrieval remained valid;
- prior controller skills were retained.

Seed 7973 did not replicate the learning half. It remained safe—no mastered
promotion—but the gap learner produced no positive proposal within 720 bits,
so no skill was committed. No longer run is authorized.

Therefore the durable-store machinery and one full promote→confirm→commit→load
path are demonstrated, but persistent safe learning is **not yet replicated**.
The current bottleneck is conservative-promotion sensitivity: safety is
reliable, while verified improvement sometimes fails to clear the confidence
threshold within the tiny budget. The next experiment should improve the
challenger's learning consistency or evidence efficiency without relaxing the
fresh-confirmation safeguard.
