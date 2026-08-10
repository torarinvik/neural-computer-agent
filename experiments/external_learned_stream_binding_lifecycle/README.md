# Outcome-trained stream-binding lifecycle policy

This experiment extends learned anonymous binding from safe caller-selected
replacement to an external policy that proposes which provisional identity
should replace which live track. The policy sees only opaque prototype vectors
and generic observation-count, reliability, delay, and similarity telemetry.
It never sees stream labels, task IDs, raw observations, or controller state.

Each proposal carries its exact logging propensity. A deterministic scalar
verifier outcome trains the policy online with one update and no replay. The
verifier remains the authority: a proposal is committed only through the
binding memory's atomic copy-on-write replacement transaction.

The pressure test uses multiple provisional identities, low-reliability
distractors, role permutation, a contradiction/all-negative control, a fresh
policy control, an outcome-shuffled policy control, atomic rejection, frozen
encoder/controller checks, and exact policy/memory persistence.

This promotes only a bounded outcome-trained proposal policy and atomic
retention boundary. It does not establish learned verifier design,
unrestricted growth, arbitrary semantic identity, or general continual
learning.
