# Persistent causal identity v2 closed-loop diagnostic (2026-08-16)

Status: **development diagnostic; not a holdout and not a promotion**.

This is the first development run of a stateful external self artifact through
the real policy-free live contract. Each arm runs rendered RGB frames through a
frozen learned frontend, separately bound event slots, an external identity
decision, the amodal controller/planner, an opaque decoder, a marker-world
transition, and a receipt-linked scalar outcome.

## Mechanism

`PersistentCausalIdentityV2` stores an action-conditioned learned-event
signature, never a slot number. At each episode it compares every current
track with that signature. A high-confidence assignment is the only persistent
update. Weak evidence, contradiction, or missing events quarantines the model;
quarantine freezes the statistics and requires two fresh high-confidence
episodes before relearning. Repeated ticks with the same `episode_id` are
update-idempotent.

## Result

| arm | integrated return | identity abstention | confident errors |
| --- | ---: | ---: | ---: |
| no persistent model | 0.250 | 0/24 | 0 |
| episode-local scorer | 0.375 | 9/24 | 0 |
| persistent causal v2 | 0.292 | 17/24 | 0 |
| oracle identity (evaluation ceiling) | 0.250 | 0/24 | 0 |

The persistent arm successfully rebound from slot 0 to slot 1 after the
episode boundary, but its signature became inapplicable later and quarantined.
The stale/reversed arm also quarantined rather than making a confident call.
This is a useful safety signal, not a positive behavioral result: persistence
did not yet beat the episode-local scorer (−0.083 return advantage), detected
the first quarantine at step 6, and recovered after 5 further steps only to a
0.25 post-recovery return. It must not advance to the reserved holdout.

## Controls

- action-shuffled history: abstained and quarantined;
- missing event presence: abstained without zero-filling and quarantined;
- exact equivalent tracks: abstained with zero support;
- partial mimic: abstained;
- crossing track order: abstained;
- birth/death single-track fixture: assigned the only available track;
- poisoned initialization: rebound to fresh evidence without a confident error.

The frontend digest stayed unchanged and the curated `AgentBrain.bank` digest
remained `07319eb1…e2e7c9`. The fixture keeps two learned tracks separately
bound within each episode; it does not yet validate true occlusion/birth/death
tracking, arbitrary crossings, corrupted memory, or a fresh-learner behavioral
holdout.

## Accounting and next gate

This run consumed 120 deterministic scalar outcome bits across 15 logical
lives, with zero optimizer updates. It is not eligible for promotion. Before
the reserved `integrated_self_model_holdout` block can be amended or spent,
v2 needs a better state-conditioned causal signature or an explicit rejection
record, then a new development audit covering true tracking, corrupted memory,
partial mimicry, and stable post-recovery behavior.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.persistent_identity_v2
```

The report is `persistent_identity_v2.json`; the accounting companion is
`sample_efficiency_ledger.json`.
