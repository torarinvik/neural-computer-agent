# Persistent causal identity v3 closed-loop diagnostic (2026-08-16)

Status: **development diagnostic; not a holdout and not a promotion**.

V3 replaces v2's prefix-sensitive global covariance with a compact graph of
opaque action-labelled transitions between learned event states.  It stores no
slot number or coordinate.  A high-confidence assignment is cached for the
episode, the graph grows without counting repeated ticks as new persistent
updates, and known transition contradictions cause abstention/quarantine.

## Closed-loop result

The loop is:

```text
rendered RGB -> frozen learned events -> separately bound tracks
-> persistent causal identity -> policy-free planner -> opaque decoder
-> marker transition -> receipt-linked scalar feedback
```

The development fixture uses the same learned state region in alternating
episodes, but reverses slot order so a persistent model must rebind.  It uses a
deterministic probing pattern to expose several action-labelled transitions.

| arm | integrated return | identity abstention | confident errors |
| --- | ---: | ---: | ---: |
| no persistent model | 0.292 | 0/24 | 0 |
| episode-local scorer | 0.250 | 9/24 | 0 |
| persistent causal identity v3 | **0.375** | 12/24 | **0** |

V3's advantage over the episode-local scorer is `+0.125` in this bounded
fixture.  It rebinds from slot 1 to slot 0 and back without persisting a slot
identity.  The stale/reversed arm abstains and quarantines rather than making a
confident assignment; its integrated return is 0.333 with zero confident
errors.

## Controls

The action-shuffled, missing-evidence, exact-equivalence, crossing-order, and
partial-mimic controls abstain.  A single available dynamic track is assigned,
and a poisoned initialization relearns from fresh evidence.  Missing evidence
is never converted into zeros.  The frontend digest and curated bank digest
are unchanged.

These are mechanism controls, not a promotion claim.  The fixture remains
bounded, deterministic, and uses a fixed two-track learned-event stream.  It
does not yet validate arbitrary occlusion, true birth/death association,
corrupted persistent memory, a fresh-learner behavioral holdout, or general
cross-frontend identity transfer.  The reserved integrated-self holdout is
untouched.

## Accounting

The run consumed 96 deterministic scalar outcome bits across 12 logical
lifetimes, with zero optimizer updates.  The report and accounting companion
must preserve the exact frontend and `AgentBrain.bank` digests before and
after the run.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.persistent_identity_v3
```

The JSON report is `persistent_identity_v3.json`.
