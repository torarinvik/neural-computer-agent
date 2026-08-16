# World-independent operators across changed dynamics (2026-08-16)

Status: **development diagnostic; not a holdout and not a promotion**.
`AgentBrain.bank` is not read or modified. This is the missing experiment from
the consolidation list: it separates reusable control-flow/model operators
from raw world-specific successor data.

## Method

Each replicate has a source and target six-place ring with different hidden
transition orderings but the same event/task protocol. The source world is used
only to verify a versioned operator bundle. The bundle contains no transitions,
policies, or successor tensors; it specifies untried-first probing and
known-edge planning. The target learner gets the same 20 training episodes and
the same 5 evaluation checkpoints in every arm.

| arm | inherited object | purpose |
| --- | --- | --- |
| reusable | verified generic operator bundle | candidate transfer |
| fresh | nothing | matched fresh learner |
| irrelevant | uniform/ignore-mismatch bundle | irrelevant-inheritance control |
| corrupted | unknown-as-self-loop bundle | corrupted-artifact control |
| raw_successor | source-world successor policies/features | world-specific inheritance control |

The threshold is a normalized return of 0.75 that must remain satisfied at
every later measured prefix. A verifier emits one binary arrival outcome per
step, so the record reports distinct verifier bits, logical lifetimes,
optimizer updates, replay, wall time, decision latency, and stable bits to
threshold separately for every arm.

## Development result

Across three matched source/target pairs:

| arm | final normalized return | stable bits to threshold |
| --- | ---: | ---: |
| reusable operator | **1.000** | **192, 192, 192** |
| fresh learner | 0.989 | 192, 384, 192 |
| irrelevant artifact | 0.989 | 192, 384, 192 |
| raw successor artifact | 0.292 | none |
| corrupted artifact | 0.000 | none |

The development transfer ratio against the fresh learner is **0.833**. This
is evidence that the generic exploration/planning control-flow can be reused
without carrying the old transition model, while raw successor features fail
under the world change. It is not yet evidence for the rendered amodal
runtime: the stream is a finite event abstraction, and the operator bundle is
hand-specified rather than learned from a held-out operator-discovery task.

The irrelevant arm is intentionally identical in behavior to the fresh arm;
its purpose is to show that merely attaching an artifact does not create the
gain. The corrupted arm's failure is a safety check, not a tuning target.

## Decision and next step

Keep the operator-bundle boundary and reject raw successor-feature inheritance
across changed dynamics. Do **not** promote or admit the bundle. The next
experiment should freeze this contract, reserve a new seed block, and repeat
the comparison with learned event streams plus a within-lifetime dynamics
reversal so invalidation/recovery is measured rather than only serialized in
the interface.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.operator_world_transfer
```

The full three-replicate diagnostic takes well under a minute.
