# Stream-separated operation learning — 2026-07-30

## Breakthrough

The controller learns the inverse numerosity operation at **83.81% mean
held-out accuracy** from the same 1,024 lifetimes that previously produced
57.20%.

The change is representational, not semantic supervision:

1. a visual operation cue arrives as an ordinary sensory frame;
2. the clean count stimulus arrives immediately afterward;
3. the controller emits one opaque action and receives its scalar verifier
   outcome.

The controller never receives a task ID, count, correct action, unattempted
answer, coordinate, or symbolic operation name.

## Why the old rendering plateaued

The operation cue was drawn over every count frame. A frozen-parent audit
measured how much of the already learned larger-count decision survived:

| overlay cue scale | inherited comparison retained |
|---:|---:|
| 0 | 90.46% |
| 1/16 | 89.49% |
| 1/8 | 88.03% |
| 1/4 | 83.66% |
| 1/2 | 74.03% |
| 1 | 63.15% |

There was no static-overlay sweet spot. Weak cues preserved comparison but
were too similar to uncued replay; strong cues distinguished the operation but
destroyed the reusable relation.

A cue-only frame followed by a clean stimulus preserved 87.54% of the parent
comparison before any new learning. It factors “what operation?” from “what
objects?”, much like a human reading an instruction and then viewing the
problem.

## Amodal canonicalization

An early legacy adapter wrote directly into actuator logits after the amodal
intention. Later intention-level skills therefore could not see the complete
decision they were meant to transform.

The compatibility migration maps that legacy residual through the learned
actuator's right inverse and adds it to intention before later slots run. This
uses only learned controller weights. On 36,864 old-skill events it caused
**zero action changes**; maximum logit drift was `1.53e-5`.

## Replication and controls

Every arm used 128 updates, 1,024 unique new lifetimes, 6,144 verifier bits,
and balanced rehearsal. The streamed protocol uses two sensory frames per
action, or 12,288 new-task sensory frames.

| arm | seed | new operation | pair relation | magnitude | numerosity |
|---|---:|---:|---:|---:|---:|
| truthful | 25151 | **83.96%** | 98.90% | 87.25% | 82.64% |
| truthful | 25161 | **83.65%** | 99.00% | 88.85% | 84.92% |
| shuffled outcomes | 25161 | 13.08% | 99.21% | 92.46% | 90.93% |
| inherited intention removed | 25161 | 14.40% | 99.16% | 92.49% | 91.13% |

Removing the cue frame at evaluation reduced the two truthful replicas to
40.21% and 38.11%. Thus success requires all three:

- truthful verifier outcomes;
- inherited comparison intention;
- the separate sensory operation cue.

Each complete local MPS run took 16–18 seconds.

## Rejected tiny forks

- Extending the overlaid-cue run from 128 to 256 updates did not improve its
  ~59% plateau.
- A nonlinear gate and an explicit locality price changed routing but did not
  improve the task.
- Weak static overlays and a cue only on the first lifetime event failed.
- Scalar and decision-axis intention operators did not improve the supervised
  ceiling and were removed.
- Disposable supervision showed 85.1% was representable when cue interference
  was removed, localizing the remaining failure to learning/protocol rather
  than missing visual facts.

## Claim boundary and frontier

This is replicated, causally audited **strong cross-operation competence**, not
yet the project's 90% mastery standard. It improves held-out accuracy by
26.61 points over the earlier overlay protocol without increasing verifier
experience, while retaining all inherited skills.

The cost is one extra sensory controller step per action. The next frontier is
to cross 90% while learning when a separate cue frame is worth its latency,
then store the inverse operation as a reusable persistent skill.

Machine-readable reports and both causal audits are stored beside this file.
