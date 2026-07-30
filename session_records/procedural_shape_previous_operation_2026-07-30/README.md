# Previous-item operation curriculum

## Result

The fixed-span procedural-shape controller now performs its first replicated
sequence manipulation. It stores three independently rendered shapes, receives
a visual ordinal cue plus an arbitrary visual operation glyph, and answers
either:

- a direct lookup: compare the candidate with the cued item; or
- a previous-item lookup: compare the candidate with the item immediately
  before the cued item.

The learner receives RGB streams, its own opaque binary action, and the scalar
success or failure of that attempted action. Shape identities, ordinals,
operation IDs, targets, correct actions, and verifier metadata remain private.

Both independent lineages pass a 24,576-lifetime natural-distribution audit:

| lineage | overall | direct | previous | conflict | weakest position/target cell |
|---|---:|---:|---:|---:|---:|
| primary | 98.30% | 98.82% | 97.78% | 96.65% | 96.44% |
| replica | 98.59% | 98.95% | 98.23% | 97.43% | 96.88% |

The two previous-item anchor cells also pass independently:

| lineage | previous from cue 1 | previous from cue 2 |
|---|---:|---:|
| primary | 98.83% | 95.67% |
| replica | 99.24% | 96.21% |

This matters because direct and previous operations can share the same visual
ordinal cue while requiring different answers. A cue-only shortcut therefore
cannot pass the operation × cue cells or the conflict audit.

## Ultra-gradual curriculum

Only one causal burden changed at each rung:

1. add a neutral operation glyph without changing the task;
2. learn the previous-item atom at span two;
3. use the first valid anchor once at span three;
4. use it after one prior query;
5. use it after two prior queries;
6. introduce the second valid anchor once;
7. use the second anchor after one prior query;
8. use it after two prior queries;
9. remove forced positions and audit both anchors in natural query order.

Every training rung rehearsed mastered distributions. Advancement required
95% overall, 95% for each populated subgroup, 95% conflict accuracy, and
retention above the same gate. Near a boundary, updates were reduced to 10–20
rather than allowing a large run to overshoot.

The replica quantifies both seed variance and compounding:

| replica second-anchor rung | target verifier bits | result |
|---|---:|---:|
| isolated one-query operation | 41,472 | 97.53% previous |
| delayed to query two | 23,040 | 98.03% previous |
| delayed to query three | 5,760 | 97.22% previous |

Before any query-three training, the learned query-two checkpoint already
scored 94.92% on the query-three operation. Only 5,760 additional target bits
were needed to cross the strict conflict gate. The later temporal extension
therefore acquired the majority of its capability from earlier rungs.

The primary and replica did not ignite at identical experience counts. That
variance is retained in the record rather than hidden by an average.

## Adversarial controls

The final audits support a causal memory-and-operation interpretation:

| control | primary | replica |
|---|---:|---:|
| all fast memory reset | 49.95% | 49.90% |
| reversal prediction flip on changed cases | 98.21% | 97.88% |
| candidate prediction flip on changed cases | 96.85% | 97.33% |
| operation-counterfactual accuracy | 97.57% | 97.84% |
| operation prediction flip on changed cases | 93.89% | 94.84% |

The valid operation counterfactual rerenders the same logical lifetime with the
other operation glyph and recomputes the verifier-private target. It is replayed
through the recurrent controller; recurrent snapshots are never tensor-swapped.

A matched shuffled-outcome control starts from the pre-second-anchor replica,
uses the same renderer, optimizer, architecture, rehearsal streams, and update
budget, but permutes scalar outcomes. It reaches only 52.57% conflict accuracy
and destroys inherited skills to 77–81%. Correct verified experience is
therefore causally necessary both for acquisition and retention.

## Checkpoints

- `artifacts/checkpoints/unified_procedural_shape_previous_primary_seed37201.pt`
  - SHA-256:
    `2b359ea78595ecca06030c86c46d0306e0149a732cbd3d06b4edff0119f14a95`
- `artifacts/checkpoints/unified_procedural_shape_previous_replica_seed39251.pt`
  - SHA-256:
    `00c87374b2f734e83f2115c0bed2c8411802e9fecd570a34a316f1666a03e152`

## Interpretation and next frontier

This is more than passive short-term recall: the controller reads a stored
sequence and applies a visually selected relative operation to it. It is still
not general working memory. Only one relation (`previous`) at span three has
been demonstrated.

The next minimal operation should be `next item` at the same span, introduced
with the same neutral-glyph, one-anchor, position-by-position curriculum.
Only after both directional atoms compose reliably should the system attempt
two-positions-back, full reversal, longer spans, or dynamic memory allocation.
