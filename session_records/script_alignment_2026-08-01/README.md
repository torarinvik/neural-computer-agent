# Script-alignment consolidation

## Why this run was necessary

The video-script architecture describes one controller that grows without
discarding earlier skills.  Before adding another task, we audited individual
checkpoints as whole repertoires.  The strongest procedural-span checkpoint
passed span three and span five, but failed binary binding, four-rule binding,
persistent recall, and pair relation.  The robust relation checkpoint passed
binary, four-rule, and relation but failed persistent recall; the persistent
checkpoint passed binary, four-rule, and persistent but failed relation.

This establishes an important accounting correction: successful descendants
in the repository are not automatically one continually growing agent.

## Sub-minute consolidation result

Starting from `unified_pair_relation_robust_three_appearance_seed9672.pt`, the
persistent recall objective was rehearsed together with binary mapping,
four-rule composition, and all three relation appearances.  The learner still
saw only RGB, its own attempted opaque action, scalar outcomes, latent active
state, and content-addressed latent reads.

The 40-update smell test took 5.34 seconds.  It reached 86.0% persistent recall
and retained every old gate, but was correctly rejected because its reversal
flip rate was 77.4% versus the 80% requirement.

The promoted 80-update replication took 7.14 seconds and passed.  A separate
2,048-lifetime blind audit on the saved immutable checkpoint found:

| Capability | Blind result |
|---|---:|
| binary few-shot binding | 94.6% normal; 94.4% reversed |
| four-rule composition | 99.0% normal; 98.6% reversed |
| relation, bars | 99.8% post-feedback |
| relation, diamonds | 100.0% post-feedback |
| relation, disconnected dots | 99.2% post-feedback |
| persistent disk recall | 94.1% normal; 94.2% reversed |
| persistent paired prediction flips | 92.2% |
| persistent memory removed | 49.1% |
| persistent memory shuffled | 49.1% |
| persistent memory corrupted | 48.7% |
| persistent retrieval top-1 | 95.0% |

All four requested capability gates passed in one checkpoint.  This is Gate 0,
not the final architecture: procedural span skills are not yet consolidated
into this lineage, and frozen-weight acquisition of a genuinely adjacent rule
through disk rows alone remains the next scientific gate.

## Five-capability span-two consolidation

Span two was then introduced at the deliberately nonzero nuisance floor.  Each
optimizer update aggregated fresh span experience with binary, four-rule,
relation, and persistent-memory rehearsal.  The 20-update smell test remained
at chance but showed a falling target loss.  A 60-update arm reached 58.0%.
The final sub-minute arm reached 77.8%, justifying a bounded 300-update run.

That run initially appeared to pass at 90.4%, but its blind presentation-
reversal flip rate was only 67.2%.  We tightened both the trainer and the
whole-controller audit before promotion.  A 120-update continuation then
passed every causal gate.  On a fresh 4,096-lifetime blind audit, the same
immutable checkpoint reached:

| Capability/control | Blind result |
|---|---:|
| span-two accuracy | 98.06% |
| span-two reverse-presentation accuracy | 97.96% |
| span-two presentation-reversal flips | 92.33% |
| span-two candidate-counterfactual flips | 96.07% |
| blank presentation | 50.26% |
| all active memory reset | 49.89% |
| binary few-shot binding | 98.74% |
| four-rule composition | 99.82% |
| relation, three appearances | 98.10–99.99% |
| persistent recall | 98.73% |
| persistent memory removed | 49.66% |

This is a genuine five-capability repertoire and a successful retention result.
It is **not** yet a compounding sample-efficiency result.  The admitted lineage
used 420 span updates, or 215,040 unique span verifier bits, versus 16,384–
20,480 stable bits in the earlier specialist lineage.  Complete rehearsal
prevented forgetting but diluted acquisition by roughly an order of magnitude.
The next target is therefore reducing bits-to-threshold on this same span-two
bridge—not increasing span length.

## Frozen-weight relation-memory milestone

The next gate was then tested without changing the five-capability checkpoint.
The visual same/different controller and its decoder stayed frozen.  A generic
external memory received only controller-produced event keys, attempted opaque
actions, and scalar verifier outcomes.  It stored successful action intentions
and retrieved them through a real disk save/reload.  Three independent
1,024-context runs passed the private context-to-action remapping audit:

| seed | disk | reversal | prediction flips | no memory | shuffled | corrupted |
|---:|---:|---:|---:|---:|---:|---:|
| 19001 | 98.14% | 98.14% | 100% | 49.61% | 48.14% | 51.95% |
| 19002 | 98.14% | 98.14% | 100% | 49.80% | 49.51% | 48.24% |
| 19003 | 98.34% | 98.34% | 100% | 49.22% | 52.83% | 48.63% |

The controller state digest was identical before and after every run.  This
closes Gate 1 for frozen-weight acquisition through a non-parametric episodic
action-memory baseline.  It is deliberately narrower than the final amodal
memory claim: the next experiment must move the same behavior into the native
latent `retrieved_memory` path and compare its verifier-bit cost against a
reset learner.

## Gradual-prerequisite and efficiency fork

The expensive span-two result was traced through progressively cheaper tests.
The Gate-0 controller first received the procedural renderer's smallest atom:
identify the currently visible shape.  It reached 100% held-out accuracy after
20 updates / 5,120 verifier bits while retaining the four-capability repertoire.

At the same 5,120-bit budget, direct span-one training from Gate 0 remained at
50.1%, while the procedural-identity parent reached 63.2%.  Extending the same
arm to 15,360 bits produced 99.1% span-one accuracy.  A separate 4,096-lifetime
whole-controller audit passed binary, four-rule, relation, persistent recall,
and span one in the same checkpoint: span-one accuracy 98.88%, candidate flips
97.78%, blank presentation 50.0%, and complete memory reset 50.05%.

This is positive transfer from an adjacent perceptual primitive, but the next
composition remains incomplete.  At 20,480 bits, span two reached 71.7–73.9%
from the span-one parent, versus only 58.0% for the direct consolidated lineage
after 30,720 bits.  The inherited curve reached 78.8% at 61,440 bits but
plateaued below mastery.  Rehearsing span one preserved the prerequisite but
gave only a small additional gain.  Batch 64 supplied four times as many
optimizer steps per bit and reached 78.1% at 16,384 bits, still below the
original compact specialist's curve.

Two tempting fixes were rejected:

- assigning 80% of aggregate loss weight to span two improved the target only
  slightly and broke binary retention;
- target-only gradient projection against aggregate repertoire gradients also
  broke binary retention and did not improve the target curve.

The architectural comparison localized the remaining cost: the consolidated
agent is width 96 with eight workspace slots and several generic adapters,
whereas the efficient specialist is width 64 with four slots.  The compact
span-two checkpoint passes span two but none of the four older repertoire
gates.  The next high-ROI fork is therefore compact-controller consolidation:
teach the old repertoire to the efficient 64-wide span parent under strict
span retention, then compare later acquisition curves against the 96-wide
lineage.  More updates on the current plateau are not justified.

## Artifacts

- `relation_to_persistent_smell_seed120101.json`: rejected near-pass.
- `relation_to_persistent_replication_seed120102.json`: admitted training run.
- `repertoire_gate0_blind_seed121102.json`: independent one-controller audit.
- `artifacts/checkpoints/unified_repertoire_gate0_candidate_seed120102.pt`:
  promoted Gate-0 checkpoint, SHA-256
  `10eec234bd22656e9b78c7c39d5af6af03c40364a70c331f8ec994b250eb27bf`.
- `repertoire_span2_strict_blind_seed124005.json`: independent five-capability
  audit.
- `artifacts/checkpoints/unified_repertoire_span2_strict_seed122005.pt`:
  promoted five-capability checkpoint, SHA-256
  `663fe4f7e7c137adf038b140a44db7a82076761d73fc68b61b651bff67c1109a`.
- `repertoire_span1_blind_seed126005.json`: blind gradual-prerequisite audit.
- `artifacts/checkpoints/unified_repertoire_span1_seed125005.pt`: promoted
  five-capability span-one parent, SHA-256
  `81f453c7c2a9d35ba2193062fc38b6ec97c6e0c8bf476aa385d941378a197dc1`.
