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
