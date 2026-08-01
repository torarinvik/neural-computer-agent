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

## Artifacts

- `relation_to_persistent_smell_seed120101.json`: rejected near-pass.
- `relation_to_persistent_replication_seed120102.json`: admitted training run.
- `repertoire_gate0_blind_seed121102.json`: independent one-controller audit.
- `artifacts/checkpoints/unified_repertoire_gate0_candidate_seed120102.pt`:
  promoted Gate-0 checkpoint, SHA-256
  `10eec234bd22656e9b78c7c39d5af6af03c40364a70c331f8ec994b250eb27bf`.
