# Shared operator-algebra diagnostic — 2026-08-11

This is a matched, promotion-quality diagnostic of a new external composition
codec. `ExternalSkillFragmentOperatorCombiner` applies one code-conditioned,
low-rank state transition to every opaque fragment segment. It is controller
independent, receives only the rich learner trace, and supports checksummed disk
persistence. The intent was to address the previously identified bottleneck:
one shared learner had not inferred a reusable composition law.

## Evidence

The seed-69316 run used the same parent, four acquired fragments, three training
orders, three held-out orders, 64 parent updates, 256 primitive updates, 128
composition updates, batch size 32, span 3, and audit count 128 as the existing
shared-composition audit.

| metric | result |
| --- | --- |
| shared training accuracy | `0.6849 / 0.7266 / 0.7786` |
| held-out accuracy | `0.6016 / 0.5833 / 0.7083` |
| wrong-order accuracy | `0.6563 / 0.6745 / 0.7214` |
| zero-code accuracy | `0.5391 / 0.5365 / 0.6667` |
| missing-evidence accuracy | `0.4688 / 0.5026 / 0.5000` |
| reward-shuffled accuracy | `0.4427 / 0.4870 / 0.5417` |
| stable shared/fresh bits | none / none |
| unique verifier bits | `449,280` |
| optimizer updates | `2,240` |
| replayed examples | `0` |
| wall time | `256.76 s` |

The frozen parent, acquired-bank checksum, persistence/corruption, zero-code,
missing-evidence, and reward-shuffled gates passed. The capability promotion was
rejected because shared targets did not reach stable mastery, held-out orders
did not generalize, and wrong-order accuracy remained above the rejection
threshold. The operator algebra therefore did not provide a verified gain.

## Decision

Retain the operator-combiner ABI, rich learner-view isolation, and atomic
checksummed persistence as implementation infrastructure. Do not promote the
low-rank transition as learned capability and do not make it the default
composition codec. The result says the bottleneck is more specific than “the
codec needs a reusable transition”: the learner still lacks a reliable causal
binding between ordered segment evidence and the final intention. The next
pressure test should isolate order credit assignment with a smaller curriculum
and an explicit wrong-order contrast before scaling to deeper programs.

Claim boundary: this is not arbitrary program induction, unrestricted memory
growth, compression, or general continual learning.
