# Preregistration: integrated self-model holdout

Status: **reserved, not consumed**. The seed block is registered in
`experiments/brainworkshop_canonical/seed_ledger.py`, but this holdout must not
run until a replacement self-model mechanism passes a new development safety
audit. The current guarded mechanism was rejected in
`session_records/brainworkshop_self_model_adversarial_2026-08-16/`.

## Claim

On unseen navigation worlds, a persistent self model reduces the experience
needed to learn and act in the integrated amodal loop, without increasing
confident identity errors when its remembered dynamics are transplanted,
reversed, poisoned, or observationally ambiguous.

The primary result is behavioral, not an offline relabeling score:

> The self-model arm must improve the integrated learning/action curve against
> a matched no-self arm on every replicate, while satisfying the mismatch and
> abstention gates below.

Offline posterior accuracy is secondary and may not rescue a failed online
behavioral gate.

## Frozen configuration

- Holdout block: `integrated_self_model_holdout`
  (`9_000_017`, `9_500_017`, `10_000_017`).
- Three replicates, with world seeds striding by `37` as in the development
  navigation family.
- Four navigation worlds per replicate; twenty-step episodes; forty
  exploration episodes per world; six trained/held-out relations as in the
  integrated diagnostic.
- Frozen self-model constants: applicability margin `0.25`, controllability
  weight `2.0`, confidence threshold `0.75`.
- Frozen frontend, temporal controller, decomposition-selection procedure,
  probe policy, discount, and action protocol.
- No optimizer updates and no writes to the curated `AgentBrain.bank`.
- No tuning, seed selection, arm removal, or threshold changes after the first
  holdout episode.

The only admissible mechanism change after this preregistration is a new
development mechanism with a new schema and a new diagnostic record. It must
not be silently substituted under the frozen self-model name.

## Arms

Every arm receives the same rendered streams, action budget, verifier outcomes,
controller, decomposition, and relation set.

1. **No-self baseline.** Episodic identity only; no remembered dynamics are
   carried across episodes.
2. **Remembered self.** The frozen guarded likelihood-plus-controllability
   posterior, with online invalidation and post-change-only recovery.
3. **Oracle upper bound.** Identity supplied only for a diagnostic ceiling; it
   is not a deployable arm and cannot satisfy the primary claim.
4. **Fresh learner control.** Same runtime and experience budget, with the
   external self artifact reset at the start of each replicate.
5. **Irrelevant inheritance.** A remembered artifact from an unrelated world;
   it must not beat the no-self baseline or increase confident errors.

The report must retain all arms, including failures. A winning-only report is
invalid.

## Pre-registered gates

Promotion requires all of the following in every replicate:

- positive held-out integrated-return advantage over the matched no-self arm;
- lower orientation cost or higher online identification, not merely a better
  offline relabeling score;
- confidently-wrong identity rate no higher than the no-self arm on honest
  worlds and at most `0.02` on transplanted and changed-dynamics streams;
- exact-mimic and missing-evidence controls abstain rather than tie-break;
- poisoned initialization recovers to confidently-wrong `<= 0.02` within six
  measured update passes, or the arm is rejected;
- dynamics reversal records first detection and first recovery delays, with no
  confident call while quarantined;
- no regression on trained relations, held-out relations, or the two-marker
  control;
- stable-prefix accounting: the first threshold that remains satisfied at
  every later measured prefix, not a single final crossing;
- exact `AgentBrain.bank` digest preservation.

Any failed gate is a rejection of the mechanism for this block. The
architectural lesson may be retained, but weights must be reset or the
mechanism redesigned before another holdout is considered.

## Required controls

The holdout must include valid pixel rerenders for:

- missing evidence and passive markers;
- action-shuffled and reward-shuffled streams;
- corrupted remembered memory;
- unannounced dynamics reversal;
- exact mimic, delayed copy, partial response, stochastic copy, and an
  independently controlled distractor;
- the matched fresh learner and irrelevant-inheritance controls.

No hidden state, coordinates, rule IDs, correct actions, or verifier labels may
reach the controller. Oracle data may be used only for discarded scoring-side
probes.

## Accounting and closure

Each replicate must record unique verifier bits, unique logical lifetimes,
optimizer updates, replayed examples, wall/GPU time, latency, stable
bits-to-threshold, retention on mastered primitives, and transfer ratio against
the fresh learner. The record must include the pre/post bank digest and
checksums for every report and ledger.

This document reserves the block; it does not consume it. The current
development rejection remains the stopping condition until a replacement
mechanism earns a fresh diagnostic record.
