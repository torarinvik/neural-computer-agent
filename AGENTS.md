# Repository operating rules

## Scientific objective

Optimize verified reusable capability per unique experience. Prefer mechanisms
that make later tasks faster to learn, not merely mechanisms that increase
accuracy on an already trained task.

## Normative architecture

Follow `docs/AMODAL_N_TO_M_ARCHITECTURE.md`. The target boundary is:

`N encoders -> amodal event bus -> one controller/memory -> intention bus -> M decoders`.

Architecture changes must preserve these invariants:

- encoder and decoder counts are variable at runtime and do not resize the
  controller;
- the controller consumes learned event tensors, never raw modality formats;
- the controller emits learned intentions, never device/protocol formats;
- encoders, controller, memory, and decoders have independently versioned and
  replaceable interfaces;
- simultaneous streams remain separately bindable rather than blindly averaged;
- latent meanings emerge from verified experience and are not assigned to
  coordinates or semantic fields by hand;
- adding an adapter must not add a modality-specific reasoning branch.

The current `self.vision -> step(frame)` path is explicitly legacy/current-state
plumbing, not the target interface. Refactor it behavior-preservingly before
claiming amodal support.

## Learner-visible information

The deployed system's frontends may receive the following raw streams. The
controller itself may receive only the resulting standardized learned events,
never the raw modality formats or privileged metadata:

- rendered vision, audio, or text streams;
- its own opaque actions and exact logging propensities;
- its own latent state, working memory, and external memory;
- deterministic scalar verifier outcomes.

Do not expose game state, coordinates, velocities, semantic task/rule IDs,
correct actions, unattempted-action labels, English reasoning traces, or a
hand-written symbolic solver.

Private metadata may be used by discarded diagnostic probes. Probe weights
must never enter the deployed agent.

## Experiment ladder

1. Start with a sub-minute run.
2. Promote to roughly three minutes only after a mechanistic or causal signal.
3. Promote to roughly ten minutes only after replication.
4. When a run fails, first test whether the curriculum jump was too large.
5. Change one difficulty axis at a time.

Never scale a configuration merely because its reward curve is noisy.

## Required accounting

Record separately:

- unique verifier bits;
- unique logical lifetimes;
- optimizer updates;
- replayed examples;
- wall/GPU time;
- latency;
- stable bits-to-threshold;
- retention on mastered primitives;
- transfer ratio against a fresh learner.

An isolated threshold crossing is not mastery. Use the first threshold that
remains satisfied at every later measured prefix.

## Required controls

Use valid pixel-level rerenders rather than hidden-state swaps. Include relevant
fresh, passive, action-shuffled, reward-shuffled, missing-evidence, memory
corruption, and reversal controls.

Retain inherited weights only if they improve the next held-out learning curve.
If weights hurt but the architecture helps, retain the blueprint and reset the
weights.

## Repository hygiene

- Keep generated caches and disposable checkpoints out of Git.
- Curate milestone checkpoints under `artifacts/checkpoints/`.
- Add checksums to `artifacts/manifests/curated_checkpoints.sha256`.
- Update the sample-efficiency ledger after every promoted or decisively
  rejected experiment.
