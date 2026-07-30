# One-event sensory RAM breakthrough — 2026-07-30

## Result

A generic one-event sensory snapshot closes most of the gap between recurrent
and history-free conditional-operation behavior. Every controller step
overwrites this RAM field with the current learned visual latent. A successor
slot may read it on the next step and bind it multiplicatively to the inherited
amodal intention.

The interface carries no cue flag, task ID, operation label, count, correct
action, or verifier-private value. It adds no controller pass and is exactly
behavior preserving for older checkpoints and at zero-output slot insertion.

At a matched 128-update budget:

| 1,024 unique lifetimes / 6,144 outcomes | Seed 25301 | Seed 25311 |
|---|---:|---:|
| sequential conditional operation | 84.75% | 84.33% |
| history-free counterfactual | 78.15% | 73.52% |
| history-free cue prediction flips | 66.95% | 53.58% |
| timing-matched blank cue | 49.95% | 49.75% |
| paired history-free blank cue | exactly 50% | exactly 50% |
| relation retention | 99.21% | 99.10% |
| magnitude retention | 92.33% | 92.30% |
| numerosity retention | 90.85% | 90.38% |

The selected 256-update candidate reaches **84.82% sequential** and **84.58%
history-free** accuracy, with **82.25%** prediction flips when only the public
operation cue is counterfactually reversed. Its blank-cue paired accuracy is
exactly 50%. Relation, magnitude, and numerosity remain at 99.17%, 91.25%, and
90.13%.

This is an unpromoted RAM-interface breakthrough, not 90% mastery.

## Localization

Fresh-state training proved that the architecture could learn an isolated
operation:

- all-independent training: 84.21% history-free;
- the same model in an uninterrupted sequence: 49.83%.

Randomizing the learner-visible previous action did not repair sequential
behavior. A staged independent-then-recurrent curriculum and a 50/50
within-batch mixture produced compromises rather than one robust
representation. This localized the shift to cumulative hidden state.

The one-event snapshot supplies the latest sensory latent independently of
that cumulative state:

```text
cue frame
  -> learned visual event
  -> overwrite latest_event RAM

count frame
  -> new slot reads latest_event
  -> tanh(W × latest_event) × inherited intention
  -> amodal answer intention
  -> replaceable actuator
```

It is a generic temporal primitive: “the immediately preceding sensory event”
rather than “the operation cue.”

## Causal controls

The matched snapshot-content ablation keeps identical architecture, input
width, parameter count, training data, and optimizer budget, but zeros only the
snapshot content for the appended slot:

| Seed 25301, 128 updates | Snapshot active | Snapshot zeroed |
|---|---:|---:|
| sequential accuracy | 84.75% | 72.18% |
| history-free accuracy | 78.15% | 55.54% |
| history-free prediction flips | 66.95% | 15.27% |

The snapshot therefore contributes +12.57 sequential points and +22.61
history-free points beyond its matched-capacity control.

A matched-seed shuffled-verifier run reaches **50.03%** sequential accuracy.
Blanking only the cue pixels returns every truthful run to chance. The
counterfactual renderer keeps the count scene fixed, reverses only the public
operation cue, and complements every verifier answer.

## Experience and compute

The 128-update result uses:

- 1,024 unique logical lifetimes;
- 6,144 attempted-action scalar outcomes;
- 12,288 sensory frames;
- two sensory controller steps per answer;
- no replay of new-task events and no unattempted-action labels.

The selected 256-update candidate uses twice those values. Extending from 128
to 256 updates improved history-free behavior substantially but did not improve
the ~84.8% sequential ceiling, so further duration scaling was stopped.

## Frontier

The one-event snapshot makes conditional operation robust to whether episode
history exists. The next bottleneck is no longer temporal leakage; it is the
remaining perceptual/decision error shared by both modes at roughly 15%.
Promotion still requires at least 90% sequential and history-free accuracy,
counterfactual cue causality, blank-cue chance, and retained prior skills.

The selected local checkpoint is
`artifacts/checkpoints/unified_event_snapshot_operation_candidate_seed25301.pt`.
Its admission status is explicitly unpromoted.
