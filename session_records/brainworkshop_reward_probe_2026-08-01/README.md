# Brain Workshop reward-only probe (2026-08-01)

This is the first learning rung on the clean-room Brain Workshop-style gym.
The public stream is vision only, the controller receives an amodal event, and
the external decoder emits two opaque keypress actions (`no match` and
`position match`). The only learning signal is the verifier's scalar reward;
targets are never passed to the policy.

## What was tested

- A fresh width-32 recurrent controller, 32 updates × 64 unique lifetimes,
  n-back 1, first with the full eight-position visual stream and then with a
  two-position curriculum, MPS.
- A promoted width-64 controller held fixed while only the new encoder, input
  bus, and output adapter learned, 16 updates × 32 lifetimes, MPS.
- Disposable verifier-label ceilings on the same architecture (8 trials and a
  shorter 4-trial curriculum).
- Every run logged gradient norms and used history-reset and time-shuffle
  controls. Reports are the JSON files beside this README.

## Results

The fresh reward-only run had live gradients and rising in-batch reward, but
did not pass the promotion gate on held-out episodes:

| run | held-out accuracy before | after | history reset | time shuffle | decision |
|---|---:|---:|---:|---:|---|
| fresh, 32 updates, 8 positions | 57.50% | 57.50% | 56.45% | 56.03% | reject |
| fresh, 32 updates, 2 positions | 57.50% | 57.50% | 56.45% | 56.03% | reject |
| frozen controller, 16 updates | 43.80% | 40.92% | 42.92% | 42.43% | reject |

The first row is intentionally not interpreted as learning. The balanced
schedule initially allowed a clock policy to exploit its fixed match count, so
the training rung now uses independent seeded match flags (`balanced_matches=
False`). This removes that shortcut while keeping the deterministic replay
property. The remaining negative result is therefore bounded: reward-only
binding was not found in this small budget, and the controller-freeze path did
not improve the task either.

The supervised ceiling also did **not** pass the temporal gate in these small
runs: 57.67% (8 trials) and 67.02% (4 trials) held out, but the corresponding
time-shuffle controls were 56.01% and 66.58%. This means the current signal is
consistent with a timing/class-prior shortcut, not a demonstrated comparison
of the current visual event with the event one step earlier. The result is
useful: increasing reward-only budget before fixing the representation or
curriculum would be low ROI.

## Engineering findings

- Single-stream verifier targets now mask the absent modality; a vision-only
  episode cannot be scored on a hidden audio bit.
- The vision encoder normalizes its input size before pooling, avoiding the
  current MPS adaptive-pooling divisibility failure.
- The history-reset audit exposed and fixed an in-place logging alias that was
  erasing earlier rewards/actions in the control report.

## Next rung

Do not simply scale this failed reward-only run. The next high-ROI rung is a
representation check: train a small self-supervised pixel/event encoder (no
semantic labels), then repeat the short supervised ceiling and a value-baseline
reward probe. A promotion still requires a held-out gain over the majority/fresh
controls and a clear drop under history reset or counterfactual temporal
scrambling.

## Breakthrough: learned visual representation + generic one-step RAM

The next localization rung succeeded twice independently. A visual encoder was
first trained without semantic labels, actions, or verifier targets using only
RGB frame reconstruction. Its throwaway pixel decoder was discarded. The
frozen encoder then fed a generic one-step RAM snapshot (`retrieved_memory`)
and a zero-initialized event×retrieved-memory adapter. The controller learned
only from scalar reward.

| seed | held-out accuracy | history reset | temporal shuffle | decision |
|---|---:|---:|---:|---|
| 44011 | 100.00% | 56.53% | 57.17% | pass |
| 44012 | 100.00% | 56.57% | 57.98% | pass |

Each run used 128 updates × 64 unique lifetimes, a two-position visual n-back-1
curriculum, and 8,192 held-out trials. The 100% result is therefore dependent
on the previous event and its order, not a fixed timing policy. The oracle-event
control reached the same conclusion earlier, while the corrected oracle shuffle
audit prevented a false positive caused by accidentally ignoring shuffled
inputs.

This is a real architecture milestone, but not yet a generalization claim:
the encoder was pretrained on the two-position stream, and the RAM snapshot is
currently a one-step diagnostic interface. The next promotion ladder is
four-position visual n-back, then eight-position visual n-back, then audio and
dual-stream n-back, preserving the same reset/shuffle/reversal gates.

## Four-position promotion

The same method was promoted one rung without changing the controller or
reward definition. A fresh self-supervised encoder was trained on the
four-position stream, then frozen; the controller learned from scalar reward
through the same generic RAM snapshot and retrieved-memory interaction.

| seed | updates | held-out accuracy | history reset | temporal shuffle | decision |
|---|---:|---:|---:|---:|---|
| 44021 | 256 | 92.74% | 56.54% | 58.06% | pass |
| 44022 | 256 | 80.87% | 56.52% | 56.63% | pass |

The matched 128-update run was rejected (56.79% held-out), then the budget was
increased only after the four-position supervised ceiling passed at 75.73% with
the same causal controls. This is the expected gradual-learning pattern: the
primitive is learnable, but its reward-only ignition threshold grows with
difficulty.

## Eight-position visual promotion

The full eight-position visual stream also passed with the same architecture
and scalar reward. A fresh no-label reconstruction encoder was trained for the
eight-position stream and discarded its pixel decoder before controller
training.

| seed | updates | held-out accuracy | history reset | temporal shuffle | decision |
|---|---:|---:|---:|---:|---|
| 44031 | 256 | 66.09% | 56.58% | 57.10% | pass |
| 44032 | 256 | 70.85% | 56.54% | 57.51% | pass |

The result is lower and more variable than four positions, but the causal gap
survives replication. The visual ladder is therefore complete: the same
controller/RAM mechanism learns 1-back comparisons over 2, 4, and 8 visual
locations. Audio and dual-stream composition are now the next untested
frontier.

## Audio-only promotion

The same amodal controller and one-step external-history interface also learn
the comparison from an audio stream. The audio encoder was trained first with
waveform reconstruction only; its throwaway decoder was discarded, and the
frozen encoder then supplied events to reward-only controller training. No
audio identity, match flag, or answer target was exposed to the policy.

| seed | updates | held-out accuracy | history reset | temporal shuffle | decision |
|---|---:|---:|---:|---:|---|
| 44041 | 256 | 67.32% | 57.17% | 56.34% | pass |
| 44042 | 256 | 67.31% | 57.19% | 56.92% | pass |

Both replicas exceed the stochastic majority baseline by about ten points,
while resetting the one-step history or shuffling event order returns accuracy
to roughly 57%. This is the first replicated cross-modality result: the
reward-only learning mechanism is not tied to visual pixels, although the
eight-symbol audio stream is currently harder than the two- and four-position
visual rungs.

The audio reconstruction pretraining also passed its no-label gate: waveform
loss fell from 0.4476 to 0.3443 with live gradients in 32 updates. The
encoder checkpoint and both reward reports are stored beside this README.

The next test is dual-stream composition: simultaneous vision and audio events
must be fused by the same controller, with per-modality causal ablations and
the existing history-reset gate. The architecture should remain unchanged;
only the number of encoder streams and the amodal event bus should grow.

## Frozen-controller external-memory breakthrough

The requested freeze test is now positive. We started from a promoted visual
controller checkpoint, attached the no-label audio encoder, and froze every
recurrent controller weight. The audio encoder was also frozen. Learning was
allowed only in the generic input bus, output decoder, value baseline, and a
new zero-initialized RAM-side relation adapter. On every step that adapter sees
only the current and previous opaque events and writes a residual into the
retrieved-memory snapshot; it receives no stimulus identity, match flag, or
answer label.

| seed | updates | held-out after | before | history reset | temporal shuffle | decision |
|---|---:|---:|---:|---:|---:|---|
| 44137 | 256 | 67.08% | 50.82% | 57.19% | 57.50% | pass |
| 44138 | 256 | 67.07% | 50.81% | 57.18% | 57.10% | pass |

This is the first demonstrated case where the central AI controller remains
frozen while the external memory/computation path acquires a new temporal
skill from scalar reward alone. The gain is causal: removing the previous
snapshot or shuffling event order removes roughly ten percentage points. A
frozen controller with only the new audio encoder, but without the RAM-side
adapter, stayed at 51.20%; allowing the audio encoder to adapt reached 61.46%
but did not meet the causal promotion gate. Those controls localize the gain to
learnable external memory computation rather than a fixed-controller shortcut.

The multimodal ceiling remains the next frontier. A fresh controller reached
56.29% on simultaneous vision+audio after 128 supervised updates, while the
frozen visual controller reached 34.99% with the same dual-stream diagnostic;
the dual-stream relation still needs a more sample-efficient composition path.
