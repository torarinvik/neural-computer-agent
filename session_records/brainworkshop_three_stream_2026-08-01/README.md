# Protected three-stream acquisition — 2026-08-01

This experiment tested the next amodal frontier: add a third token/text
stream to an experienced vision+audio policy while preserving the inherited
controller and the two old streams.

## Setup

- one recurrent controller, loaded from the two-stream parent;
- vision and audio encoders and their RAM/intention paths inherited;
- a replaceable token frontend (`text`) with an identity-preserving opaque
  initialization (one-hot token basis, no match labels or answer mapping);
- a generic per-stream input-confidence gate, initialized as a no-op for the
  new stream;
- only the new stream's encoder, RAM bridge, intention bridge, gate, and
  factorized opaque output rows were trainable;
- reward-only factorized verifier feedback; controller and inherited streams
  received no updates.

The parent was first improved with 64 updates of label-free audio waveform
reconstruction followed by 256 reward-only audio updates. Its held-out
baseline was vision **100.0%**, audio **62.6%**; history reset returned audio
to **56.8%** and time shuffle to **56.2%**.

## Population result

The selected seed `47407` ran a 64-update smell test, then 256 additional
updates. On 4,096 held-out trials the final per-stream results were:

| condition | vision | audio | text | exact text-target action |
|---|---:|---:|---:|---:|
| before adding text | 100.0% | 62.4% | 56.3% | 56.3% |
| after 256 reward updates | **100.0%** | **62.4%** | **94.1%** | **94.1%** |
| history reset | 56.4% | 56.1% | 56.2% | 56.2% |
| temporal shuffle | 55.9% | 57.2% | 59.4% | 59.4% |
| cross-episode text swap | 100.0% | 62.0% | 56.5% | 56.5% |

The promoted continuation used 256 updates × 32 fresh lifetimes = 8,192
training lifetimes (65,536 verifier decisions) and about 51.7 seconds on the
local CPU. The earlier 64-update population race is recorded separately; no
stable bits-to-threshold or fresh-learner transfer claim is made yet.

The cross-episode swap preserves the original targets but replaces each
episode's text stream with another episode's text. The new skill therefore
depends on the text stream's temporal relation rather than a fixed output
bias. Matching inherited tensors were compared against the parent: **zero**
controller, vision-encoder, or audio-encoder tensors changed; only the
expanded decoder shape and new-stream parameters differ.

## What this proves

This is a protected three-stream proof of concept: a frozen central controller
can acquire a new temporal primitive through an external modality adapter
while retaining inherited visual and audio behavior. The input bus now has a
generic cold-start gate, so a new stream can enter without perturbing old
skills and earn influence from verified reward.

## What it does not prove

The selected population winner is not yet a universal claim. A separate seed
(`47405`) stayed near chance after 256 updates, and the neutral-output control
also stalled. This exposes initialization/exploration variance. The next
experiment should compare a small population or learned exploration policy
against a fresh learner, then require independent successful seeds and a
stable bits-to-threshold advantage before promoting the result as compounding
sample efficiency. The token frontend is also not a natural-language encoder;
an unseen speech/text frontend and decoder still require their own causal
qualification.

Artifacts:

- `population_seed47407_continued.json` — selected winner and controls;
- `protected_text_identity_seed47403.json` — exploratory successful run;
- `protected_text_identity_seed47405.json` — independent failed run;
- `protected_text_zerohead_seed47406.json` — neutral-output control;
- `audio_encoder_ssl_seed47401.json` and `audio_stage_ssl_seed47402.json` —
  label-free audio-parent preparation.
