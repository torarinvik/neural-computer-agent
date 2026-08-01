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

## Exploration-robust continuation

The first population exposed a real reliability problem: independent seed
`47405` stayed at 55.4% after 256 updates, while the selected seed succeeded.
An entropy coefficient alone did not fix that seed (55.4%). The successful
low-cost intervention was generic and label-free: initialize the new opaque
output bit at a neutral 50/50 distribution (`--neutral-new-output`) and keep
the factorized policy exploratory with `--entropy-coef 0.1`. No verifier label,
task name, or answer mapping is introduced.

Three independent seeds then reproduced the same staged learning curve. After
256 updates they were still in the expected exploration valley (text
58.3–72.9%); after only 128 continuation updates all crossed 90%:

| seed | text after 256 | text after 384 | history reset | cross-episode text swap |
|---|---:|---:|---:|---:|
| 47405 | 60.4% | **94.1%** | 56.3% | 56.2% |
| 47408 | 58.3% | **94.3%** | 56.3% | 56.2% |
| 47409 | 72.9% | **93.9%** | 56.2% | 56.5% |

Vision stayed at 100% and audio at 62.4% in every continuation. Each run used
384 updates × 32 fresh lifetimes = 12,288 training lifetimes (98,304 verifier
decisions), and the inherited controller, vision encoder, and audio encoder
were unchanged. The reset and cross-episode controls returning to chance are
the causal checks that the improvement is a learned temporal text skill rather
than a decoder prior. The 384-update recipe is now the reproducible protected
three-stream baseline; a learned stopping rule remains open.

## What this proves

This is a protected three-stream proof of concept: a frozen central controller
can acquire a new temporal primitive through an external modality adapter
while retaining inherited visual and audio behavior. The input bus now has a
generic cold-start gate, and neutral output plus entropy-preserving exploration
makes that acquisition reproducible across independent seeds. A new stream can
enter without perturbing old skills and earn influence from verified reward.

## What it does not prove

The robust result is still a protected benchmark claim, not unrestricted
N-stream or natural-language transfer. It does not yet prove that the agent
discovers its own stopping point or that the text frontend understands language;
the token frontend is an opaque protocol encoder. The original winner used a
different seed and 256-update continuation, so the neutral/entropy recipe is
reported separately rather than silently replacing that history. The next
experiment should measure a learned stop/continue decision and compare its
verifier bits-to-threshold against this fixed 384-update baseline. An unseen
speech/text frontend and decoder still require their own causal qualification.

Artifacts:

- `population_seed47407_continued.json` — selected winner and controls;
- `protected_text_identity_seed47403.json` — exploratory successful run;
- `protected_text_identity_seed47405.json` — independent failed run;
- `protected_text_zerohead_seed47406.json` — neutral-output control;
- `neutral_entropy_seed47405.json` and `neutral_entropy_seed47405_plus128.json`;
- `neutral_entropy_seed47408.json` and `neutral_entropy_seed47408_plus128.json`;
- `neutral_entropy_seed47409.json` and `neutral_entropy_seed47409_plus128.json` —
  the three independent exploration-robust continuations;
- `audio_encoder_ssl_seed47401.json` and `audio_stage_ssl_seed47402.json` —
  label-free audio-parent preparation.
