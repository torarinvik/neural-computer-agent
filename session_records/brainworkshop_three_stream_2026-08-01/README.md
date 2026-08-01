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
The three-stream baseline is now reproducible; a learned stopping rule remains
open.

## Compounding temporal-memory frontier

The next difficulty was a genuine 2-back relation. The original one-snapshot
bridge could not learn it, so the runtime gained a generic depth-2 RAM bridge:
current, prior-1, current×prior-1, prior-2, and current×prior-2. The first
three blocks are unchanged, and checkpoint migration copies the trained
1-back bridge into those leading blocks. The controller itself remained
frozen; no n-back label or position was passed to it.

The evaluator now gates n-back runs on **eligible trials only** (trials after
the forced warm-up prefix), preventing a fixed no-match prefix from inflating
accuracy. At the matched 256-update budget, the inherited 1-back skill produced
a large sample-efficiency gain:

| seed | inherited 1-back → 2-back | fresh 2-back | inherited reset | inherited cross-stream swap |
|---|---:|---:|---:|---:|
| 47405 | **83.8%** | 50.6% | 49.9% | 50.2% |
| 47408 | **84.3%** | 50.4% | 49.8% | 50.0% |
| 47409 | **87.6%** | 50.0% | 49.9% | 50.0% |

The inherited mean is **85.2%** versus **50.3%** fresh: a +34.9-point gain
from reusing the learned 1-back temporal primitive, with all three inherited
runs passing causal reset and cross-stream controls. A longer inherited run
reached **92.0% eligible** at 384 updates; its reset, temporal-shuffle, and
cross-stream controls were 49.9%, 54.6%, and 50.8%. A disposable supervised
diagnostic reached 93.5% at the same depth, confirming that the RAM interface
is sufficient and that the remaining gap is reward-only credit assignment.
This is the first direct evidence that stored temporal skill makes a harder
primitive learn faster, rather than merely making the final task solvable.

## Protected 3-back ladder

The next rung added a depth-3 generic RAM bridge and trained 3-back from the
promoted 2-back checkpoints. A single 2-back rehearsal stream preserved 2-back
but allowed 1-back to fall to 84.56% mean, so the runtime was extended with a
comma-separated verifier-only rehearsal list. Each update now mixes the new
3-back loss with independent 1-back and 2-back replay losses; no task label,
n-back value, or correct action is passed to the controller.

Across three independent seeds at 256 updates:

| seed | 3-back after | 1-back retained | 2-back retained | reset/cross controls |
|---|---:|---:|---:|---:|
| 47405 | 85.55% | 91.04% | 89.40% | ~50% |
| 47408 | 85.74% | 90.90% | 89.10% | ~50% |
| 47409 | 86.23% | 91.20% | 89.58% | ~50% |
| **mean** | **85.84%** | **91.05%** | **89.36%** | **~50%** |

The inherited 3-back runs began at 48.4–48.6% eligible accuracy, so the
improvement is causal rather than a decoder prior. A matched 1-back parent
without 2-back inheritance also reached 85.94% at 256 updates. That control is
important: the protected ladder is proven, but a 3-back sample-efficiency gain
over a 1-back parent is not yet proven. The next measurement is a
bits-to-threshold race from equal 1-back starting checkpoints, not a longer
fixed-budget run.

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
The next experiment should measure a learned stop/continue decision and compare
its verifier bits-to-threshold against the fixed 256-update compounding
baseline. The next capability rung is 3-back with the same gradual RAM
expansion. An unseen speech/text frontend and decoder still require their own
causal qualification.

Artifacts:

- `population_seed47407_continued.json` — selected winner and controls;
- `protected_text_identity_seed47403.json` — exploratory successful run;
- `protected_text_identity_seed47405.json` — independent failed run;
- `protected_text_zerohead_seed47406.json` — neutral-output control;
- `neutral_entropy_seed47405.json` and `neutral_entropy_seed47405_plus128.json`;
- `neutral_entropy_seed47408.json` and `neutral_entropy_seed47408_plus128.json`;
- `neutral_entropy_seed47409.json` and `neutral_entropy_seed47409_plus128.json` —
  the three independent exploration-robust continuations;
- `nback2_inherited_depth2_direct256_seed47405.json`,
  `nback2_inherited_depth2_direct256_seed47408.json`, and
  `nback2_inherited_depth2_direct256_seed47409.json` — inherited 2-back
  acquisition;
- `nback2_fresh_depth2_direct256_seed47405.json`,
  `nback2_fresh_depth2_direct256_seed47408.json`, and
  `nback2_fresh_depth2_direct256_seed47409.json` — matched fresh controls;
- `nback2_inherited_depth2b_seed47405_384_audit.json` and
  `nback2_supervised_depth2_seed47405_256.json` — mastery and architecture
  diagnostics;
- `nback3_rehearsal1_2_depth3_seed47405_256.json`,
  `nback3_rehearsal1_2_depth3_seed47408_256.json`, and
  `nback3_rehearsal1_2_depth3_seed47409_256.json` — protected 3-back runs with
  simultaneous 1-back and 2-back rehearsal;
- `retention_nback1_after_multirehearsal_seed47405.json`,
  `retention_nback1_after_multirehearsal_seed47408.json`, and
  `retention_nback1_after_multirehearsal_seed47409.json`, plus matching
  `retention_nback2_after_multirehearsal_*.json` files — full-ladder audits;
- `nback3_fresh_depth3_seed47405_256.json` — matched 1-back-parent 3-back
  control;
- `audio_encoder_ssl_seed47401.json` and `audio_stage_ssl_seed47402.json` —
  label-free audio-parent preparation.
