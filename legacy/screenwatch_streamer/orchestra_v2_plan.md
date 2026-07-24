# Orchestra v2 — symbolic tracks, audio, model auditions, evidence ledger

Status: **M1–M7 + M9 COMPLETE** (M8 attempted, blocked on gated-repo auth). All milestones delivered
or delivered-to-the-external-wall. Successor to `cursor_vlm_plan.md` (COMPLETE).

**Progress:**
- **M1 ✓** `tracker.elisa` (commit 5eba2dc) — model-free object tracker: grid reconstruction →
  foreground connected components → persistent tracks (ACTIVE/LOST/ENDED) → I9-structural OBS/INF
  records. Reproduces the by-hand centroid analysis on motion/motion-trap; deterministic.
- **M2 ✓** trap suite (commit c9d285b) — 4 scenegen scenes + `track_probe.py`/`track_test.sh` +
  occlusion reasoning. **motion 100% (VLM 50%), motion-trap 100% (VLM 25%)**, crossing-swap 100%,
  occlude-vanish 100% — all deterministic, 0% confab. Measured limits recorded (id-through-occlusion,
  merge, textured-bg → need appearance model / SHIFT path — deferred).
- **M3 ✓** integration — SPEC I9 + viola member + tracker CLI contract + evidence-family rule + `track`
  cursor verb; watcher_protocol v3.2 (routing: motion → `track` first). Bass `TRACKS` summary line
  deferred (measure encoder-loop cost first — do not slow capture).
- **M4 ✓ (cymbal)** `audiotriage.elisa` (commit 6448c19) — deterministic audio triage, 512-pt radix-2
  FFT + RMS/ZCR; TRANSIENT/SILENCE/TONE/LEVEL_SHIFT. **M4 capture** `audiocap.swift` (commit, compiles;
  SCStream system audio → PCM ring) — LIVE-VERIFICATION PENDING (needs Screen-Recording + live audio,
  the co-run gate). Direct-PCM path is the CI gate and is green.
- **M5 ✓** `audiogen.elisa` (commit 0b2d93a, WAV synth) + 5 audio trap scenes + `audio_probe.py`/
  `audio_test.sh` — cymbal scores **100% / 0% on all 5** (transient ordering, silence timing, single
  level-shift, tone count + 1000 Hz, av-sync within one 16 ms hop of 10.000 s). Deterministic.
- **M7 ✓** Qwen3.5-2B audition — better honest-motion captioner (75% vs 2.5's 50%) but **fails all
  motion traps (0%)**: prior-fill is architecture-independent within the Qwen family. Decision (pre-
  registered): NOT the motion authority; I8 unchanged; viola owns motion. Harness worked unchanged.
- **M6 (sax) ✓ — unblocked + auditioned.** Root cause: MiDashengLM targets transformers **4.57**; the
  5.13.1 pin (Qwen2.5-VL's) made `generate` emit token 0. Fix: own venv `.venv-aud` (transformers 4.57,
  `setup_aud.sh` + `screenaud` shim). On 4.57 it produces real captions (MPS, ~5 s load / ~1 s infer).
  **Measured policy:** fluent but **untrusted on synthetic OOD audio and it confabulates** (beeps →
  "a person speaking"; impacts → "a cat"; absent-alarm probe → **"yes"**). ⇒ Inferred-only, split by
  claim type (like I8); cymbal owns timing/count; positive trust policy needs a real-audio (live
  `audiocap`) audition.
- **M8 (Marlin) — resolved: literal checkpoint gated (user step), scientific question ANSWERED via
  proxy.** The literal `NemoStation/Marlin-2B` is a GATED repo (401) — running it needs the user to
  accept its terms on HF + provide an HF token (a user-only action; no credentials exist on this
  machine, and the assistant may not create accounts/accept terms). A **committed ready-to-run harness**
  (`eval/marlin_audition.py`, torchcodec + mp4-from-fixture wired) executes the moment access is granted.
  Its purpose — *does a video-native small VLM in Marlin's family break the motion prior-fill?* — is
  answered **NO** by the non-gated proxy **Qwen3-VL-2B** (25%/75% on both scenes). Four Qwen video
  models now all prior-fill the traps; Marlin (a Qwen3.5-2B fine-tune, coarse 2 FPS training) inherits
  it → coarse captioner only, never the motion authority. The verdict holds without the gated download.
- **M9 (representation ladder) ✓ — A/B/C bake-off run in BOTH regimes, verdict reached.** `eval/
  ledger.py` builds the typed evidence ledger (member/family/OBS-INF/conf/evidence-pin per I7/I8/I9) +
  deterministic story.md projection. Two live 3-arm runs (`eval/bakeoff.md`): (1) single-episode
  motion-trap with a VLM prior-fill — all 75%/25%, all resisted the prior, C wasted 55% more bytes;
  (2) 20-event long stream, ≤280-char cap (4× compression), 5 retention probes on early facts — all
  **100% retention / 0% confab**, cost **B(214) < C(268) < A(271)**. **Verdict: stop at rung B** — the
  typed ledger matches prose's faithfulness in fewer bytes and carries provenance natively; C/D aren't
  justified by any measured probe. Cross-cutting: **provenance (OBS-vs-INF, symbolic-beats-neural), not
  representation format, is what defeats the prior-fill.** `story.md` = projection of a rung-B ledger.
  Remaining scale-up (only if dogfooding demands it): entity-retrieval probes to test C; relational
  queries to test D.

## Why this plan

The M3 trap tests proved the central lesson: generative models propose semantics, but
exact motion/timing/state-transition claims must come from symbolic measurement, and
*every* member — neural or symbolic — earns trust per claim type through trap tests
before its claims graduate to knowledge. This plan applies that discipline to the four
agreed gaps, in dependency order:

1. **Persistent object identity** (the viola) — the delta stream knows pixels changed;
   nothing knows *what* moved, whether it's the same thing, or whether it vanished vs
   got occluded. This is the highest-leverage gap: contact events, occlusion reasoning,
   and audio↔visual correlation all key off track identities.
2. **Audio** (cymbal + sax) — impact transients, warning tones, music-state changes,
   off-screen activity. Currently invisible to the whole orchestra.
3. **Violin auditions** — Qwen3.5-2B (early-fusion, may break the Qwen2.5 prior-fill
   inheritance) and conditionally Marlin-2B (coarse captioner only).
4. **Representation ladder** — story.md → typed ledger → identities → graph, decided
   by probes, not by argument.

Governing principle (now shared across the whole design):

> High-rate capture ensures evidence exists. Symbolic analysis extracts exact
> measurable changes. Generative models propose semantics. The trust system decides
> which claims graduate from hypothesis to knowledge.

Non-goals (explicitly declined): multi-stream serving, Jetson/L4 deployment tiers,
continuous always-on neural models, pose/segmentation members, benchmark batteries
(Video-MME etc.) — we score against our own trap suites only. ASR stays parked.

---

## M1 — The viola: symbolic tracker over the delta stream

A new **offline, deterministic, dependency-free** member. Elisa program `tracker.elisa`
(binary `tracker`), consuming what already exists — no new capture path.

### Input

Primary input: the per-frame **changed-cell grids** the encoder already computes
(dw×dh ASCII grids, `stage_delta` ROWS blocks). Two consumption modes:

- `tracker batch <batch.txt>` — parse FRAME/SAME/SHIFT/ROWS blocks from a batch file.
- `tracker arch <ring_dir> <seq0> <seq1>` — decode Tier-A frames via the same code
  path as `arch_tool show`, diff consecutive frames at tile granularity (reuse the
  encoder's tile-hash diff), then proceed identically. This mode gives exact-pixel
  centroids; batch mode gives grid-resolution centroids mapped through the existing
  SPEC grid↔pixel mapping. Start with batch mode (cheaper, zero decode); arch mode is
  the refinement pass the boss can request.

### Core pipeline (per frame)

1. **Connected components** over changed cells (4-connectivity, grid coords).
   Emit per-component: bbox, area (cells), centroid, mean-change magnitude.
2. **Global-shift check first**: if the frame's SHIFT line (already produced by the
   encoder) explains ≥ SHIFT_DOMINANCE (0.9) of changed cells, classify the frame as
   CAMERA/SCROLL motion and do *not* advance object tracks against it — offset track
   predictions by the shift instead. This is the scroll-vs-motion discriminator.
3. **Association**: greedy nearest-centroid matching with gates:
   - distance gate: |predicted − observed| ≤ max(2·recent_velocity, MIN_GATE cells)
   - size gate: area ratio within [0.5, 2.0]
   - prediction = last centroid + last velocity (constant-velocity model, no Kalman —
     zero-overhead first, upgrade only if traps demand it)
4. **Track states**: `ACTIVE` → `LOST(n)` (unmatched, keep predicting, n ≤ LOST_MAX=12
   frames) → `ENDED`. A component matching a LOST track's prediction becomes
   `REACQUIRED` — and this is **INFERRED**, never OBSERVED (see output contract).
5. **Event derivation** (per track, per frame): `APPEAR`, `VANISH` (ENDED with no
   nearby occluder candidate), `OCCLUDED?` (ENDED while overlapping a static
   unchanged region — inference), `REVERSE` (velocity sign flip sustained ≥3 frames,
   with hysteresis — reuse the Wolf3D axis-hysteresis lesson), `MERGE` (two tracks'
   bboxes overlap then one component), `SPLIT`, `CONTACT?` (bbox overlap ≥1 frame —
   inference: overlap ≠ contact).

### Output contract — OBSERVED vs INFERRED is structural, not stylistic

One line per record, same evidence grammar as everything else:

```
OBS   comp   seq=412 t=17233 track=- bbox=12,4,3,3 centroid=13.4,5.1 cells=7
OBS   shift  seq=413 t=17274 dx=0 dy=-2 coverage=0.96
INF   assoc  seq=413 track=17 comp_centroid=15.1,5.0 conf=0.91 gate=dist:1.7/4.0
INF   event  track=17 kind=REVERSE t=18412 seq=441 conf=0.88 [arch seq 438..444]
INF   event  track=17 kind=VANISH  t=6011  seq=147 conf=0.72 alt=OCCLUDED:0.20
```

Rules (these become SPEC invariant **I9**):
- `OBS` records state only what a deterministic computation measured: a component
  existed, with these cells, at this seq. No identity, no cross-frame claims.
- Every cross-frame claim (`assoc`, all `event` kinds, REACQUIRED) is `INF` and
  carries a confidence and, where ambiguous, the named alternative (`alt=`).
- A track may report `identity=UNKNOWN` — laundering an uncertain association into
  a confident one is the tracker equivalent of VLM prior-fill and is a bug.
- Confidences are *placeholders until M2 calibrates them* — the trap suite measures
  what conf thresholds actually mean and writes the numbers back into SPEC.

### Elisa implementation notes

- New file `tracker.elisa` + shared parsing pulled from `encoder.elisa`/`arch_tool.elisa`
  where practical (batch parsing may be worth factoring into a small shared module —
  Elisa modules per [[elisa-modules-namespaces]]).
- Tracks live in a `darray` of track structs in one region; per-frame components in a
  per-frame scratch region. This is a good dogfood case for region inference on a
  loop-carried builder — watch for the [[region-byvalue-builder-uaf]] pattern.
- Use `machine over` for the track state machine (ACTIVE/LOST/ENDED) — dogfoods
  docs/123/125 on a real consumer.
- Deterministic: no time, no randomness; same input ⇒ byte-identical output (this is
  the M2 repro gate).

### Gate M1

- `tracker batch` runs over an existing motion fixture batch and emits well-formed
  records; deterministic across two runs (byte-identical).
- Component + centroid outputs hand-checked against the known scenegen geometry on
  `motion` (centroid path 69→569→119 within grid quantization).

---## M2 — Tracker trap suite (trust is an eval output — for symbolic members too)

Extend `scenegen.elisa` with scenes whose *ground truth is exact by construction*.
All scenes 640×360, 20s, deterministic, rendered through the real encoder like
motion/motion-trap. Scene ids continue from 3.

| id | scene | geometry (exact) | what it traps |
|----|-------|------------------|---------------|
| 4 | `crossing-swap` | two squares, A: x=40→580 at y=120; B: x=580→40 at y=132 (12px vertical offset so they pass, bboxes overlap mid-crossing t≈9.5–10.5s); same size, different fill values | identity swap at crossing: does track 17 exit still attached to A? Truth records which exits where; correct answer may be `identity=UNKNOWN` after crossing — that scores *better* than a confident wrong swap |
| 5 | `occlude-vs-vanish` | static 80×80 block at x=300; square A moves right, passes *behind* block (drawn under it) t=8–11s, re-emerges; square B (top lane) moves right and genuinely vanishes at t=8s, never returns | OCCLUDED? vs VANISH discrimination + REACQUIRED correctness: A must be reacquired (INF, with occlusion alt), B must end as VANISH |
| 6 | `scroll-vs-motion` | whole scene content shifts left 2px/frame t=5–10s (redraw a striped background offset) while one square independently moves right | SHIFT-dominance path: object velocity must be reported in *scene* coords, not contaminated by global motion |
| 7 | `contact-merge` | two squares approach; run A: stop 8px apart (near-miss); run B (scene 8): overlap for 500ms then separate | CONTACT? inference must fire on B, not A; overlap duration measured |
| 3 | `motion-trap` (existing) | — | the VLM-failed probes, answered symbolically: pt_vanish, pt_reverse, pt_continuous |
| 2 | `motion` (existing) | — | pm_dir, pm_bounce, pm_pos symbolically |

Also two **stress dimensions** applied to scenes 4 and 5 as variants (env knobs, not
new scene ids): frame-phase offset (start rendering at t=+21ms) and encoder grid
coarseness — the tracker must be robust to quantization phase, and if it isn't, the
trap suite is where we find out.

### Harness

- `eval/track_test.sh <scene>` mirroring `trap_test.sh`: render fixture → run
  `tracker batch` → `eval/track_probe.py` converts tracker records into probe answers
  (pure Python, no model) → score with the existing `score_memory.py` matchers.
- Probes are the *same probe format* as the VLM ones so results are directly
  comparable in one table: the deliverable is a scenarios.md section
  "Tracker trap-test (viola)" placing the viola row next to the violin row for
  pt_vanish/pt_reverse/pt_continuous — the punchline of the whole arc.
- **Calibration**: for each INF kind, the suite reports (conf bucket → empirical
  accuracy). Write measured thresholds into SPEC ("assoc conf ≥ .85 ⇒ ≥95% correct
  on suite"; below that, boss must treat identity as UNKNOWN).

### Gate M2

- Scenes 2,3,6,7: all probes correct symbolically (these are the ones centroid
  analysis already solved by hand — the tracker must reproduce that).
- Scene 4: no *confident wrong* identity after crossing (UNKNOWN acceptable).
- Scene 5: A reacquired, B vanished, zero cross-contamination.
- Determinism: two runs byte-identical.
- Measured trust table written into SPEC §"tracker trust policy (I9) — MEASURED".

---## M3 — Viola integration (SPEC, protocol, cursor)

1. **SPEC.md**: add I9 (tracker OBS/INF contract + measured trust table), viola member
   row, `tracker` CLI contract section, and the evidence-family rule (see below).
2. **watcher_protocol.md → v3.2**:
   - New cursor verb **`track`** (symbolic tier, sits beside show/ocr/compare):
     query `type: track`, `span:`, optional `region:`; sub-watcher runs
     `tracker arch <ring> <s0> <s1>`, answers motion predicates from records.
   - Routing rule: **motion-predicate questions go to `track` first**; `describe` is
     never the sole basis for a motion claim (this operationalizes I8's OPEN_QUESTION
     escape: many of those OPEN_QUESTIONs now become answerable).
3. **Evidence-family weighting** (SPEC, one paragraph): witnesses carry a family tag
   (`qwen-visual`, `symbolic-tracker`, `ocr`, `audio`). Agreement within a family that
   is measured-untrusted for the claim type adds ~0 confidence; within a partially
   trusted family, modest; across families, substantial. Marlin and any Qwen-VL are
   the same family.
4. **Bass hookup (cheap, optional in this milestone)**: encoder already writes
   ACTIVITY lines; add a `TRACKS n_active=<k> events=<APPEAR,...>` summary line per
   batch by invoking the component pass inline (or post-hoc via tracker batch —
   decide by measuring encoder-loop cost; do NOT slow the capture path — if inline
   costs >5% of frame budget, keep it post-hoc).

Gate M3: an end-to-end dry-run like M4-of-the-last-plan — a real recorded span, boss
asks a motion question, sub-watcher answers via `track` with evidence pointers that
re-resolve through `arch_tool show`.

---## M4 — Audio capture + the cymbal (deterministic audio detector)

Audio enters the orchestra the same way video did: **lossless capture first, cheap
deterministic triage always-on, neural interpretation only on triggered windows.**

### Capture (`audiocap.swift` → binary `audiocap`)

- ScreenCaptureKit **system-audio** tap (SCStreamConfiguration `capturesAudio` — no
  microphone, no mic permission; screen-recording permission already granted for
  screencap). Mono mixdown, 16 kHz, s16le.
- Ring layout mirrors Tier-A: `aud_<epoch>.idx` + raw PCM chunks
  (`<seq> <t_ms> <n_samples> <off>` index lines), 1-second chunks, same pruning
  discipline, same directory as the video ring (shared session dir, distinct prefix).
- Timestamps from the same clock the video path uses (host time → ms) so
  audio↔video correlation is a subtraction, not a calibration project. **This is the
  make-or-break detail**: gate includes a sync check (see M5 scene `av-sync`).

### The cymbal (`audiotriage.elisa`)

Deterministic, always-on, Elisa-native (dogfood: real DSP in Elisa):
- Per 32ms hop (512 samples): RMS energy (dB), spectral flux via 512-pt real FFT
  (implement in Elisa — it's ~60 lines and a good `-Wperf`/vectorization dogfood),
  zero-crossing rate.
- Detectors (all thresholded, thresholds are constants pending M5 calibration):
  `TRANSIENT` (flux spike over rolling median), `SILENCE_START/END` (RMS below floor
  for ≥300ms), `LEVEL_SHIFT` (rolling RMS band change ≥6dB sustained ≥1s, the
  music-intensity proxy), `TONE` (single-bin dominance ≥500ms — alarms/beeps).
- Output: `AUDIO <kind> t=<ms> [dur=<ms>] [db=<x>] [conf=<c>] [aud seq a0..a1]`
  appended to the same triage stream bass writes, so the singer sees audio events in
  its normal cycle with zero protocol change beyond a new line kind.

### Gate M4

- Record 30s of desktop audio+video (same consent/deletion discipline as the co-run),
  play a known clip; TRANSIENT/SILENCE/LEVEL_SHIFT lines appear at hand-verifiable
  times; PCM ring seq ranges re-resolve to playable WAV via a tiny `aud_tool show`.
- Concurrency: cymbal + audiocap + frame_dump co-run, frame throughput within noise
  of the M4-v1 baseline (4–8 fps regime).

---## M5 — Audio trap scenes (audiogen)

`audiogen.elisa`: deterministic WAV synthesizer (sine, noise bursts, exponential-decay
impacts, amplitude ramps), the audio twin of scenegen. Scenes rendered to WAV, then
*played through the real capture path* where feasible (afplay → system-audio tap) —
and, as the deterministic fallback that CI can run, fed to audiotriage directly as PCM
(both paths scored; the direct path is the repro gate, the played path validates
capture).

| scene | content | traps |
|---|---|---|
| `transient-in-noise` | pink noise bed −30dB, three 10ms impacts at t=4,9,9.15s | detection + **ordering of two rapid events** (150ms apart) |
| `silence-gap` | tone bed with exact 800ms silence at t=7s | silence timing precision |
| `level-ramp` | music-proxy: band-limited noise ramping +9dB over 3s | LEVEL_SHIFT vs many TRANSIENTs (must not machine-gun) |
| `tone-alarm` | 1kHz beep ×3, 200ms each | TONE + count |
| `av-sync` | audiogen impact at t=10.000s + scenegen white-flash frame at t=10.000s, co-recorded | measured a/v skew; gate: |skew| ≤ 50ms, and the measured skew constant is written into SPEC |

Truth/probes in the existing format; `eval/audio_test.sh` mirroring track_test.sh.
Calibrated thresholds written back into audiotriage constants + SPEC.

Gate M5: all probes green on the direct path; av-sync skew measured and recorded.

---## M6 — The sax: MiDashengLM-0.6B audition (triggered audio interpreter)

Same shape as the violin's M1–M3, compressed because the machinery now exists:

1. `setup_aud.sh` — **decision point first**: try the HF-transformers fp32 path on MPS
   inside a `.venv-aud` (mirrors screenvlm; likely easiest), fall back to the GGUF +
   their llama.cpp fork only if transformers is unworkable. Verify the fork claim at
   this point, not before. Model ~0.6B ⇒ trivial RAM either way.
2. `screenaud` / `screenaud.py` — CLI contract mirroring screenvlm exactly:
   `screenaud <aud_ring> <t0_ms> <t1_ms> [--q ...] [--json]` → `ANSWER/EVIDENCE
   [aud seq a0..a1 t=..]/COST`. Pulls PCM from the ring, never from files.
3. **Audio traps for a *generative* listener** — reuse M5 scenes plus two adversarial
   ones: `misleading-context` (calm music + gunshot-like impact: does it report the
   impact or narrate the music?) and `absent-event` (asked "was there an alarm?" over
   a scene with none — the confabulation probe).
4. Measured trust policy → SPEC ("sax trust policy — MEASURED"), split by claim type
   exactly like I8 (plausible outcome: good at *what kind of sound*, untrusted on
   *exact timing/count* — the cymbal owns timing).
5. Protocol: sax is **triggered only** — boss/sub-watcher dispatch on cymbal events or
   describe-shaped audio questions; never always-on.

Gate M6: trap table published in scenarios.md; sax answers a real triggered window in
a dry-run with evidence pointers resolving to playable PCM.

MOSS-Audio-4B is the *escalation* candidate: **deferred** — audition only if the sax's
measured policy leaves goal-relevant questions unanswerable (decide after dogfooding,
not now).

---## M7 — Violin audition: Qwen3.5-2B through the existing harness

1. Pre-flight: check pinned `transformers==5.13.1` supports `qwen3_5` multimodal
   (docs page exists; verify the pinned version, bump the pin in `setup_vlm.sh` only
   if needed, re-run `--verify`). Confirm the chat-template video path matches
   screenvlm's `{"type":"video"}` recipe; adjust `screenvlm.py` per model card if the
   preprocessing contract differs (frames-as-video list should still work — it's the
   same qwen-vl-utils lineage; if not, add a per-model adapter table in screenvlm.py).
2. Run the full existing suite: `MODEL=Qwen/Qwen3.5-2B ./eval/trap_test.sh motion`,
   `motion-trap`, plus the frame-density sweep (8/16/32) on motion-trap, plus one
   text/scene fixture to confirm the reading strength held.
3. Decision rule (pre-registered, so the result can't be argued with after the fact):
   - **Adopt as default violin** if: text/scene ≥ current 3B AND motion-trap
     perception ≥75% with confab ≤25% across the density sweep.
   - **Adopt for text only** (keep I8 motion policy unchanged) if text ≥ current but
     motion still prior-fills.
   - **Decline** if text regresses.
   - Any outcome: scenarios.md gets the comparative table (2.5-3B / 2.5-7B / 3.5-2B),
     and I8's family note is updated ("measured across Qwen2.5-VL and Qwen3.5").

Gate M7: table published; decision recorded with the pre-registered rule cited.

## M8 — Marlin-2B: conditional, coarse-captioner-only audition

Run **only if** M7 showed the Qwen3.5 base improved motion or if a coarse
"what happened in this 2-minute span" captioning verb proves valuable in dogfooding.
Constraints acknowledged up front: `trust_remote_code` custom code, CUDA-oriented,
torchcodec video-file input (mismatch with our ring-frame feeding), 2 FPS / 240-frame
training regime. Audition scope: `caption`/`find` on *long spans* (its actual trained
task), scored against a new long-span scenario — NOT against motion-trap (it is
pre-declared not-a-motion-member; its motion-trap result is collected once for the
family table, expected to fail, and that's fine). MPS port time-boxed to one session;
if custom code fights Metal, decline with reasoning (principled-decline).

---## M9 — Representation ladder (extends the step-6 bake-off)

Pre-registered progressive ladder — each rung must beat the previous on measured
probes to justify its complexity:

- **A. story.md only** (current baseline — already scored by the referee).
- **B. append-only typed event ledger + story.md as projection**: `ledger.jsonl`,
  one record per event: `{t0,t1,kind,summary,family,obs_or_inf,conf,evidence:[...]}`;
  story.md is *regenerated from the ledger* by the singer, never independently
  edited. (Note: log.md + arch pointers is already a proto-B; B formalizes it.)
- **C. B + track/entity identities**: events reference viola track ids and sax/aud
  seq spans; retrieval probes may ask entity-level questions ("the square that
  vanished — where did it reappear?").
- **D. full temporal relation graph** (entities, relations, contradiction tracking,
  expiry, revision). Built ONLY if C measurably fails retrieval/honesty probes that
  D's machinery specifically addresses.

Scoring dimensions (referee): reconstruction, retrieval, confabulation, **and
maintenance cost** — measured as (LOC of representation-maintenance code) + (singer
tokens per cycle spent updating it). A rung that wins probes but doubles cycle cost
loses; that's the zero-overhead principle applied to memory.

Expected stopping point: B or C. D is presumed declined until proven necessary.

---## Sequencing, effort, and dependencies

```
M1 viola core ──► M2 viola traps ──► M3 integration ─┐
                                                      ├─► M9 ladder (needs C's ids from M3)
M4 audio capture+cymbal ──► M5 audio traps ──► M6 sax ┘
M7 Qwen3.5 audition (independent — can run any time, cheapest first result)
M8 Marlin (conditional on M7)
```

Recommended order of execution: **M7 first** (one session, uses only existing
machinery, and its outcome may adjust I8 before anything else lands), then M1→M2→M3
(the core arc, ~3–4 sessions), then M4→M5→M6 (audio arc, ~3 sessions), then M9.
M8 only if triggered.

Per-milestone commit discipline as before: each milestone = one or two commits with
its gate evidence in the message; scenarios.md is the running lab notebook; memory
updated at arc boundaries.

## Risks / decision points (pre-registered)

- **Encoder grid too coarse for small objects** (M1): if dw×dh quantization makes
  crossing-swap centroids ambiguous, fall back to arch-mode pixel diffs for the trap
  suite and record the grid-resolution limit in SPEC.
- **transformers pin vs Qwen3.5** (M7): bump only inside `.venv-vlm`, re-verify 2.5-VL
  still runs after the bump (regression: re-run one motion probe on 3B).
- **MiDashengLM fork requirement** (M6): if the HF path works on MPS, skip the fork
  entirely; if both fail, principled-decline and try the next small ALM.
- **A/V clock skew** (M4/M5): if ScreenCaptureKit audio timestamps disagree with the
  video clock by a variable amount (not a constant), correlation quality degrades —
  the av-sync trap measures this; a variable skew > 50ms is a stop-and-investigate.
- **Cymbal FFT perf in Elisa** (M4): target real-time ×10 headroom on one core; if
  `-Wperf` shows the loop stays scalar, that's a compiler dogfood finding to file,
  with vDSP-via-extern as the escape hatch.
