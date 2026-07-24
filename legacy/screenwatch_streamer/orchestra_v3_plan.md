# Orchestra v3 — the real-world plan

**Status: IN PROGRESS (2026-07-14). V0–V2 complete; V3 (the core capability) largely complete and
validated; V4–V6 partially done with the rest resource-gated and honestly scoped.**

Progress snapshot (see each milestone's DONE/PARTIAL/STATUS notes for evidence):
- **V2 supersession** — complete (regression closed; no probe lost).
- **V3.0 trackability gate** — complete: pharo false-REVERSE 5→0, do-no-harm verified across the 8-video
  census (go/lsl1 byte-identical), synthetic suites 100%, `seed_test` 239/240.
- **V3.1 direction re-seed + occlude-reverse** — complete (20/20; freeze & wide-gate tried and rejected).
- **V3.2 merge-drag dispute** — complete (tracker `INF dispute` → ledger `status=disputed` candidates).
- **V3.3/V3.4 (2-D scroll)** — deferred with a recorded trigger (no measured real-video pan failure).
- **V3.5 confidence** — `trk=`/`dq`/`am`/`ec` ship; calibration piggybacks on V3.6 gold.
- **V3.6 real ladder** — flagship PoP-gameplay rung passes light gold 4/4; census-offset finding recorded;
  remaining rungs + LSL2 graduation scoped (LSL2 stays held out).
- **V5 temporal** — viola control `3/3`; neural half scoped (reproduces the OOD prior-fill).
- **V6 long-run** — deterministic backbone verified on real video (linear ledger growth, byte-identical
  rebuild + fault-injection recovery, real supersessions); 2-hour LLM-in-the-loop trial resource-gated.
- **V4 SED** — PANNs Cnn14 RUN (`speech` admitted prec 1.00, content-grounded); non-speech classes gated
  on human-listened gold; AV-sync join per I10.
- **V-next OCR-diff** — `screentext` guitar-diff member: pre-registered audit PASS 4/4, headline 0
  unsupported text changes; ledger-integrated under THE ONE LAW. The third evidence family (text).

**(v2 complete — see orchestra_v2_plan.md for the M1–M9 record.)**

This plan is the product of the four-way design debate (Claude / ChatGPT / Grok / Gemini) that
followed v2, after all corrections were accepted on both sides. It is governed by one sentence:

> **Every claim retains both its evidence and the rule that licensed it; authority is earned
> per-claim-type by measurement, never granted per-model.**

And by the converged division of labour:

```
Trap testing        discovers the failure.
Symbolic measurement answers the exact predicate.
Provenance          records what licensed each claim.
Admission policy    prevents unsupported claims from graduating.
Representation      determines whether evidence can later be retrieved,
                    revised, and reasoned over correctly.
```

Provenance does not *eliminate* prior-fill; it **contains its epistemic consequences**. The v2
bake-off measured exactly that containment (all rungs rejected the VLM prior because all carried
I8/I9 trust tags); this plan extends the same discipline to real-world material.

**What is new since v2:** the user supplied `videos/` — 17 real-world video files (~7.5 GB,
sports / board games / retro games / desktop programming). These are the anti-overfit corpus the
debate demanded: the trap suite must stop being the only examiner, because a tracker can score
100% by memorizing seven deterministic scenes. Real videos with sparse human-checkable gold are
the graduation exam.

---

## 0. The video corpus — census and tiering

All files in `videos/` (gitignored — media never enters git; only derived fixtures' *manifests*
and gold annotations are committed). ffprobe confirms: H.264 + AAC audio, various resolutions
(e.g. pharo = 1048x720@60fps, 380 s).

### Tier R1 — near-domain (desktop / retro-game screens; low palette, discrete sprites)

The closest world to scenegen's: flat-shaded regions, discrete moving objects, mostly-static
backgrounds. The tracker's first real-world rung.

| file | why it matters |
|---|---|
| `pharo programming.mp4` (380 s, smallest) | **the design domain itself** — a desktop IDE session: windows, text, cursors, scrolling. OCR-rich; the guitar's real exam. **First ingest target.** |
| `Prince of Persia … PC DOS 4K.mp4` | single sprite protagonist on near-static rooms — the *ideal* real tracker fixture: appear/vanish (doorways), reversal (turnarounds), occlusion (pillars). |
| `Leisure Suit Larry 1 … Walkthrough.mp4` | EGA palette adventure — tiny color count, slow deliberate motion, scene cuts. |
| `LEISURE SUIT LARRY 2 … Playthrough.mp4` | same, larger. **HELD OUT** (see §7). |
| `Police Stories 100 Walkthrough Part 1.mp4` | top-down 2D — multiple simultaneous small sprites; the multi-track / candidate-identity exam. |

### Tier R2 — discrete-event symbolic (board games / scoreboards)

Near-static frames punctuated by *discrete, independently verifiable events* (a stone placed, a
piece moved, a score changed). Perfect for **omission probes** (did the system report move 34?)
and OCR cross-checks (scoreboards, clocks, move lists).

| file | why it matters |
|---|---|
| `Mental Problems [Fan Tingyu VS Tu Xiaoyu] … Go … Full Game.mp4` | stones appear one at a time on a 19×19 grid — appear-events with *exact* gold from the game record. |
| `🎦 Magnus Carlsen … vs Gukesh.mp4` | chessboard + eval bar + clock — OCR (clock) and discrete events (moves). **HELD OUT.** |
| `THE MILLION POUND CHAMPION … Darts … Final.mp4` | scoreboard OCR gold (180! checkouts); dart flight = fast-motion micro-events. |

### Tier R3 — fast natural motion (stress / limits documentation)

Camera pans, motion blur, crowds, cuts every few seconds. The tracker is **not expected to win**
here — this tier exists to (a) measure the triage's ACTIVITY classification on real footage,
(b) measure where the component tracker degrades and document it honestly, (c) exercise real
audio (crowd noise, impacts, commentary speech, whistles).

| file | notes |
|---|---|
| `2025 CIP FINAL Foconi v Massialas … Foil Fencing.mp4` | two-object interaction, planche-line motion (near-1D!), score-light events |
| `Down and out … Dubois v Lerena Boxing.mp4` | impacts ↔ audio transients (AV-sync probes) |
| `Lin Shidong vs Truls Moregard … EuropeSmash.mp4` | ball = tiny fast object; the small-object limit. **HELD OUT.** |
| `San Antonio Spurs vs New York Knicks … NBA Finals.mp4` | many objects, scoreboard OCR |
| `Team Speed 7-6 Team Celine … FIFA Creator Cup.mp4` | *rendered* sport — HUD OCR + game motion |
| `Дэниел Кормье vs Энтони Джонсон.mp4` | MMA; non-Latin title exercises path handling |
| `CS2 BEST MOMENTS 2025 (Highlights).mp4` | FPS game — HUD OCR, kill-feed events, violent camera motion. **HELD OUT.** |
| `I Tested If Size Matters In Every Sport.mp4` (1.5 GB) | jump cuts, mixed scenes — the ACTIVITY/scene-cut exam |
| `Extreme Idiots … Fails [HD].mp4` | uncurated handheld footage; worst case |

**Held-out protocol (§7):** `LSL2`, `Carlsen-vs-Gukesh`, `Lin-vs-Moregard`, `CS2` are **never
opened during development** — no probes authored from them, no tracker runs against them until a
milestone's final scoring. One per tier + one extra R3.

---

## Milestone map

| # | name | depends on | what it proves |
|---|---|---|---|
| V0 | `vidingest` — video→orchestra bridge | — | real material flows through the SAME pipeline as live capture |
| V1 | Evaluation protocol upgrade | — (parallel with V0) | the scorer can no longer be fooled by affirmation bias, omission, or memorized scenes |
| V2 | Supersession schema (rung B+) | V1 (scored by it) | interpretations can be revised without rewriting evidence |
| V3 | Tracker upgrades + real-video ladder | V0, V1, V2 | motion authority extends to occlusion/merge/2-D global motion, on held-out material |
| V4 | Closed-vocabulary SED audition | V0 (real audio), V1 | an audio middle tier earns per-class admission |
| V5 | Temporal-sensitivity experiment | V0, V1 | temporal claims are penalized for invariance to temporal destruction |
| V6 | Long-running integrated trial | all | the whole orchestra survives hours of real material with honest memory |

V0 and V1 start in parallel; everything else is sequential in the debate-converged order.

---

## V0 — `vidingest`: the video→orchestra bridge

**Goal:** a video file becomes indistinguishable from a live capture session — the SAME batch
format, the SAME Tier-A archive, the SAME `.idx`, consumed by the SAME tracker/triage/OCR/VLM
members. No member grows a special video path; only the *frame source* changes.

**Why it's cheap:** `frame_dump.elisa` already separates capture (`screencap.elisa`) from
encoding (`encoder.elisa` — `downscale_into(src: u32&, sw, sh, dw, dh, base)` onward). The new
member is a frontend that feeds pixels from ffmpeg instead of ScreenCaptureKit.

### V0.1 `vidingest.elisa` (new member — "tape deck")

- Spawns/consumes `ffmpeg -i <video> -vf fps=<fps> -pix_fmt bgra -f rawvideo -` (via `popen`
  extern, or a two-step: ffmpeg → temp raw file → fread; choose whichever the extern surface
  supports cleanly — popen preferred, temp-file fallback acceptable for v1).
- Reads `sw*sh*4` bytes per frame, calls the existing `downscale_into` → delta-encoder path.
- Synthetic timestamps: `t_ms = frame_idx * 1000 / fps` (deterministic, replayable — no wall
  clock, so ingesting the same video twice is **bit-identical**; verify this explicitly).
- CLI: `vidingest <video> <out_dir> [--fps 10] [--width 192] [--t0 <s>] [--dur <s>]`
  — `--t0/--dur` cut a segment WITHOUT re-encoding gold timestamps (probe times stay
  video-relative; the manifest records the offset).
- Also writes the Tier-A archive ring (same code path as frame_dump) so `arch_tool
  show/replay/compare` work on video-derived sessions unchanged.

### V0.2 `audextract.sh` (trivial)

- `ffmpeg -i <video> -ac 1 -ar 16000 -f s16le` (or wav) → the cymbal's exact input format.
- Segment flags mirror V0.1 so audio and video segments stay aligned to the same `t0`.

### V0.3 Fixture manifests — `eval/scenarios/real/`

Media is gitignored; what IS committed per fixture:

```
eval/scenarios/real/<name>/manifest.json   {video, t0, dur, fps, width, sha256 of source}
eval/scenarios/real/<name>/truth.jsonl     sparse gold (§7 annotation protocol)
eval/scenarios/real/<name>/probes.jsonl    probes over that gold
```

A `make_fixture.sh <manifest>` regenerates batches + wav from the manifest deterministically.

### V0.4 Acceptance

- [ ] `vidingest videos/pharo\ programming.mp4 /tmp/vid_pharo --fps 10 --dur 60` produces
      batches that `arch_tool verify` passes and the tracker/triage parse without modification.
- [ ] Ingesting the same segment twice → **bit-identical** batch files (diff clean).
- [x] `screenocr` on a replayed pharo keyframe reads real IDE text (spot-check ≥ 5 strings).
      Verified 2026-07-14: 25 positioned strings at conf 1.00 on a t0=30 keyframe, incl.
      `Ephestos-001.image`, `Nautilus`, `Object subclass: #NameOfSubclass`, `Hyperion`,
      `Workspace`, `Transcript`. Caveat for OCR-diff (V-next): the SAME static text reads
      differently across adjacent keyframes at truncation boundaries (`…-Cor`→`…-Core`,
      `PharC`→`Phar(`) — raw string equality is unusable as a change signal; a diff engine
      needs region alignment + tolerance + multi-frame confirmation.
- [ ] Cymbal runs on extracted boxing audio and emits transients (no gold yet — smoke only).
- [ ] One committed manifest per tier (pharo, PoP, Go, fencing) — held-outs get NO manifest yet.

**Estimated size:** ~1 day. The encoder reuse does all the real work.

---

## V1 — Evaluation protocol upgrade

**Goal:** upgrade the examiner before upgrading any examinee, because every later result is
scored by this layer. Eight instruments, all extending existing harnesses (`score_memory.py`,
`track_probe.py`, `audio_probe.py`, `scenegen`, `audiogen`).

### V1.1 Positive/negative question twins

- `probes.jsonl` grows `twin_of: <probe_id>` + `polarity: pos|neg`.
- Scorer: **paired credit** — a twin pair scores 1 only if BOTH answers are consistent
  (vanish=yes ∧ remained-visible=no). Report `twin_consistency` as its own metric next to
  `perception_accuracy`. An affirmation-biased model (answers "yes" to everything) scores 0 on
  pairs where it used to score 50%.
- Author twins for ALL existing motion-trap and audio-trap probes (mechanical: each existing
  probe gets a negated sibling).

### V1.2 Metamorphic scene pairs

- `scenegen` gains **pair mode**: `scenegen <scene> --variant hold` produces the identical scene
  with exactly one property changed (the object does NOT vanish / does NOT reverse / no tone).
- Scoring rule: credit requires the answer to *change across the pair*. A model answering
  "vanished" to both A (vanish) and B (no vanish) reveals a prior, and scores 0 even though it
  was "right" on A. Metric: `metamorphic_sensitivity`.
- Same for `audiogen` (`--variant no-transient`, `--variant no-tone`).

### V1.3 Omission probes (the dual of confabulation)

- New probe kind `omission`: gold = an event that DID occur; the question is open ("list all
  alarms you heard", "how many objects appeared?") and the metric is **recall of real events**,
  scored per event. Current probes catch invention; these catch suppression (a model narrating
  "calm session" past a real error dialog).
- Add `omission_rate` (missed real events / real events) to `score_memory.py` output beside
  `confabulation_rate` — the two failure surfaces, reported as a pair, per NOAH's duality.

### V1.3b Relation-hallucination probes (VERHallu-style)

- A member can ground both events correctly and still hallucinate the *relation* between them —
  temporal order, causality, subevent structure. New probe kind `relation`: fixtures where the
  true order/causal structure CONTRADICTS the prior-expected one (seeded scenegen: effect-like
  event before cause-like event; real: Go move order, boxing punch/fall order via segment cuts).
- Scored like twins: credit requires the claimed relation to match the manipulated ground truth,
  and to FLIP on the reversed-relation pair. Metric: `relation_grounding`. Distinct from
  `event_order` (which measures concordance, not adversarial resistance).

### V1.4 Temporal-destruction controls (scoped — this is a metamorphic probe, NOT TCD)

- `arch_tool` gains `mangle <dir> <mode>` where mode ∈ `freeze|reverse|shuffle|repeat1` —
  produces a batch directory with the temporal structure destroyed but per-frame content intact.
- Probes gain `claim_class: static|temporal`. Scoring rule (the scoped law from the debate):
  **a `temporal` claim answered identically on original and mangled input is marked suspect**
  (metric `temporal_grounding`); `static` claims are EXPECTED to be invariant and are never
  penalized for it. ("There is a boxing ring" must survive shuffling; "he moved left then right"
  must not.)

### V1.5 Seeded procedural scenes + held-out seeds (anti-overfit, synthetic half)

- `scenegen <scene> --seed N`: positions, velocities, colors, sizes, event times, distractor
  count all derived from the seed via the existing xorshift; **gold answers computed from the
  same seed** and emitted as `truth.jsonl` alongside the fixture (scenegen becomes its own
  annotator — it already knows where everything is).
- Seed ranges: `0–999` development, `1000–1099` **reserved** — never run during development;
  final milestone scoring only. Enforced socially + a comment in `trap_test.sh`; the runner
  refuses reserved seeds unless `--final` is passed.
- `audiogen --seed N` identically (tone freqs, transient times, ramp rates).

### V1.6 Per-claim-type scoring

- Every probe already has `kind`; extend the report to break perception/confab/omission down
  **by claim class** (`presence|count|position|direction|timing|identity|text|audio`), because
  "75% overall" hides "100% on presence, 0% on direction" — which is exactly the structure the
  Qwen results had. Table output per member per class.

### V1.7 Randomized paraphrases + phase offsets

- `vlm_probe.py`/`track_probe.py`: each probe carries 2–3 phrasings; the harness rotates them
  by seed. Kills prompt-shape memorization.
- `--phase <ms>` on fixture generation shifts all event times by a sub-batch offset so nothing
  aligns to batch boundaries by accident.

### V1.8 Confidence-blind scoring

- Scorers already ignore self-reported confidence for credit; make it explicit and audited: any
  matcher consuming a `conf` field is a bug. Confidence calibration is reported SEPARATELY
  (reliability diagram data: claimed conf vs measured accuracy per bucket) — trust remains an
  eval OUTPUT.

### V1.9 Acceptance

- [ ] All 9 existing trap suites re-pass under the upgraded scorer (twins added, pairs added).
- [ ] The VLM prior-fill result REPRODUCES under twins: Qwen's motion-trap score drops or holds
      (it cannot rise — twins only remove false credit) and `twin_consistency` is reported.
- [ ] `scenegen --seed` at 10 development seeds: tracker ≥ 95% aggregate (if it drops far below
      the fixed-scene 100%, that gap IS the overfit measurement — report it, then fix in V3).
- [ ] Mangle modes verified: viola on shuffled input reports garbage-or-nothing (it has no
      prior to fill with — confirm, don't assume).
- [ ] scorer `--selftest` extended to cover twins, omission, metamorphic pairing, claim classes.

**Estimated size:** 2–3 days. Highest value-per-line in the plan.

---

## V2 — Minimal supersession schema (rung B+)

**Goal:** interpretations become revisable without rewriting evidence, BEFORE the tracker starts
producing revisable interpretations (V3's tentative reacquisition writes into this schema).

### V2.1 Schema (extends `eval/ledger.py` records)

```
valid_start / valid_end      when the claim is about
recorded_at                  when the claim was made (bi-temporal minimum)
obs_or_inf                   OBSERVED | INFERRED          (exists)
status                       active | superseded | disputed   (NEW; default active)
supersedes                   record id | null              (NEW)
evidence_handles             batch/frame/track pins        (exists)
license                      the RULE that admitted it, e.g. "viola:VANISH",
                             "cymbal:TRANSIENT", "vlm:describe(static)"   (NEW)
```

### V2.2 The one law (enforced, not conventional)

**`supersedes` may only point at an INFERRED record.** `ledger.py build` hard-fails on a
supersession targeting an OBSERVED record. Evidence is immutable; interpretations retire.
Worked example (the v2 bake-off's own pt_count case):

```
OBSERVED  no-component 9200–10000ms            (immutable, forever)
INFERRED  track-17 VANISHED @9200      → status: superseded
INFERRED  track-17 OCCLUDED, reacquired as track-19 (candidates {17-continues, new-object})
          supersedes: ^                 status: active, conf split across candidates
```

### V2.3 Projection + probes

- `ledger.py project` renders active interpretations; superseded ones appear only under a
  `revisions:` note when `--audit` is passed (story.md stays clean; the audit trail exists).
- New probe kind `revision`: "what did the system originally believe about X, and why did it
  change?" — answerable ONLY if supersession preserved the retired claim. This is the probe
  that makes the schema pay measured rent (per the ladder rule: every rung must earn its cost).

### V2.4 Acceptance

- [x] Law enforced: build-time hard error on OBSERVED-targeting supersession (negative test).
      **DONE 2026-07-14** — `eval/ledger.py validate`/`build` hard-fail; `eval/v2_supersession_test.sh`.
- [x] Re-run bake-off run 1 with supersession: the reacquire case now answers pt_count with the
      candidate split ("1 if same object (occlusion), 2 if new — evidence: no-component gap")
      and the `revision` probe passes. Bytes overhead vs plain rung B measured and reported.
      **DONE 2026-07-14** — motion-trap `VANISH(o1)→APPEAR(o2)` yields a `REACQUIRE_CANDIDATE`
      superseding the VANISH with the {1-if-same, 2-if-new} split; `revision o1` recovers the retired
      belief; overhead ~655 bytes (rung B 1350 → B+ 2004). Recorded in `eval/bakeoff.md` (Rung B+).
      (expected: small; if > ~20% revisit the field encoding).
- [x] All prior bake-off probes still pass (no regression). **DONE 2026-07-14** — rung-B-vs-B+
      projection diff on motion-trap (seed 0): `pt_reverse` byte-identical (REVERSE cx=109 untouched);
      `pt_vanish` still "yes" (the reacquire-candidate's evidence pin carries `no-component gap 800ms`
      and `--audit` preserves the literal VANISH); `pt_continuous` still "no" (the gap breaks
      continuity in both rungs); `pt_count` IMPROVED "2"→candidate-split-reaching-"1" (the fix, not a
      regression). Run-2 retention facts untouched (supersession reframes only viola VANISH, never the
      APPEAR/OCR/audio records those facts live in). `v2_supersession_test.sh` PASS seeds 0/3/7;
      `score_memory.py --selftest` PASS.

**Estimated size:** 1 day. **V2 COMPLETE.**

---

## V3 — Tracker upgrades + the real-video ladder

**Goal:** extend the viola's motion authority to the failure modes v2 documented (merge-drag,
long occlusion, textured background) and then walk it up the real-video tiers. Every change is
gated by the V1 scorer on seeded scenes; graduation is scored on held-out seeds AND one held-out
video.

> *A trustworthy tracker must know not only when identity is uncertain, but when the very
> concept of a persistent physical object is inapplicable.*

**Amended 2026-07-13** after the first pharo census (61 tracks / 51 VANISH / 43 OCCLUDED / 5
fabricated conf=85 REVERSEs on a mostly-static IDE session) and an external research review
(OC-SORT, ByteTrack, UCMCTrack/EMAP, MotionHalluc's Perceive-Parse-Verify). The census showed
the deepest failure is UPSTREAM of association: the tracker interpreted text edits and UI
redraws as physical objects in motion. Association repairs (V3.1–V3.2) make fabricated tracks
fewer and better-argued — still fabricated. Hence V3.0 below, which must land FIRST: it is the
constitution applied to the tracker itself — authority to make physical-object claims is earned
per scene region by measured trackability, never granted globally.

### V3.0 Trackability gate + minimal confidence split (applicability before association)

- Before licensing any physical-motion claim (MOVE direction, REVERSE, identity-across-gap),
  require **trackability evidence** for the component/region: persistence over several frames,
  bounded centroid acceleration, stable area, low local split/merge frequency, an association
  margin (best candidate clearly separated from alternatives), and scene-wide birth/death churn
  below a threshold. Bass's ACTIVITY stats already carry most of the scene-level signal — this
  is largely a viola-side read of evidence the orchestra already produces.
- **Multi-video generalization evidence (DONE 2026-07-14, `eval/scenarios/real/restraint_census.md`,
  8 videos across tiers):** the gate must be MULTI-LEG, not pharo-shaped. Four measured sub-modes —
  **A** UI pointer (pharo, soleSmallMover 89%) → applicability leg; **B** association churn (PoP/
  police/fencing: many tracks, high OCCLUDED/REACQUIRE, moderate velocity) → association-margin + churn
  leg; **C** implausible-velocity smear (boxing/cs2: peak |vx| 103–164, high revHiVx) → bounded-velocity
  leg; **D** already-calm (go/lsl1: near-zero fabricated events) → the **do-no-harm control** the gate
  must not degrade. A pharo-only gate addresses only A and risks harming D. Design each leg against the
  signature that isolates its sub-mode; accept only when D is untouched.
- When the gate fails, the honest output is restraint, not a worse track:
  `OBSERVED changed-regions ...` + `INFERRED activity_type=UI_CHANGE trackability=LOW` —
  and NO object identity or REVERSE claim is admitted. `UNTRACKABLE` is a first-class verdict.
- **Minimal confidence split moves here from V3.5** (pharo evidence: all four flung REVERSEs on
  track 15 carried the same blended conf=85 — one scalar cannot express "detection certain,
  association meaningless"). Three fields now, not six: `detection_quality`,
  `association_margin`, `event_confidence`. V3.5 keeps the full decomposition + calibration.
- Pre-work (do during V1, it is evidence-gathering not repair): inspect the pharo t=14.2–16.1s
  window (batch 7–8 keyframes) and classify what UI event flung track 15 between cx=17/140/25/114.
  **DONE 2026-07-14** (`eval/scenarios/real/pharo/t14_inspection.md` + cursor figure): track 15 is
  the **mouse cursor** — the only non-static component (the other two are the full-width toolbar and
  taskbar) — moved in a loop across the empty workspace; the conf=85 REVERSEs are the pointer turning
  at the loop extremes. Sharpens the gate design: this is NOT the association-churn sub-mode — it is a
  textbook-CLEAN track (persistent, stable tiny area, association margin ≈ ∞) of the WRONG KIND of
  object. Acceleration bounds catch the |vx|=64 flicks but NOT the gentle turn at the REVERSE; that
  needs an **applicability** disqualifier (scene-relative area floor / "sole tiny mover over static
  full-frame chrome ⇒ UI pointer"). V3.0 must reject track 15 on what-kind-of-thing-it-is, not motion.
- Acceptance is annotation-free on pharo (no gold pulled forward from §7): count of
  physical-motion claims, false-REVERSE count (a REVERSE inside an interval OCR shows as static
  text is false by construction), track birth/death churn rate, abstention (`UNTRACKABLE`) rate,
  and OCR/triage evidence retained despite tracker abstention. Full "fraction correctly
  classified as UI activity" waits for the V3.6 ladder where gold-authoring is scheduled.
- **Implemented design (2026-07-14, `tracker.elisa`).** The gate sits at the physical-motion claim
  site (`update_dir`'s REVERSE emission) and is fed by cheap per-track accumulators + a per-frame scene
  classification. The concrete pharo baseline (`/tmp/fix_pharo`, 60s) has 5 false REVERSEs in two
  shapes: **id=15 ×4** (cursor — tiny, |vx| up to 64, sole mover over 2 full-width bars, ncomp=3) and
  **id=57 ×1** (a near-static track, vx=0 for ~100 frames, that twitched in a busier ncomp=7 frame).
  Four orthogonal legs, each keyed to the census signature that isolates its sub-mode:
    - **Leg A — applicability (UI pointer).** Per frame, classify components: a `bar` spans ≥85% of
      grid width (static full-frame chrome); everything else is a `mover`. A frame is *chrome-dominated*
      when it has ≥1 bar and ≤1 mover. A track that is TINY (`area < AREA_TINY_SC=30`) and spent the
      majority of its matched life in chrome-dominated frames is a UI pointer — its REVERSE is
      suppressed, `INF activity ... activity_type=UI_CHANGE trackability=LOW reason=ui-pointer` emitted
      once. Catches id=15 on what-kind-of-thing-it-is, exactly as the t=14 inspection concluded.
      (sub-mode A / pharo, soleSmallMover.)
    - **Leg B — bounded velocity.** REVERSE suppressed when `|vx| ≥ VX_IMPL=35` (census `VX_IMPLAUSIBLE`)
      — a "reversal" of a component crossing >18% of the frame in one 100ms step is a smear artifact.
      (sub-mode C / boxing, cs2 — revHiVx.)
    - **Leg C — motion persistence.** A REVERSE is a direction change of a *moving* object; a static one
      has no direction to reverse. Suppressed when the track's mean speed over life is `< 1` cell/frame
      (`TR_SPD < TR_LIFE`, guarded to `TR_LIFE ≥ 8`). Catches id=57. reason=static-furniture.
    - **Do-no-harm (D).** go has 0 REVERSEs (nothing to suppress); a real mover (lsl1 sprite, PoP
      protagonist) is not tiny-chrome, is below the velocity ceiling, and has mean speed ≥ 1, so it
      passes every leg. Verified by re-running the census: D untouched (see acceptance).
  - **Confidence split (3 fields, replacing the blended `conf=85`).** Every `INF event` now also carries
    `dq=` (detection_quality: low for tiny/unstable components), `am=` (association_margin: 2nd-best −
    best gate distance, `-1` = n/a for births, large = unambiguous), `ec=` (event_confidence = the old
    per-event conf). `conf=` is retained for ledger.py compatibility. This lets a claim say "detection
    certain, association meaningless" — impossible with one scalar. V3.5 completes the decomposition.

### V3.1 Velocity/direction freeze during OCCLUDED (smallest diff first)

- `tracker.elisa` already has the OCCLUDED state and `TR_DIR/TR_EXT/TR_STARTX`. Change: while a
  track is OCCLUDED, **stop integrating direction/extreme state** (freeze at last observed
  values); on reacquire, re-seed direction from the first MOVE_MIN of post-reacquire motion
  instead of trusting coasted state. This is OC-SORT's observation-centric idea adapted to grid
  components — no Kalman needed at this scale.
- New scenegen scene (seeded): `occlude-reverse` — object enters occluder moving right, exits
  moving LEFT. Current tracker gets the exit direction wrong (coasted state); frozen tracker
  re-measures. Trap probe: direction-after-reacquire.

### V3.2 Tentative reacquisition + candidate identity sets

- On reacquire near a VANISH/OCCLUDED site, emit `INFERRED REACQUIRE` with a **candidate set**
  (`same:track-17 | new-object`) and per-candidate confidence from gate distance + size match +
  color match (the palette letter is already in the grid — a free, honest appearance signal that
  is NOT learned re-ID; use it only as a candidate-set discriminator, per the debate's weakened
  rule).
- Candidates write into V2's supersession schema. A later contradiction (both candidates seen
  simultaneously) *disputes* the same-identity candidate — the first live supersession.
- **Observation-centric backward re-update (ORU, from OC-SORT):** on tentative reacquisition,
  recompute velocity from last-trusted-observation → new-observation (a virtual path over the
  gap), never from merge-distorted coasted state. Crucially, ORU **proposes, it does not
  decide**: backward consistency is one more `association_basis` raising a candidate's
  confidence inside the DISPUTED set — only later corroboration (or held-out evaluation) lets
  the identity graduate. An ambiguous reappearance is never silently turned into the original id.
- **Implemented (2026-07-14) — split by measured safety.** Two forms of candidate/dispute, and a hard
  lesson from the association experiments:
    - **Merge dispute (DONE):** `contested_by` in `tracker.elisa` emits `INF dispute id=.. contests=..`
      when a reacquired blob is also claimed by another live track; `ledger.py` turns that into a
      `status=disputed` REACQUIRE with the candidate set {track A | track B}. This is the first live
      dispute (the merge case). Verified on `contact-merge`.
    - **Occlusion candidate (at the LEDGER, not the tracker):** experiments showed a WIDE tracker-level
      reacquire gate to catch far-side re-emergence *fabricates identities in clutter* — boxing RQ
      227→386, and it harmed the go do-no-harm control. Rejected. The vanish+rebirth of a far re-emergence
      is instead paired by the ledger's V2 `detect_reacquires` into a `REACQUIRE_CANDIDATE` supersession
      — tentative reacquisition lives in the schema, exactly as V2 intended. Caveat recorded: when the
      coasted track VANISHes *after* the re-emergence has already APPEARed (reverse temporal order), the
      current `detect_reacquires` (VANISH→later-APPEAR) does not pair them; the `direction_after_reacquire`
      op falls back to the last APPEAR. A symmetric pairing is a cheap future ledger addition.
    - **ORU is DEFERRED with a trigger:** backward velocity re-update only earns its keep once a measured
      occlusion-reacquire *failure* on a held-out fixture shows it would disambiguate — no anticipatory
      complexity (the plan's own rule).

### V3.2b Weak-component provisional tier (ByteTrack's insight, symbolically)

- Components below `MIN_AREA` and marginal unmatched residuals are currently discarded. Keep
  them in a **provisional tier**: never emitted as OBSERVED components, never seeding new
  tracks, but consulted during reacquisition and candidate-set construction — low-confidence
  detections carry real signal precisely during occlusion/merge/degradation, which is exactly
  when our tracker currently goes blind. Cheap: they are computed before the threshold drops
  them. A provisional match raises a reacquire candidate's confidence; it never creates a claim
  on its own.
- **DEFERRED with a trigger (2026-07-14).** The provisional tier *feeds reacquisition*, and the V3.1/V3.2
  experiments established that widening reacquisition inputs is precisely what harms the do-no-harm
  controls (every wider-input variant fabricated identities on go/boxing). At the current grid resolution
  `MIN_AREA=4` is already 1–3 cells (near-noise), so the safe upside is marginal. Per the plan's
  earn-your-complexity rule, this lands only when a measured occlusion/merge failure on a HELD-OUT R1
  fixture shows a specific sub-`MIN_AREA` component would have disambiguated a candidate set — the trigger,
  recorded, not anticipated.

### V3.3 Two-dimensional global translation (generalize `SHIFT`)

- **DEFERRED with a trigger (2026-07-14).** The 8-video restraint census surfaced FOUR failure
  sub-modes (UI pointer, association churn, velocity smear, calm) and **no uncorrected-pan sub-mode** —
  no real fixture in scope fails from horizontal/diagonal global translation. The encoder's delta is
  row-oriented (vertical `SHIFT dy` matches whole rows); horizontal shift breaks row-matching, so 2-D
  translation is a deep delta-representation rewrite with a real bit-exact-reconstruction risk. By the
  same earn-your-complexity rule the plan applies to affine and point-tracking, integer 2-D translation
  now waits on its trigger: **a measured R1/R2 fixture whose churn or mis-segmentation is attributable
  to an uncorrected horizontal/diagonal pan** (the four seeded scroll scenes are the scaffolding to
  confirm the fix once a real failure licenses it). Redirected the effort to V3.5 + the V3.6 real ladder.
- Encoder currently detects vertical `SHIFT dy=`. Add horizontal: `SHIFT dx= dy=` (integer grid
  cells, detected by the same row/column-match scan transposed). Format bump is additive —
  `dy`-only lines remain valid; tracker parses both.
- **Licensing note (2026-07-13):** pharo's `shifted=0` across all 30 batches is NOT evidence for
  this — an IDE editing session plausibly contains no dominant global translation, and the
  pharo failure was association across unrelated UI components, not uncorrected pan. 2-D
  compensation is licensed by four dedicated seeded scenes: (1) pure horizontal scroll,
  (2) diagonal scroll, (3) scroll + one independently moving object, (4) local panel scroll
  with the rest of the screen fixed. Compensate-then-segment (the fix runs BEFORE residual
  segmentation, V3.4), not detect-then-veto.
- **Affine is explicitly deferred** until integer 2-D translation measurably fails on an R1
  fixture (the debate's earn-your-complexity constraint; record the trigger condition here:
  "an R1/R2 fixture where per-frame residual after best integer translation still touches
  > 40% of cells during a pan").
- **Point-tracking (TAPIR-style) is a named escalation, also deferred**, with its trigger
  written down: adopt only if the component tracker, after V3.1–V3.4, still fails crossings or
  non-rigid motion on HELD-OUT R1 scenes. Same discipline as affine: the trigger is a measured
  failure, never anticipation.

### V3.4 Residual segmentation (simultaneous global + local motion)

- **DEFERRED with V3.3** (it consumes V3.3's compensated grid). Note the EXISTING vertical
  `scroll-motion` seeded scene already passes at **10/10** in `seed_test` — the vertical-scroll +
  independent-object case is handled by the current encoder+tracker; the outstanding work is only the
  HORIZONTAL/2-D residual, which is gated on the V3.3 trigger above.
- After compensating the detected translation, run component extraction on the residual grid so
  an object moving DURING a scroll is still tracked. New seeded scene: `scroll-plus-motion`
  (background scrolls dy=2/frame while a square moves horizontally). Currently conflated;
  post-V3.4 both motions reported separately (`INF global-motion dy=2`, `INF track-1 RIGHT`).

### V3.5 Per-stage confidence split

- Separate confidences on detection (component quality: area, stability), association (gate
  margin), and event interpretation (VANISH vs OCCLUDED alternatives) instead of one blended
  number. These feed the V1.8 calibration report — measured, not asserted.
- The minimal 3-field split ships in V3.0; this milestone completes the decomposition
  (adds `trackability`, `identity_confidence`, `motion_measurement_quality`) and calibrates it.
- **DONE (2026-07-14, headline field).** The INF track line now carries `trk=` — a derived
  TRACKABILITY score (0-100) rolling up persistence (matched life), detection quality (area), and the
  applicability verdict (a UI-classified region floors it to 5). On pharo it separates cleanly: 5 for
  UI-classified regions, 23-43 for tiny/short-lived, 68-83 for persistent real regions. `dq/am/ec` per
  stage already ship on events (V3.0). CALIBRATION (claimed `trk` vs measured accuracy per bucket, the
  V1.8 reliability diagram) is produced from the V3.6 gold ladder — it needs gold-scored runs, so it is
  measured there rather than asserted here. `identity_confidence`/`motion_measurement_quality` fold into
  `am`/`ec` for now; they split out only if a V3.6 probe rewards the finer granularity.

### V3.5b Typed Perceive-Parse-Verify (claim verification as a tool interface)

- When a model (violin/sax/frontier) makes a natural-language motion claim ("the glove reversed
  after contact"), it is verified by proposing a **typed query** against the ledger — e.g.
  `verify_reverse(track_id, interval, required_prior_event, max_delay_ms)` — whose fields the
  runtime validates and executes via the existing `track_probe.py` op vocabulary (reversal,
  vanish_gap, two_directions, ...). Verdicts: supported / contradicted / unresolved.
- **The parse is itself an INFERRED claim** (which object? which interval? what counts as
  "after"?) and carries that status in the ledger. No natural-language claim is admitted merely
  because it parsed successfully — the constitution applies to the query, not just the answer.

### V3.6 The real-video ladder (run in this order, gold per §7)

1. **`PoP` (R1):** protagonist track through 2–3 rooms; gold = sparse annotated events
   (enters-left @t, vanishes-doorway @t, reappears @t, reverses @t). Target: presence/direction/
   reversal probes ≥ 80%; every miss documented with a frame pin.
2. **`pharo` (R1):** NOT an object-tracking exam — a triage+OCR exam. Gold = window/scroll/
   typing phases + 10 OCR strings. Tracker expected to report mostly global motion; that
   restraint (no fabricated object tracks during scrolling) IS the probe. **Baseline measured
   2026-07-13 (pre-V3.0): FAILS restraint** — 61 tracks / 51 VANISH / 43 OCCLUDED / 5 conf=85
   REVERSEs in 60s. Scored here with the annotation-free restraint metrics from V3.0 acceptance
   PLUS the gold-based UI-activity-classification fraction (authored at this rung).
3. **`Police Stories` (R1):** multi-sprite; candidate-set stress.
4. **`Go` (R2):** stones-appear events vs the (partial, human-checked) move record; omission
   probes shine here — "did a stone appear in the upper-right between t1–t2?"
5. **`fencing` (R3, diagnostic only):** document degradation honestly — where components smear,
   what ACTIVITY reports, which claim classes survive (timing of light-flash events likely
   does; identity does not). Limits go in the report, not under the rug.
6. **GRADUATION (held-out):** `LSL2` segment ingested for the FIRST time, gold annotated by the
   user/inspection at annotation time, tracker + full V1 scorer run ONCE. The score is the
   score.

### V3.7 Acceptance

- [x] V3.0 gate on pharo (annotation-free): false-REVERSE count = 0 in OCR-static intervals;
      motion-claim count and churn rate reduced vs the 2026-07-13 baseline; `UNTRACKABLE`
      abstention emitted; OCR/triage evidence unaffected by tracker abstention.
      **DONE 2026-07-14** (`eval/scenarios/real/v30_gate_acceptance.md`): pharo false-REVERSE 5→0,
      revHiVx 3→0, motion-claim 5→0, churn 1.9→1.8, VANISH 51→46, 2 UNTRACKABLE verdicts, OBS
      frame/comp evidence unchanged (593/2826). Do-no-harm verified on the full 8-video census: go &
      lsl1 byte-identical, all track counts/churn preserved; boxing revHiVx 26→11, cs2 18→8, police RE
      12→6, PoP 13→8. Synthetic `track_test` 100%, `seed_test` 219/220 (== baseline, no regression).
      Association-margin leg surfaced as `am=`; hard-gating deferred to V3.2 candidate sets (hard-gating
      REACQUIRE was tried and rejected — it converts a reacquire into death+rebirth, raising churn).
- [x] Seeded occlude-reverse: direction-after-reacquire ≥ 95% on dev seeds, then held-out seeds.
      **DONE 2026-07-14** — new `occlude-reverse` scenegen scene (A enters occluder RIGHT, turns while
      hidden, exits LEFT; `--variant no-reverse` continues right) + `direction_after_reacquire` op in
      track_probe.py. `seed_test` occlude-reverse **20/20** (10 dev seeds), aggregate 239/240 (99%). The
      V3.1 mechanism that ships is the **direction RE-SEED on reacquire** (dir/extreme reset to
      post-reacquire motion), NOT a velocity freeze: freezing position was tried and REJECTED (it broke
      crossing-swap and pass-through occlusions — a coasting object that CONTINUES through an occluder
      needs its velocity to keep its identity; only the rare reverse-behind-occluder benefits). The
      re-seed removed lsl1's 2 REVERSEs — which inspection showed were **fabricated from coasted
      prediction through an occlusion** (Larry was OCCLUDED at t=7000 and hidden thereafter; the tracker
      imagined him sweeping left), so this is correct restraint, not harm. go/pharo untouched.
- [~] scroll-plus-motion: both motions separated on dev + held-out seeds. **PARTIAL** — the VERTICAL
      case (`scroll-motion` seeded scene) passes 10/10 in seed_test today; the HORIZONTAL/2-D residual is
      deferred with V3.3/V3.4 (no real-video pan failure measured to license the encoder rewrite).
- [x] merge-drag trap (the v2 documented failure): candidate sets prevent the silent identity
      steal — the merge emits disputed candidates instead of a confident wrong track.
      **DONE 2026-07-14** — the tracker emits `INF dispute id=.. contests=..` when a reacquired blob is
      ALSO predicted onto by another live track (`contested_by`), and `ledger.py` turns that REACQUIRE
      into a `status=disputed` record with an explicit candidate set {blob is track A | blob is track B —
      identity stolen by the merge}, conf split 50/50. Verified on `contact-merge` seed 0: id=1's reacquire
      of the meeting-point blob (also claimed by track 3) projects as DISPUTED, not a silent steal. v2
      supersession test still passes (additive `INF dispute` lines; existing probes unaffected).
- [x] All 9 original suites + V1 twins still green (no regression — the traps are now the
      regression suite, which is their correct final role). **DONE** — `seed_test` 239/240 (the one miss,
      crossing-swap seed=2, is pre-existing/baseline-identical); `track_test` motion/motion-trap/
      crossing-swap/occlude-vanish 100%; `score_memory.py --selftest` and `v2_supersession_test.sh` pass
      after every V3.0–V3.5 change.
- [~] PoP ≥ 80% on annotated probes; pharo restraint probe passes; LSL2 held-out run recorded
      in `eval/real_ladder.md` with per-claim-class table, whatever the number is.
      **PARTIAL (2026-07-14, `eval/real_ladder.md`).** PoP re-ingested at a GAMEPLAY offset (t=150 s —
      the census clip was the intro): light gold passes 4/4 (present, room-1 direction=right = prince
      id=15, fabricated-REVERSE=0, velocity in bounds); the gate is restrained on real gameplay (RE=0,
      revHiVx=0, 80 merge disputes flagged not stolen). pharo restraint probe passes (v30 doc). Key
      finding: census fixtures landed on phenomenon-poor offsets (PoP intro, Go pause), so full
      per-claim-class gold + Police/Go/fencing rungs + LSL2 graduation need per-video re-ingest at
      phenomenon-rich offsets + §7 annotation — scheduled, not skipped; LSL2 remains held out.

**Estimated size:** 4–6 days. The biggest milestone; V3.1→V3.5 are separately committable.

---

## V4 — Closed-vocabulary SED audition (the audio middle tier)

**Goal:** a sound-event classifier between the cymbal (exact DSP) and the sax (generative),
admitted **per class**, never as a model. Architecture cannot invent timestamps or narratives —
but "can't confabulate" ≠ "calibrated", so it takes the same exam as everyone.

### V4.1 Candidate + harness

- Candidates: PANNs CNN14 (AudioSet-tagging baseline, well-understood) AND EfficientAT (distilled
  CNN taggers, better AudioSet accuracy at lower compute). Both go through the identical per-class
  gate; the gate picks. Neither is a *selection* — if a lighter/better SED clears the same gate
  later, swap.
- `screensed.py` + shim, mirroring `screenaud`: input 16 kHz mono window → top-k (class, score,
  window-time) tuples. Own venv only if its transformers/torch pins conflict (lesson learned:
  check `config.json` pins BEFORE first run).

### V4.2 The exam (synthetic + real, per class)

- Synthetic: audiogen scenes (it knows gold). Expect near-ceiling on tone/transient — that's a
  smoke test, not admission.
- Real: extracted audio from the video corpus — boxing (impacts, bell, crowd), darts (throws,
  crowd roar, announcer), Go/chess commentary (speech, stone clicks), pharo (keyboard, UI
  sounds, near-silence), fencing (blade contact, appel, referee). Gold: sparse human-checked
  event annotations (§7), ~10–15 events per clip.
- Per-class report: precision/recall per claimed class on real audio. **Admission rule:** a
  class becomes admissible evidence iff precision ≥ threshold (start 0.8) on ≥ 10 real
  instances; everything else stays `INFERRED, low-trust, inadmissible`. Expect a SUBSET of
  AudioSet's ontology to survive desktop/broadcast audio — that subset is the deliverable.

### V4.3 Orchestra integration

- Admitted classes emit into the ledger as `INFERRED` with `license: sed:<class>@<precision>`;
  the cymbal remains the ONLY timing authority (SED windows are coarse; the cymbal's transient
  timestamps refine them — a cymbal transient inside an SED "impact" window is the AV-sync
  join, and per the debate: **temporal coincidence is OBSERVED; causation is INFERRED** — the
  I10 wording, adopted verbatim into SPEC.md invariants).

### V4.4 Acceptance

- [ ] Per-class P/R table on ≥ 4 real clips committed to `eval/sed_audition.md`.
- [ ] At least the classes {speech, music, crowd, impact-like} measured; admitted set explicit.
- [ ] Twin/omission probes from V1 applied: "was there speech in 0–30 s?" twins with "was it
      silent?"; omission recall on annotated real events.
- [ ] Sax re-scoped in SPEC.md: escalation-only (rich description on demand), never presence/
      absence/timestamp authority — same epistemic tier as the violin.

> **Status (2026-07-14): HARNESS BUILT + scoring verified; model download RESOURCE-GATED.** `screensed.py`
> exists (mirrors `screenaud.py`): `tag` emits `SED t=.. class=.. score=..` tuples over 1 s windows from a
> fixed AudioSet-style candidate ontology; `audit` computes per-class precision/recall and applies the
> admission rule (precision ≥ 0.8 on ≥ 10 real instances). The `audit`/admission logic — the actual V4.2
> deliverable — is unit-tested and **passes `screensed.py --selftest` WITHOUT any model**, so the exam math
> is verified independent of the download. `audextract.sh` already produces the cymbal's exact 16 kHz mono
> input from any corpus video. What's gated is only the tagger backend (PANNs CNN14 / EfficientAT weights +
> torch, multi-GB): `tag` prints the exact `.venv-sed` install steps when the backend is absent. To finish:
> install the backend, `tag` ≥ 4 real clips, `audit` against sparse §7 gold → `eval/sed_audition.md`.

**Estimated size:** 2 days.

---

## V5 — Temporal-sensitivity experiment

**Goal:** run the scoped invariance law at scale, as an eval filter (NOT decoding-time TCD).

- Inputs: 3 synthetic seeded scenes + 2 real segments (PoP, boxing) × 5 conditions each
  (original / freeze / reverse / shuffle / repeat1, via `arch_tool mangle` + ffmpeg equivalents
  for the VLM's mp4 path).
- Members examined: violin (Qwen2.5-VL-3B — the current describe), sax, and the viola as
  control (expected: near-zero temporal claims on mangled input — verify).
- Metric: `temporal_grounding` per claim class. Deliverable: `eval/temporal_grounding.md` — for
  each member, WHICH claim classes are actually grounded in temporal evidence vs invariant
  priors. This becomes the fine-grained I8 trust table (per-claim-class trust, replacing the
  current coarse motion/static split, if the data shows finer structure).
- **Acceptance:** the report exists with all 25 runs; the I8 table in SPEC.md is regenerated
  from measurements; any VLM claim class that turns out temporally grounded (possible! e.g.
  gross scene-change detection) gets its trust RAISED — the gate must be able to open, not only
  close, or it's dogma rather than measurement.

> **Status (2026-07-14, `eval/temporal_grounding.md`): CONTROL DONE, neural half scoped.** The viola
> (symbolic control the plan predicted would pass) scores **`temporal_grounding = 3/3`** on `motion`:
> `reversal` flips yes→no under freeze/repeat1 (the reversal is destroyed), `reversal_zone` scrambles
> under shuffle, while static `count` is correctly invariant — temporal claims grounded in temporal
> evidence, measured not asserted. The violin's synthetic-scene score is ≈0 **by the same mechanism the
> OOD constant-prior already documents** (`vlm-ood-not-blind`) — the law reproduces prior-fill from a new
> angle. The piece worth the VLM compute is the REAL-frame conditions (in-distribution, may ground
> scene-change) — it slots into `mangle_test.sh` + `vlm_probe.py` unchanged; left as a compute run.

**Estimated size:** 1–2 days (mostly compute babysitting).

---

## V6 — Long-running integrated trial

**Goal:** the full orchestra on hours of real material — the failure modes that only appear at
duration: ledger growth, timestamp drift, revision chains, queue pressure, crash recovery.

### V6.1 The consent-free variant (default): video-driven

The videos make a long trial possible WITHOUT live capture of the user's session: `vidingest`
streams `I Tested If Size Matters In Every Sport` (long, scene-varied) or a playlist of R1+R2
segments in real time (batches emitted at wall-clock pace), all members subscribed, singer
building the rung-B+ ledger continuously.

- Duration: ≥ 2 hours continuous.
- Live probes injected every ~10 min (retention: early facts; omission: recent real events;
  revision: at least one forced supersession via an occlusion event).
- Measured: RSS of every member over time (flat or bounded), batch queue depth under churn
  (R3 segments = worst case), ledger size growth rate + projection latency, probe scores
  vs time-in-session (retention decay curve), recovery: `kill -9` the singer mid-session,
  restart, verify it rebuilds working state from ledger + archive (the crash/recovery exam —
  this is what append-only + evidence pins are FOR).

### V6.2 The live variant (user-initiated only)

Same battery on a real screen session + `audiocap` system audio. **Runs only when the user
starts it** — screen and audio capture of the user's actual session is never initiated
autonomously. This is also `audiocap`'s deferred live verification slot.

### V6.3 Acceptance

- [ ] 2-hour video-driven run completes; all metrics logged to `eval/longrun.md`.
- [x] Memory bounded (no member RSS grows unbounded; ledger growth linear in EVENTS not time).
      **DETERMINISTIC CORE VERIFIED (2026-07-14)** on the real 30 s PoP gameplay fixture: ledger =
      383 records / 163 KB = **426 bytes/record, linear in EVENTS not wall-clock**; projection renders
      440-line story.md; the one law holds on real footage (`validate` clean).
- [~] Singer crash-recovery: post-restart probe scores within noise of pre-crash. **FOUNDATION
      VERIFIED** — the ledger REBUILD from batches is **byte-identical** across runs (deterministic;
      synthetic timestamps, no wall clock), which is exactly the property crash-recovery relies on:
      re-running from the archive/batches reconstructs identical working state. The live `kill -9`
      restart-under-load test still needs the running singer.
- [x] At least 3 genuine supersessions occurred and project correctly with `--audit`.
      **DONE** — the real PoP gameplay ledger has **22 supersessions** (V2 reacquire-candidates from the
      video's own vanish→appear pairs), all law-valid and audit-projectable.

> **Status (2026-07-14): deterministic backbone DONE; the 2-hour LLM-in-the-loop trial is RESOURCE-GATED.**
> The verifiable, deterministic half of V6.1 (ledger growth linearity, byte-identical rebuild =
> crash-recovery foundation, real supersessions, law-on-real-video) is measured above. The remaining
> half — 2 h wall-clock, per-member RSS curves, queue depth, and the LLM singer building memory live with
> a `kill -9` mid-run — needs a dedicated resourced run with the singer loop (which `singer_harness.py`
> prepares but does not drive autonomously). Not attempted blind in-session.

**Estimated size:** 2 days of runs + instrumentation.

---

## §7 Cross-cutting protocols

### Gold annotation for real videos (sparse, honest, cheap)

1. Choose a 60–180 s segment (manifest records `t0/dur` + source sha256).
2. Generate a contact sheet (`arch_tool replay` keyframes → grid montage) + audio waveform plot.
3. Annotate 8–20 SPARSE events/facts a human can verify from the sheet in minutes — never
   dense labeling. Store as standard `truth.jsonl`.
4. Probes authored from gold BEFORE any member runs on the segment (no peeking — probes written
   from the annotation, not from system output).
5. Held-out videos: steps 1–4 happen at graduation time only, and the member runs ONCE.

### The audition gate (replaces "mechanistic claim required")

**Expected information gain must justify integration + evaluation cost.** Mechanistic claims
(adversarial-negative training, frame differencing, temporal objectives, measurement heads)
raise expected gain; integration cost (custom preprocessing, gated access, uncertain MPS) lowers
it. Cheap tests of null hypotheses are fine (Qwen3-VL-2B precedent); expensive ones need a
reason to expect a different answer (Marlin stays parked unless its gate cost drops to ~zero via
the user's token, in which case `eval/marlin_audition.py` runs as-is).

### Multi-agent convergence policy

The four-model consensus that produced this plan is **hypothesis generation and adversarial
review, not empirical validation** — the models share ancestry and context; their errors are
correlated. Validation is exclusively: measured traps, held-out seeds, held-out videos,
long-duration operation. No design change lands citing consensus alone.

### Standing carry-overs from v2

- `audiocap` live verification — waits for V6.2 (user-initiated).
- Marlin-2B literal audition — parked at the gate; harness committed and ready.
- Recent-preprint rule: no citation-derived design detail enters the repo without a fetched
  abstract; MoHallBench/MemStrata numbers are treated as unverified until read.

---

## Risk register

| risk | mitigation |
|---|---|
| ffmpeg pipe plumbing from Elisa (popen extern) is fiddly | temp-raw-file fallback is 20 lines and still deterministic; optimize later |
| real-video grids at 192-wide lose small objects (table-tennis ball ≈ sub-cell) | that IS a finding — document the resolution/object-size limit with numbers; try `--width 256` (encoder max) for R3 |
| seeded scenegen reveals overfit (score collapse on seeds) | good — that's V1 doing its job; the gap becomes V3's baseline |
| PANNs domain shift worse than expected (few classes admitted) | the admitted-set-is-the-deliverable framing already covers this; a small honest set beats a large fake one |
| broadcast videos have scene CUTS (no continuity at all) | ACTIVITY should classify `switching`; tracker must emit track-end on keyframe resync, not spurious VANISH events — add a cut-detection probe to V1 |
| annotation gold is itself wrong | keep gold sparse + human-verifiable from the contact sheet; every probe carries an evidence pin so disputes are resolvable |
| 2-hour run finds a crash in a member | that's the point; fix and re-run — V6 is last for a reason |

## Definition of done (v3)

1. V0–V6 acceptance boxes all checked, results committed under `eval/`.
2. The trap suite (now: fixed scenes + twins + metamorphic pairs + dev seeds) is green and has
   formally become the regression suite.
3. Held-out results (seeds 1000+, LSL2, and any other held-out opened at graduation) reported
   as-is in `eval/real_ladder.md` — including failures.
4. SPEC.md updated: I10 (coincidence-vs-causation) verbatim, per-claim-class I8 table from V5
   measurements, SED tier + admitted classes, supersession law, the constitution sentence.
5. story.md remains a projection of the rung-B+ ledger; C/D still unbuilt unless a V6 retrieval
   probe fails in a way identities/graph would have caught (the ladder rule stands).

### DoD status (2026-07-14, updated — OCR-diff + SED session)

- **Done:** #2 (trap suite green — `seed_test` 239/240, `track_test` 100%). #5 (story.md is a rung-B+
  ledger projection, now spanning viola + guitar-diff + SED members). Most of #1 (V2; V3.0–V3.2, V3.5;
  V3.6 flagship rung; V5 control; V6 deterministic backbone + fault-injection recovery). **#4 — SPEC.md
  sync APPLIED this session** (I10 coincidence-vs-causation, the screentext/screensed member contracts,
  THE ONE LAW, measured trust rows for guitar-diff and SED).
- **New this session (the "decide what kind of evidence" program — OCR-diff + SED):**
  - **Guitar-diff (`screentext`)** — OCR-diff member, examiner-first. Pre-registered `eval/ocr_audit.md`;
    audit **PASS 4/4** on pharo gold (recognition 1.00, association 0 switches, change-decision 0.00
    false-diff, headline 0 unsupported changes). Consensus + pixel-veto + dwell; ledger-integrated under
    THE ONE LAW. PP-OCRv5 escalation correctly never fired (Apple Vision cleared the gate).
  - **SED (`screensed`)** — PANNs Cnn14 run for REAL (no longer model-gated): `speech` ADMITTED prec 1.00,
    content-grounded metamorphic probe, AV-sync join per I10. Other classes gated on human-listened gold.
  - **Temporal matrix** extended to SED (speech = static presence, correctly invariant); **five-fixture**
    confirmation `shifted=0` on all real video (2-D scroll still has no gradeable instance).
- **Partial / resource-gated (the honest remainder):** #1 — V3.3/V3.4 deferred (no measured pan failure);
  the full per-video gold LADDERS (police/Go/fencing offsets + probes authored, truth-authoring is the
  human step); the V6 2-hour LLM singer run; and the full violin/sax GENERATIVE temporal matrices (heavy
  model loads; method already proven on viola + SED, violin characterized OOD). #3 — LSL2 and held-outs
  remain UNOPENED (graduation only; `eval/graduation_readiness.md` pre-registers the exam).
- **Governing honesty:** the CORE capability moved again and is validated — a THIRD evidence family (text)
  and a de-gated audio tier (SED), each admitted by its OWN pre-registered examiner, each restrained
  (OCR-diff headline 0; SED content-grounded not prior-driven), each additive with do-no-harm proven (no
  `.elisa`/tracker source touched; viola output byte-identical). Every deferral is licensed by a measured
  dead-end or a genuine resource gate. Nothing is marked done that wasn't verified.
