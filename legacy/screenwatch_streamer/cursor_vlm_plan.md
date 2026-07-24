# Plan: the perception cursor + local-VLM member ("violin")

Status: COMPLETE (M1–M5). Roadmap position: extends steps 4/7 (archive queries → attention-driven
retrieval) and feeds step 6 (representation bake-off, candidate 7).

Progress: **M1 done** — `screenvlm` + `setup_vlm.sh` + repo-local `.venv-vlm`; verified on a live
34s capture (read exact title-card text; load ≈ 10 s, infer ≈ 8–12 s, peak RSS ≈ 10.7 GB, MPS);
partial-span / evidence-exhausted paths exercised. **M2 done** — SPEC gains I8, the violin member,
the screenvlm contract, "The cursor" section, `describe` query type + evidence grammar; watcher_
protocol → v3.1. **M3 done** — `motion`/`motion-trap` scenegen scenes (bit-exact), the
`eval/trap_test.sh` + `eval/vlm_probe.py` harness, and the measured confab profile: on abstract
motion both the 3B and 7B fail (honest 50%, traps 25%/75% confab — miss the vanish, call a mid-field
reversal an edge-bounce; density-invariant, so prior-fill not sampling), while the 3B is strong at
text/scene. Trust policy FILLED IN (SPEC §I8): `describe` Inferred-only, split by claim type —
text/scene usable, spatial-motion untrusted (needs symbolic corroboration, never an EVENT alone).
**M4 done** (plumbing) — drove the full loop on a real 34s capture: escalation → `queries/q_v1.md`
(type describe) → screenvlm ("LOVE CAMP 7") → `q_v1.answer.md` (INFERRED, OCR-corroborated) →
boss consolidation into log.md with `[arch seq 31..85 t=4072..10946]`; evidence pointers re-decode,
the event parses, the text claim re-grounds via `arch-ocr.sh` (OCR reads "LOVE CAMP" + "Video
Nasties Ranked | Episode 1 | 72-61"). Resource numbers: M1 standalone (load ~10s, infer ~12s, peak
RSS ~10.7 GB) plus a live co-run — recorder throughput 7.75 fps under VLM load vs 4.15 fps idle
(the VLM does not starve the change-driven recorder; screenvlm peak ~11.7 GB, within 24 GB).
**M5 done** — bake-off candidate 7 registered (design note, below / eval/scenarios.md).

All milestones M1–M5 complete. The honest headline: the violin is a strong *text/scene* reader that
runs safely beside the recorder, but local Qwen2.5-VL (3B and 7B) cannot be trusted for abstract
spatial-motion event semantics — so the `describe` verb earns trust by claim type, not blanket.

## 0. Motivation and thesis fit

The orchestra today has a blind spot with a precise shape. The delta-text stream is excellent at
**where/when** (churn location, scroll, scene change — symbolic, always-on, cheap) and, via OCR, at
**what text**. It is weak at **what is happening** for non-textual motion content: during the
live 3-minute YouTube watch test the watcher could quote OCR'd titles and measure churn, but the
video content itself was just statistics. The eval battery's `game-motion` scenario targets exactly
this gap and currently has no member that could pass it.

Separately measured fact (VJEPA2 session, this machine, 24GB M-series): **Qwen2.5-VL-3B-Instruct
does genuine temporal perception locally** — correct event causality ("the truck suddenly starts
moving forward, causing her to step back"), reads on-screen text, ~6–12s per 6s/16-frame clip on
MPS, ~7GB fp16. The 7B is better (recognizes animation, reads fine text) but costs ~16GB + ~30s —
too heavy to run beside the recorder. Known weaknesses, also measured: sparse frame sampling misses
split-second events; token caps truncate; smaller models hallucinate at the edges; priors can beat
pixels. So the VLM is a **neural claimant, not an oracle** — invariant I1 applies to it exactly as
it applies to the singer.

The architectural insight: a local VLM converts "perceive this clip" from a **context-token cost**
(the scarce resource the whole orchestra economizes) into a **latency cost** (~10s of local GPU).
That changes where deep perception should live: not by pushing more pixels into the watcher's
context (the serialized-tiles idea), but by pulling bounded interpretations from a model that sits
next to the pixels. The tiles experiment remains in the step-6 bake-off as a *research* question
about the watcher's own visuospatial integration; the VLM is the *production* route and becomes
bake-off candidate #7 ("VLM-cursor"), compared at equal total cost.

**The cursor** is the unifying interface this plan formalizes: an agent-controlled attention
pointer `(time-span, region, question-kind)` over the recorded sequence. Its mechanics mostly exist
(`arch_tool show/replay/compare`, `arch-ocr.sh`); what's new is one missing verb (`describe`, the
VLM) and the protocol that makes all four verbs one budgeted instrument:

| cursor verb | member | answers | kind |
|---|---|---|---|
| `show`   | arch_tool + vision-read | what does (t, region) look like, exactly | symbolic decode |
| `ocr`    | arch-ocr.sh (screenocr) | what text is at (t, region) | symbolic-ish |
| `compare`| arch_tool               | did region R change between t1,t2 (count+bbox, exact) | symbolic |
| `describe` | **screenvlm (NEW)**   | what happens during [t0,t1] — motion, causality, events | **neural** |

Three-rate perception, one cursor: always-on symbolic stream for monitoring → exact tools for
text/change verification → VLM for event semantics. Every answer carries an evidence pointer into
the Tier-A ring so any neural claim is re-groundable to exact pixels — and confab-scorable.

## 1. Decisions (made here, with rationale)

1. **Model: Qwen2.5-VL-3B-Instruct, fp16, MPS.** Measured best speed/quality on this machine that
   coexists with the recorder (frame_dump RSS is modest; 3B ≈ 7GB leaves headroom on 24GB). The 7B
   is a config-file swap later; not while recording.
2. **Stateless CLI member (`screenvlm`), like screenocr.** Pixels in → text claim out; no daemon,
   no state. Obeys I3 (members communicate via stream + blackboard only). Per-call model load is
   acceptable for v1 (warm HF cache; measure — expect ~10–20s). An opt-in `--stdin` Q&A loop that
   loads once is a cheap later add for boss sessions; NOT in v1.
3. **Frames come from the Tier-A ring, not from a video file.** `arch_tool show` per selected seq →
   PPM → PIL → numpy stack → `processor(videos=[frames])`. No ffmpeg, no PyAV, no torchvision
   read_video (the exact API that broke in the VJEPA2 session). This also makes every describe
   answer *automatically* evidence-pointed: the input frames ARE archive seqs.
4. **Frame selection: uniform sample of ≤16 seqs across [t0,t1], density a query knob**
   (`--frames N`, cap 32). Downscale to 448-wide before the model (the proven recipe; full-res
   2880×1864 through the vision encoder would blow memory/time for no gain at this task).
5. **Greedy decoding (`do_sample=False`), fixed sampling** → reproducible answers for eval.
6. **Time-addressing lives in the wrapper.** The arch `.idx` is plain text (`seq t_ms kind len w h
   hash`); Python parses it directly to resolve t↔seq. No new arch_tool command (keeps Elisa churn
   zero); revisit only if a third consumer needs span resolution.
7. **Trust policy is an EVAL OUTPUT, not an assumption.** Phase 4 trap-tests screenvlm the way
   counter-skip trap-tested the watcher; the measured confab profile decides how describe answers
   may enter memory (see §5). We do not wire describe into the watcher before that gate.
8. **New invariant I8 (SPEC):** a `describe` answer is interpretation (I1); it must carry the arch
   seq span it actually saw; it may never override a symbolic member's measurement (OCR/compare
   contradicting the VLM wins); and it enters story.md only per the trust policy from the eval.

## 2. Phase 1 — `screenvlm`, the violin (new member)

**Deliverables:** `screenvlm.py`, `setup_vlm.sh`, SPEC "screenvlm CLI contract" section.

CLI contract (mirrors screenocr's spirit):
```
screenvlm <batch_dir> --span t0,t1 [--frames 16] [--region x,y,w,h]
          [--q "question"] [--arch-tool ./arch_tool] [--json]
```
stdout (text mode):
```
ANSWER <the model's answer, single paragraph>
EVIDENCE arch seq <s0>..<s1> t=<t0>..<t1> frames=<n> region=<x,y,w,h|full>
COST load=<s> infer=<s> model=Qwen2.5-VL-3B
```
Exit 0 on success; 2 on bad args; 1 if the span is not (fully) in the live ring — with a partial
answer + `EVIDENCE ... partial=true` if ≥4 frames were recoverable, else no answer.

Implementation steps:
1. `setup_vlm.sh`: `uv venv .venv-vlm --python 3.12` + pin the proven versions (torch 2.13,
   transformers 5.13, accelerate, qwen-vl-utils, pillow, numpy). Venv gitignored. HF cache is
   shared with the VJEPA2 project (same model id ⇒ no re-download).
2. `screenvlm.py`:
   a. Parse all `arch_<seg>.idx` in the dir → `[(seq, t_ms, w, h)]`; select seqs covering the span
      (uniform, ≤ --frames). Fail fast if the span predates the ring (pruned) — distinct message,
      since the boss must know "evidence-exhausted" from "error".
   b. For each seq: `arch_tool show <dir> <seq> <tmp.ppm> [x y w h]` (region crop at source res);
      tolerate a pruned-mid-run seq by substituting the nearest surviving neighbor and marking
      `partial=true` (the ring may advance under us — single writer, but pruning is live).
   c. PIL open → RGB → resize to 448-wide (LANCZOS) → numpy stack.
   d. Load Qwen (the caption_qwen.py recipe verbatim: AutoProcessor +
      AutoModelForImageTextToText, fp16, MPS, greedy, max_new_tokens 160). Default question:
      "Describe what happens in this screen recording clip. Report only what is visible."
   e. Print the contract lines; `--json` emits the same as one object.
3. Smoke test on a real capture: record 30s of the live screen with frame_dump, run
   `screenvlm /tmp/screen_batches --span 5000,11000`, check the answer names actual on-screen
   events; record load/infer times + peak RSS in the plan's measurements section.

Acceptance gate P1: contract output on a live capture; graceful pruned-span behavior; measured
cost numbers written into SPEC.

## 3. Phase 2 — cursor protocol in SPEC + watcher_protocol

**Deliverables:** SPEC §"The cursor" + query-protocol extension + I8; watcher_protocol v3.1.

1. SPEC: add the member row (violin / screenvlm / neural, on-demand), the CLI contract, and the
   cursor table (§0 above) as the canonical statement of active perception. Extend the query
   protocol: `type: temporal | spatial | zoom | describe`; a describe query adds `span: t0,t1` and
   optional `region`, and its budget line now means "≤ N screenvlm invocations". Keep the global
   hard stops (≤3 queries per escalation) unchanged.
2. Evidence-pointer grammar gains the archive form everywhere it's described:
   `[arch seq <s0>..<s1> t=<t0>..<t1>]` alongside `[batch <b> f<i> rows a-b]`.
3. watcher_protocol: the **singer never calls screenvlm** (it's expensive attention — the boss's
   budget). New escalation trigger listed explicitly: "ACTIVITY video/switching batches that are
   goal-relevant but OCR-empty → escalate with a describe-shaped question". Boss playbook: may
   dispatch `describe` like any query; the sub-watcher playbook gains the execution recipe
   (resolve span → run screenvlm → write answer file with status/finding/evidence, marking ALL
   VLM content as Inferred, never Observed).
4. Trust policy placeholder written as: "describe answers are Inferred-only and may not create
   story.md EVENTS until the Phase-4 gate fills in this section" — the doc ships honest.

Acceptance gate P2: SPEC/watcher_protocol updated, internally consistent (cursor table, I8, query
type, evidence grammar), no implementation claims ahead of reality.

## 4. Phase 3 — eval: motion scenes + the VLM trap-test

The counter-skip lesson, applied to the new member *before* it gets trusted. All scenes are
scenegen additions (deterministic, through the real encoder, archive included — screenvlm reads
the fixture's own ring, which is exactly the production path).

**Deliverables:** scenegen scenes `motion`, `motion-trap`; `eval/scenarios/{motion,motion-trap}/`
truth+probes; a `describe`-mode scoring run; measured VLM confab profile; the trust policy.

1. `motion` (straightforward): a bright square translating left→right across the dark field over
   20s, one bounce off the right edge at t=10s. Probes: direction at t=5s (gold "right"), did it
   bounce/reverse (yes), where is it at t=15s (left-half, substring match), how many objects (1).
2. `motion-trap` (priors-vs-pixels): same square, but (a) at t=8s it **vanishes** for 2s mid-flight
   and reappears displaced; (b) at t=14s it reverses direction WITHOUT reaching the edge. A
   continuity-prior model reports smooth motion and an edge-bounce; a perceiving one reports the
   vanish and the causeless reversal. Probes target exactly those: "did the square ever disappear"
   (yes), "did it reverse at an edge or mid-field" (mid-field/substring), "was motion continuous"
   (no).
3. Trap-test protocol: run screenvlm directly (no watcher in the loop) on both fixtures' rings,
   one probe per invocation (`--q`), answers into `answers.jsonl`, score with score_memory.py
   (perception + confab are the relevant metrics; no log.md ⇒ event metrics n/a). 3 repetitions to
   check greedy-decode stability across frame-sampling phase.
4. **The gate decides the trust policy.** Outcomes and their policies, written into SPEC §I8:
   - traps ≥ 75% AND straightforward ≥ 90% → describe answers may propose story.md EVENTS, tagged
     `Inferred(vlm)`, if corroborated by a symbolic signal (churn/compare/OCR) at the same span.
   - traps < 75% but straightforward strong → describe stays a *hypothesis generator*: answers go
     to state.md HYPOTHESES/OPEN_QUESTIONS only; boss must corroborate via compare/ocr before any
     EVENT. (This is the expected outcome for a 3B; fine — still closes the what-is-happening gap.)
   - straightforward weak → violin benched; document numbers; re-audition the 7B offline-only.
5. Sampling-density sensitivity: rerun the vanish probe at `--frames 8/16/32` — the vanish lasts
   2s of a 20s span, so this measures the "split-second event falls between samples" failure mode
   honestly and calibrates the default.

Acceptance gate P3: both scenes render+verify bit-exact; probes scored; a filled-in trust policy
in SPEC backed by numbers; findings appended to eval/scenarios.md.

## 5. Phase 4 — integration dry-run on real content

1. Live run mirroring the 3-minute YouTube watch test: frame_dump + a watcher; when the video
   phase produces the known OCR-empty churn stretch, the watcher escalates; boss (me, manually
   this first time) dispatches one describe query over that span; verify the full loop: escalation
   → query file → screenvlm → answer file → boss consolidation `[arch seq ...]` into log.md.
2. Resource check while recording: peak unified-memory pressure and whether frame pacing degrades
   during inference (frame_dump is real-time; screenvlm is nice-to-have — if pacing suffers, run
   screenvlm at `nice 10`, or accept dropped frames as the documented cost of deep attention).
3. Confab spot-check: pick one describe answer, re-ground it by hand via show/ocr on the same
   span; record the result in eval/scenarios.md findings.

Acceptance gate P4: one end-to-end describe query answered on live data with correct evidence
pointers, resource numbers recorded.

## 6. Phase 5 — bake-off registration (step 6 hookup)

Register **candidate 7: VLM-cursor** in the step-6 bake-off design: watcher context receives only
the text stream + cursor answers (no images), vs candidates 1–6 including sequential tiles, at
equal measured cost (context tokens AND wall-clock reported separately — the VLM's advantage is
precisely that its cost is in a different currency, and the bake-off must show both). No
implementation in this plan beyond the design note in eval/scenarios.md.

## 7. Risks and mitigations

| risk | mitigation |
|---|---|
| Memory pressure: 3B (~7GB) + capture + agents on 24GB | 3B only while recording; measure in P4; 7B offline-only |
| VLM hallucination enters memory | I8 + trust policy gated on measured trap performance; Inferred-only tagging; symbolic override |
| Ring prunes the span mid-query | wrapper tolerates gaps (`partial=true`), distinct evidence-exhausted exit; boss protocol already has that status |
| Sparse sampling misses brief events | `--frames` knob + P3 density sensitivity measurement; default set by data |
| Per-call model load too slow for boss loops | measure first; if painful, add `--stdin` batch mode (load once, N questions) — still stateless per invocation series |
| Full-res frames blow the vision encoder | 448-wide resize in the wrapper (proven recipe); region crops keep native detail when zoomed |
| transformers/torch drift breaks the recipe | pin exact proven versions in setup_vlm.sh; venv isolated |
| Prompt primes the model (counter-skip lesson) | default prompt says "report only what is visible"; probe prompts authored neutral; recorded in scenarios.md |

## 8. Milestones

| # | milestone | acceptance |
|---|---|---|
| M1 | screenvlm answers on a live capture | contract output, cost measured, pruned-span handling |
| M2 | SPEC/watcher_protocol cursor + describe + I8 | docs consistent, trust policy explicitly pending |
| M3 | motion + motion-trap scored | numbers in scenarios.md, trust policy filled in |
| M4 | live end-to-end describe query | evidence-pointed consolidation, resource numbers |
| M5 | bake-off candidate 7 registered | design note only |

Order is strict M1→M3 (M2 can interleave); M4 after M3 (no live integration before the trap gate).

## 9. Non-goals (this plan)

- No daemon/server for the VLM; no streaming perception (the always-on channel stays symbolic).
- No 7B in the live loop; no audio (ASR stays parked, step 8).
- No watcher-initiated screenvlm calls — attention spend stays with the boss.
- No replacement of the text grid: the VLM augments the cursor; the stream remains the backbone.
- Sequential-tiles experiment neither built nor dropped — it is bake-off candidate vs this.
