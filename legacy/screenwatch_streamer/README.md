# Elisa Screen-Understanding Orchestra

Turns the live macOS screen into a stream an LLM can *watch*: a symbolic pipeline (Elisa + Swift)
measures what changed and a small hierarchy of LLM agents interprets it. The design is an
"orchestra" — cheap deterministic members do everything measurable; neural members only interpret,
and each tier wakes the next on signal (see `SPEC.md`, the single source of truth for all
interfaces).

## How it works

`frame_dump` (Elisa, ScreenCaptureKit) captures the display and emits one **delta-encoded text
batch** every few seconds: a full ASCII+color keyframe, then per-frame deltas (`SAME`, `SHIFT dy=n`
for scrolls, `ROWS a-b` for changed rows), plus a `STATS` line and a symbolic `ACTIVITY` triage
label (`still|typing|scrolling|video|switching`). A calm batch is ~15–20K tokens at 192-cell width —
roughly 7× cheaper than full-frame dumps at 4× the resolution — and every frame is losslessly
reconstructable. Each batch also gets a **full-res JPEG keyframe sidecar**, and `screenocr`
(Swift/Vision) provides positioned text on demand, including `--crop` zooms for active perception.

On top of the stream sit the neural members (`watcher_protocol.md`): a cheap **watcher** agent
maintains a **3-tier structured memory** — a ≤2 KB working set (`state.md`: typed OBSERVATIONS /
OBJECTS / RELATIONS / HYPOTHESES, with stable object ids), an append-only episodic trace (`log.md`),
and a consolidated semantic story (`story.md`) that a slow **consolidator** pass folds aged episodes
into. Every tier forgets *representation* under a hard cap but keeps *evidence* pointers back into the
batch stream (and, before pruning, into the exact Tier-A archive). A **boss** agent, woken only by
escalations, dispatches budgeted zoom/OCR queries. Short-lived watchers inherit this external memory
and observe an unbounded stream while total memory stays compact — the point of the whole design.

## Files

| File | Role |
|---|---|
| `frame_dump.elisa` | live recorder: ScreenCaptureKit source + main (drives the encoder) |
| `encoder.elisa` | the delta encoder + batch serializer + triage, shared by the recorder and scenegen |
| `scenegen.elisa` | deterministic synthetic scene source → real encoder (eval fixtures) |
| `screencap.elisa` | ScreenCaptureKit bridge in pure Elisa (Obj-C runtime over FFI) |
| `archive.elisa` | Tier A exact-frame ring: XOR-delta + LZFSE, checksummed, byte-capped |
| `arch_tool.elisa` | archive verifier + query engine: `verify` / `show` / `replay` / `compare` |
| `arch-ocr.sh` | OCR an exact archived frame (resolve an evidence pin to positioned text) |
| `tracker.elisa` | the **viola**: symbolic object-identity tracker over the delta stream (I9 OBS/INF) |
| `audiocap.swift` | system-audio capture → 16 kHz mono PCM ring (`aud_*.pcm`) — live-verify pending |
| `audiogen.elisa` | deterministic audio-scene synthesizer → WAV (the audio twin of scenegen) |
| `audiotriage.elisa` | the **cymbal**: symbolic audio triage (512-pt FFT) → TRANSIENT/SILENCE/TONE/LEVEL_SHIFT |
| `screenvlm.py` | the **violin**: local video-VLM (Qwen2.5-VL-3B) — `describe` cursor verb |
| `screenaud.py` | the **sax**: local audio-LM (MiDashengLM-0.6B) — audition blocked on transformers version |
| `screenasr.swift` | system-audio speech transcription (SpeechAnalyzer) — parked, ready |
| `screenocr.swift` | Vision OCR CLI: positioned text, `--crop` region zoom |
| `ocr_watch.sh` | eager OCR trigger on scene-change batches |
| `SPEC.md` | system contracts: members, stream format v2, blackboard layout |
| `watcher_protocol.md` | watcher/boss agent protocol (external memory, escalation) |
| `eval/` | scoring harness: `score.py` (triage), `score_memory.py` (watcher memory), `trap_test.sh` (violin), `track_test.sh` (viola), `audio_test.sh` (cymbal) + `scenarios.md` |
| `experiments/sensory_codec/` | isolated ten-game visual/audio/text sensory-bottleneck and frozen-SmolVLM2 experiment; not admitted to the trusted ledger |
| `experiments/event_stream_snake/` | isolated frozen-LLM audiovisual Snake experiment with dense, fixed, and learned event emission; not admitted to the trusted ledger |
| `experiments/event_stream_reflex/` | isolated immediate-reward audiovisual reflex arena with frozen SmolVLM2 and sparse event gates; not admitted to the trusted ledger |
| `experiments/continuous_reflex/` | noisy held-out audiovisual reflex windows with dense, fixed, learned, and random-budget stream controls; not admitted to the trusted ledger |
| `experiments/forward_transfer_attention/` | sensory-only few-shot attention transfer through latent memory, with causal corruption controls and transactional compression audits |

## Build & run

Requires the Elisa compiler (`elisacore`) and the Screen Recording TCC permission (ScreenCaptureKit
prompts on first run).

```sh
elisacore build frame-dump --project .
./frame_dump [out_dir] [width=192] [fps=24] [n_seconds=3] [token_cap=40000] [retain=40] [imgs=1] [arch_mb=2048]

elisacore build arch-tool --project .
./arch_tool verify /tmp/screen_batches            # prove the exact ring is bit-exact
./arch_tool show    /tmp/screen_batches 20 f.ppm  # decode frame 20 (add "x y w h" to crop)
./arch_tool replay  /tmp/screen_batches 20 60 5 rp_   # lossless replay 20..60 step 5
./arch_tool compare /tmp/screen_batches 20 40 d.ppm   # exact pixel diff: changed count + bbox
./arch-ocr.sh       /tmp/screen_batches 20 0,0,600,120  # OCR an exact archived frame

swiftc -O screenocr.swift -o screenocr
./ocr_watch.sh /tmp/screen_batches ./screenocr   # eager OCR loop (optional)

elisacore build scenegen --project .              # deterministic eval fixtures (no screen perms)
./scenegen counter /tmp/run 192 10 20000          # render a scenario through the real encoder
python3 eval/score_memory.py /tmp/run --arch-tool ./arch_tool   # after a watcher fills answers.jsonl
```

Batches land in `/tmp/screen_batches/` (`batch_<n>.txt` + `batch_<n>.jpg`, `latest.txt` pointer,
pruned to the retention window); the agent blackboard lives in `/tmp/screen_watch/`. Point a watcher
agent at `watcher_protocol.md` to start observing.

## Verifying

Two levels of scoring. `eval/score.py` grades the **symbolic** triage: it aligns a scripted session's
phases (`eval/session_terminal.command`) against the recorded batches and checks the ACTIVITY labels.
`eval/score_memory.py` grades the **neural** watcher: against a scenario's authored ground truth
(`eval/scenarios/<name>/{truth,probes}.jsonl`) it scores perception accuracy, memory retention,
event order, and — the metric that matters most — confabulation rate, plus tokens/latency. Honest
"unknown" answers are misses but never confabulations, so the harness rewards a watcher that declines
over one that guesses. `python3 eval/score_memory.py --selftest` checks the metric math. See
`eval/scenarios.md` for the 8-scenario battery. Reconstruction fidelity of the delta stream is
lossless by construction (churn re-keyframes; dropped frames never update the baseline) and of the
archive is a tested property (`arch_tool verify`).

## Neural-computer active context

`experiments/syllogimous_neural_computer/` contains an isolated persistent-memory controller and a
learned context selector. Long-term latent rows can be copied into a compact active store in
RAM/VRAM with `PersistentMemory.select`; the controller receives only selected latent rows, never
symbolic game state. The selector sees a recurrent sensory query plus latent keys/values, and may
choose no row or one row. A conservative calibration pass enables the smaller context only when
both query and audit accuracy and cross-entropy are at least as good as the full store on two
held-out splits.

The transfer and boundary behavior are covered by the neural-computer tests. The current
frozen-controller result is an important negative result: the selector beats random on some seeds,
but does not yet reliably identify the useful row. The larger replication therefore falls back to
the full eight-row store (zero regression, zero savings). The oracle singleton gap remains large,
showing that the task is learnable and that the next bottleneck is representation/credit assignment,
not the memory-transfer mechanism itself.

## Experimental discipline

Neural-computer experiments follow a probes-first ladder. Decompose each capability into
deterministically labeled sub-facts; use disposable supervised probes to establish what is present
on both sides of a suspected boundary; then make the smallest generic architectural change whose
effect has a probe-verifiable acceptance criterion. Diagnostic weights are discarded. Capability
claims require a fresh model trained only through sensory input and behavioral outcomes, followed
by causal interventions such as memory corruption, stream reversal, and retention audits.

Fixed-budget failures are always reported as “no learning within N updates,” never as proof that an
architecture cannot learn. Negatives that control an architectural decision require matched
positive controls and wider budgets. Correctness is primary; retained old capabilities come next;
chance-normalized early-learning speed and response latency are measured separately so a fast
shortcut cannot outrank a slower correct solution. Every environmental answer remains private to
the verifier—the controller receives only the visual/audio stream and its own learned memory.

## History

This repo began as a "comic book" recorder (screen → PNG contact sheets of downscaled panels, the
Wolf3D comic-capture pipeline generalized off SDL). The delta-encoded text stream + keyframe-sidecar
system replaced it; the comic pipeline was removed (see git history — the living copy of
`comic_capture.elisa` remains in the elisa-wolf3d repo).
