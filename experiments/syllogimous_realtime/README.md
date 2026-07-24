# Real-time audiovisual Syllogimous

An isolated pixels/PCM-only relational-reasoning environment inspired by
Syllogimous v3. See [CONTRACT.md](CONTRACT.md) for the human-equivalent timing and
agent-boundary rules.

Current milestone:

- native, deterministic Elisa generators and independent solvers for
  distinction, comparison, temporal, 2D/3D/4D direction, categorical
  syllogisms, relation analogies, nested Boolean expressions, and WCST-style
  sorting using the original card vocabularies;
- a stream-driven `machine over` Boolean evaluator, checked across 512 seeds;
- semantics-preserving negated surface forms and configurable Boolean operators;
- explicit meta-relation dependencies plus typed paradox/logic rare overrides;
- the exact 446-word meaningful vocabulary and exact 1,530-label abstract
  vocabulary, with persistent Stroop/button presentation state;
- an answer-free public-view boundary and pull-based render-card stream for
  relational, categorical, Boolean, analogy, sorting, and fixed puzzles;
- protected final-evaluation seeds;
- sequential premise cards and conclusion;
- real wall-clock deadline that includes inference;
- raw RGB frames and continuous 16 kHz PCM packets;
- `WAIT/NEXT/PREVIOUS/TRUE/FALSE` actions;
- correctness-dominant reward with at most a 0.05 speed bonus.
- a clean-room typed-family Elisa rasterizer in `syllogimous_raster.elisa`,
  exercised by `syllogimous_render_smoke.elisa`.
- an answer-free synchronized sensory API in `syllogimous_sensory.elisa`; its
  deterministic dual-tone audio vocabulary is synthesized with `machine over`
  from public render cards and never from evaluator state;
- typed cross-round score, millisecond response records, and terminal feedback
  in `syllogimous_ledger.elisa`, with exact lifetime totals and a bounded rolling
  history that does not cap training duration;
- a strict text-output action parser in `syllogimous_actions.elisa`, including
  carousel backtracking and sorting moves, so models never receive callable game
  hooks;
- a real-time host coordinator in `syllogimous_driver.elisa` and public scene
  renderer in `syllogimous_scene.elisa`, covering carousel backtracking,
  display-all layouts, player-ordered sorting, timer decay, polling timeouts, and
  terminal sensory feedback;
- a cross-round lifecycle in `syllogimous_runtime.elisa` that owns selection,
  generation, scoring, the exact 1200 ms feedback interval, and automatic
  preparation of the next round while preserving the audiovisual-only boundary;
- a native `syllogimous_host` process whose stdin is newline-delimited model
  actions and whose stdout is timestamped RGB/PCM envelopes only;
- exact visible text for all twelve upstream paradoxes and all twelve logic
  puzzles, rendered through the same answer-free card stream.
- the upstream negation explainer as an actual first carousel card and matching
  audiovisual display-all instruction, rather than dormant configuration state.

For research comparison, the original Syllogimous v3 repository is cloned next
to this project as `../Syllogimous-v3-upstream` and pinned at commit
`01238d0b1a9b508257e6b5580063b1f76ad3eeb3`. Its CC BY-NC 3.0 license makes this
experiment noncommercial unless separate permission is obtained. No upstream
code is imported by the model-facing environment.

The generator/evaluator has private logical state. Models receive only
`SensoryPacket(frame, pcm, timestamp_ms)`.

```sh
.venv-vlm/bin/python -m unittest \
  experiments.syllogimous_realtime.test_environment -v
```

The matching stage0 compiler has been rebuilt and installed from the current
Elisa-core working tree, including the large error-union/temporary-storage fixes
needed by the full typed session. The Python reference environment and its tests
run independently.

The model-size slots are evaluated by `run_model_matrix.py`.  Unconfigured slots
are recorded as `unassigned`; configured slots run the same packet-only causal
episode protocol, with action traces and grouped metrics.  This prevents a
nominal “1B” label from being mistaken for an installed checkpoint:

```sh
python3 -m experiments.syllogimous_realtime.run_model_matrix \
  --models experiments/syllogimous_realtime/model_matrix.example.json \
  --local-files-only --device mps --episodes 1 \
  --output experiments/syllogimous_realtime/model_matrix.json
```

Registry entries may include `actual_parameters` and `modality`; the result
records whether the checkpoint size matches its nominal slot. The current
matrix therefore reports the cached 256M and 500M models explicitly, while the
1M–100M slots remain unassigned rather than being populated with incompatible
text-only checkpoints.

## Validation and experiment tooling

`challenge_reference.py` is an independent Python implementation of all 90
challenge semantics. `validate_challenges.py` runs deterministic seed sweeps,
checks public-answer recomputation and family coverage, and writes a JSON
manifest suitable for later comparison. Each generated challenge now records
realized value/premise length, Boolean nesting depth, distractor count, and
multimodal interference metadata; the validator reports histograms for all four
dimensions. For example:

```sh
python3 experiments/syllogimous_realtime/validate_challenges.py \
  --count 1000000 --difficulty max \
  --report experiments/syllogimous_realtime/challenge_validation_max_1m.json
```

The max-profile five-million-seed sweep currently has zero failures and covers
all 90 families; the one-million report remains available as a faster smoke
check. `evaluation.py`, `baselines.py`, `streamer_experiments.py`, and
`model_loop.py` provide common metrics, baseline names, the frozen-LLM adapter
matrix, and the causal timestamped stream/action loop. `repro_manifest.py`
records compiler, commit, hardware, model, configuration, and seed provenance.

`validate_native_challenges.py` independently cross-checks native Elisa output
against the Python solver. The current 10,000-record cross-check has zero
mismatches; it caught and corrected RNG-consumption and negative-division bugs
that a self-consistent Python sweep could not detect.

`boundary_leak.py` performs a black-box differential check: it holds all visible
cards constant, flips hidden answer/family fields, and verifies identical RGB
and PCM packets across a causal action sequence. The current 256-seed report
has zero divergent packets.

The native transport can be exercised without a model checkpoint:

```sh
python3 - <<'PY'
from experiments.syllogimous_realtime.host_client import run_host
events = run_host(["./experiments/syllogimous_realtime/syllogimous_host"],
                  lambda packet: "WAIT", max_packets=1)
print([(event.kind, event.action) for event in events])
PY
```

`adapter_rl.py` contains the frozen-listener policy-gradient gate objective;
`streamer_experiments.py` enumerates dense/fixed/random/learned gates and RGB,
audio, sparsity, and threshold ablations. The adapter receives only sensory
tensors and rollout reward, while the listener parameters remain frozen. Its
objective includes a small minimum-coverage term, so suppressing every packet
cannot beat emitting at least one modality; task correctness remains weighted
1.0 and latency/coverage are secondary terms.

`run_adapter_matrix.py` applies those four gates to the same frozen VLM episode
runner. It records per-variant accuracy, timeout, latency, emitted sensory
tokens, and action traces; `--adapter-checkpoint` enables the learned-gate row.
The fixed and random rows are controls, and suppressed packets become causal
`WAIT` steps rather than being silently removed from the clock.

The checked-in 256M smoke matrix is intentionally a diagnostic: all four
variants timed out on the local MPS run, while the learned gate suppressed all
model calls. This is evidence about the current latency bottleneck, not a claim
of learned gameplay competence.

`adapter_256m_coverage_smoke.pt` and its metrics are the corrected one-episode
training smoke: the anti-silence objective caused the gate to emit a nonzero
mean of 0.67 modalities per packet, but the frozen VLM still timed out. It is a
training-path check, not a competence result.

The first H100 20-episode adapter run is recorded in
`adapter_256m_h100_20ep.json` and its checkpoint. All 20 episodes timed out,
with roughly one emitted modality per packet; the corresponding H100 gate matrix
is in `adapter_matrix_h100_256m.json`. This confirms that the adapter learns
non-silent stream decisions, but does not yet improve task success.

`run_baselines.py` executes the five named baselines against the causal Python
environment and writes metrics in `baseline_metrics/`. The vision-only,
vision-plus-audio, and full-stream rows are deliberately simple packet-only
controls, not semantic OCR/VLM quality claims. Real checkpoint results belong
in the model matrix and include the actual model id, hardware, action traces,
and any unavailable/unassigned slots.

With a locally available checkpoint, `run_vlm_host.py` connects a Transformers
VLM to the native host:

```sh
python3 -m experiments.syllogimous_realtime.run_vlm_host \
  --model <local-or-hub-model> --local-files-only --packets 100 \
  --log experiments/syllogimous_realtime/vlm_run.jsonl
```

The model callback receives only a PIL image derived from the RGB packet (or
PCM for `--audio-only`); the host process remains the sole owner of game state.
Each transport log labels the synchronized input as both `frame` and `audio`
and records their boundary timestamps separately (`frame_received_ns` and
`audio_received_ns`), alongside inference completion and action timestamps.

Cached SmolVLM2 smoke runs are recorded in
`model_latency_results.json`: the 256M and 500M checkpoints both emitted a
valid action through the native host after chat-template decoding was fixed.
The corrected MPS measurements are approximately 3.8 s and 4.9 s per packet;
the earlier CPU rows are retained only as superseded diagnostics.

The first H100 NVL comparison is in `cloud_model_results.json`. Cloud inference
reduced steady-state packet latency to roughly 0.2–0.5 s, but the 256M and 500M
models still failed to reach a valid answer before the deadline; the 2.2B model
reached the conclusion in time but chose the wrong answer. This separates the
hardware bottleneck from the remaining visual-reasoning/prompting problem.

`run_vlm_episode.py` scores a real policy against the causal Python episode.
The first 256M/MPS two-premise smoke episode timed out at 12.3 s total
inference latency, demonstrating that KV-cache reuse and a faster GPU are
needed before interpreting model accuracy.

`event_gate.py` implements causal dense/fixed/random gating from RGB change and
PCM energy only. `run_vlm_host.py` exposes `--gate`, `--frame-threshold`,
`--audio-rms-threshold`, and `--audio-silence-ms`; suppressed packets are
logged and never reach the model. A native-host gate smoke test is included in
the tooling suite.

`run_gate_ablation.py` executes the gate matrix and writes delivered/suppressed
packet counts, inference-call counts, and suppression rates. The checked-in
`gate_ablation.json` contains 36 rows across four gate modes, three RGB
thresholds, and three audio silence windows; its transport-control note keeps
these results separate from gameplay accuracy.

`reproduce.sh` reruns the semantic audit, native/Python cross-check, million-
seed validation, gate matrix, and Python test suites. The generated
`experiment_manifest.json` indexes the resulting reports and checkpoints and
pins both the compiler source revision and compiler-binary hash.

`train_adapter.py` performs the actual frozen-VLM policy-gradient update and
exports a versioned gate checkpoint. The checked-in 256M/MPS smoke checkpoint
and metrics validate the full path; the corrected one-episode run still timed
out at 11.3 s, so it is a training-path validation rather than a performance
result.
