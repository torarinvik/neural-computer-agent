# Canonical Brain Workshop frontier

This directory is the retained n-back and working-memory research surface.

```text
rendered symbol -> learned amodal event -> fixed controller
                -> external indexed history and compute file
                -> intention -> opaque keypress decoder
                <- trusted scalar outcome event
```

The verifier privately owns the n-back target. The learner receives learned
events, its opaque actions and propensities, external memory, and scalar
outcomes. It never receives the horizon as a semantic task ID or a correct
action label.

## Live acquisition rung

`live_session.py` connects a batch-one Brain Workshop device to the production
cognitive tick runtime. Every stimulus is consumed once, every opaque action
receives an authenticated causal receipt, and each present scalar outcome
causes an optimizer update before the next action. Warm-up actions are closed
with explicit missing evidence rather than a fabricated zero reward.

`live_nback_pilot.py` is the fast virtual-clock diagnostic. On seed 17, the
default sub-minute n-back-1 run measured held-out accuracy of 0.7063, 0.8294,
0.8968, and 0.9603 after 115, 230, 345, and 460 unique training outcome bits.
The first threshold that remained above 0.80 at every later measured prefix
was 230 bits. This is a mechanistic online-learning signal, not a promotion:
the frontend still receives synthetic symbols, and transfer, retention,
multistream composition, autonomous routing, and pixel/audio operation remain
unmeasured.

Run the diagnostic with:

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.live_nback_pilot \
  --report-out /tmp/live-nback-pilot.json
```

`live_executive_route.py` is the next frozen-controller rung. It admits two
verified temporal-equality skills, exposes a public mode cue through the normal
learned event encoder, and lets a memory-side router learn which immutable slot
to run. The default 16-lifetime run uses 136 unique training verifier bits and
then scores 1.0000 on three held-out 1-back and three held-out 2-back
lifetimes. The controller,
decoder, and executive programs remain frozen; route evidence is the only
mutable state. Route mastery uses one aggregate outcome per lifetime by
default, while action outcomes remain available to the ordinary runtime.

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.live_executive_route \
  --report-out /tmp/live-executive-route.json
```

## Rendered vision and audio rung

`rendered_environment.py` is a clean-room device surface. Its learner-visible
observations contain only RGB frames and audio waveforms; sequence symbols,
match flags, depth, and correct actions remain private. Independent replaceable
frontends emit normalized learned events with opaque learnable source keys.

`rendered_live.py` maintains a separate history for every source key. One
shared temporal reader processes all sources, and a source-conditioned shared
transform runs before permutation-invariant composition. Reversing the order
of simultaneous events therefore does not change execution.

The default seed-17 sub-minute pilot produced:

| Live condition | 115 bits | 230 bits | 460 bits | 920 bits | Stable >= 0.80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| rendered vision | 0.7103 | 0.8294 | 0.8849 | 0.9881 | 230 bits |
| rendered audio | 0.8611 | 0.9444 | 0.9643 | 0.9960 | 115 bits |
| exact dual action | 0.2540 | 0.3333 | 0.4048 | 0.9246 | 920 bits |

On a separate dual control prefix, ordinary accuracy was 0.9087. Reversing
event order was exactly unchanged; resetting history each tick fell to 0.2500;
missing vision fell to 0.5278; and missing audio fell to 0.4802. Training used
one scalar exact-action outcome per trial, batch size one, immediate updates,
and zero replay. This remains a mechanistic diagnostic rather than a promoted
claim: retention and transfer against a fresh learner are not yet measured.

Run it with:

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.rendered_live_pilot \
  --report-out /tmp/rendered-live-pilot.json
```

## Physical human-parity rung

`physical_live.py` connects the same runtime to the public Brain Workshop 5
interface. The learner receives only display-captured RGB pixels,
acts through the ordinary visible position-match key, and learns from the
green/red/blue feedback rendered to a human. No source import, stats file,
target flag, correct action, trial ID, or synthetic reward enters the runtime.
Every scalar outcome carries the SHA-256 digests of its complete public screen
evidence window; `--evidence-dir` archives the exact frames behind those
digests. Explicit green is positive and explicit red/blue is negative under
the application's default scoring. A neutral true-negative window closes as
absent evidence, never as a fabricated reward or failure.

The calibrated 12 Hz control detected all 12 stimulus onsets and captured both
positive and negative feedback. The first ten-session campaign validated this
I/O path, but it updated controller weights and is architecture-invalid for task
acquisition. Its evidence remains only as an I/O calibration record in
`session_records/brainworkshop_physical_live_learning_rejected_2026-08-14/`.

`PretrainedControllerProgramMachine` enforces the intended ownership boundary.
Its relation, source-conditioning, and intention-decoding weights come from a
controller pretrained across independently projected visual frontend families.
They are frozen during Brain Workshop. A fresh task file begins with a uniform
categorical temporal address, executes one logged address per decision, and is
the only optimizer target. Copying the pretraining run's learned address is an
explicit `--inherit-program-prior` transfer control and is excluded from task-
learning claims. Campaign reports record controller and program digests
separately.

`physical_program_bank.py` completes the lifecycle after a campaign. It pools
only the standardized learned event payloads saved in the physical session
reports, evaluates the provisional address file from public per-lifetime
accuracy, and admits it to `ExternalTemporalProgramBank` only after a stable
prefix. The admitted file contains the address logits but no optimizer state.
It is checksummed, bound to the frozen controller digest, and selected by an
external opaque-context reward ledger with exact behavior propensities.
Rejected admissions leave the live bank digest unchanged. Loading a selected
file forces deterministic read-only execution.

The live retrieval runner uses the generic variable-port `INPUT` boundary.
Public feedback is resolved from its causal action receipt and delivered during
the same tick to `TemporalProgramOutcomeObserver`; there is no end-of-episode
reward handoff. A disposable-bank physical validation received five reward
inputs, scored 4/5, and changed only the external route ledger. Controller and
program updates remained zero.

For the first curriculum axis, `tools/brainworkshop_position_cells.patch`
adds public `BRAINWORKSHOP_POSITION_CELLS`, `BRAINWORKSHOP_TRIALS`, and
`BRAINWORKSHOP_MUTE_MUSIC` settings to upstream Brain Workshop.
It preserves Position N-Back timing, input, and scoring while sampling from a
center-out prefixes containing 2 through 8 visible grid cells. The mode label displays
the active cell and trial counts. Cell count is an environment/frontend difficulty setting,
not a controller input or semantic task ID. The mute setting disables background
music only; task-relevant audio stimuli, sound effects, and scoring are unchanged.
The trial override sets an exact public session length by disabling the usual
n-back-dependent trial-count factor. The current live setting is 60 trials.

Apply the curriculum patch to an upstream checkout, then launch the 2-cell
environment:

```bash
git apply /absolute/path/to/tools/brainworkshop_position_cells.patch
BRAINWORKSHOP_POSITION_CELLS=2 BRAINWORKSHOP_TRIALS=60 \
  BRAINWORKSHOP_MUTE_MUSIC=1 \
  python brainworkshop.py
```

Build and run the bounded frozen-controller campaign while the Position N-Back
window is frontmost and on its ready screen:

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.physical_train_pilot \
  --sessions 10 --seconds-per-session 64 --tick-hz 12 \
  --archive-evidence --output-dir /tmp/brainworkshop-program-training
```

The command writes the verified bank to
`/tmp/brainworkshop-program-training/AgentBrain.bank` after eight stable public
lifetimes; shorter pilots produce an admission-rejection receipt without
altering a live bank. The first fresh lifecycle replication used a three-
lifetime gate: the initial three-cell curriculum jump failed admission, while
the two-cell learner reached lifetime scores `0.50, 0.67, 1.00, 1.00, 1.00,
1.00` and committed slot 0 after 37 outcomes. Four fresh read-only sessions
then used `physical_bank_transfer_pilot.py` to withhold actions for three live
events, select the bank artifact, and execute it with frozen weights. Public
feedback was 13/15, with 15 route observations, zero controller/program
updates, and zero replay. The production live router now provides a bounded
multi-program rule-selection test under the same constraint: it sees a learned
event derived from the public mode cue, never an `n_back` or task ID. Open-ended
cue discovery, autonomous program admission, and larger physical GUI curricula
remain separate tests.

Run the read-only bank path with:

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.physical_bank_transfer_pilot \
  --bank /absolute/path/to/AgentBrain.bank --warmup-events 3
```

Run a resumable curriculum rung that checkpoints route evidence after every
fresh GUI lifetime with:

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.physical_bank_curriculum_pilot \
  --bank /absolute/path/to/AgentBrain.bank --sessions 12 \
  --seconds-per-session 64 --output-dir /absolute/path/to/evidence
```

Use a transactional copy of the canonical bank for a new difficulty rung and
replace the curated file only after `retention_gate_passed` is true. Admitted
program tensors remain immutable; this runner learns only reward-attributed
opaque routing evidence.

The promoted blank-program 2-cell campaign used 86 public outcomes and 86
program updates with zero controller updates and zero replay across 14 physical
lifetimes. Its first full rolling-44 score was `0.8864`, every later prefix
remained above `0.80`, and its final rolling score was `1.0000`. All 168 stimulus
onsets and actions were preserved. A retention session measured live tick
latency at 45.3 ms p50 and 116.9 ms p99. Across 32 rendered seeds, normal blank-
program learning reached the stable gate 32/32 times; blank-frozen, reward-
shuffled, action-reversed, and missing-history controls reached it 0/32 times.
The evidence is in
`session_records/brainworkshop_physical_blank_program_promoted_2026-08-14/`.
Read-only transfer from two to three visible cells then replicated for two
12-trial sessions: all three event clusters appeared, all 24 decisions matched
the visible sequence, and Brain Workshop scored 100 twice with unchanged
controller and program digests. This remains sub-minute probation pending a
roughly three-minute three-cell retention run; see
`session_records/brainworkshop_physical_3cell_transfer_probation_2026-08-14/`.
Dual N-Back remains blocked on a human-parity audio input: this Mac currently
exposes only its microphone, not a clean system-audio loopback. Until that
device is present, a screen-only run must be labeled Position N-Back rather
than dual.

## Promoted frontier

`executive_compositional_transfer.py` is the first positive bits-to-threshold
transfer result for the new persistent executive. A verified one-step
temporal-equality skeleton narrows a held-out 2-back search to the smallest
failed binding: relative delay. Across seed blocks 17-32 and 101-116, warm and
fresh learners admitted the same executable solution on all 32 seeds, but warm
search used 30,208 target verifier bits versus 84,992 fresh bits, a 2.8136
fresh/warm transfer ratio. Warm was strictly faster in every seed. Irrelevant
inheritance, destroyed reward, shuffled actions, and missing history admitted
nothing; source retention was perfect with zero optimizer updates and replay.
Evidence is in
`session_records/brainworkshop_executive_compositional_transfer_promoted_2026-08-14/`.

The source and target are also admitted into restart-safe `AgentBrain*.bank`
files with complete allow-listed operator manifests. Freshly reconstructed
operators retain 1.0 source and target behavior after reload. The canonical
`ExternalAgentBrainBank` now makes this a heterogeneous container: executive
skills and legacy temporal-address route banks can coexist under one controller
digest. Legacy torch banks are imported explicitly and retain their opaque
route evidence. This remains a bounded diagnostic because the candidate
library is externally generated.

The promoted run also composes a freshly verified receive-only fragment with the
verified 1-back temporal loop, admits the child through the same stable-prefix
verifier gate, persists parent slots and digests, and reloads the composed child
at 1.0 source mastery. Non-final parents must have a reachable terminal handoff,
so a persistent loop cannot shadow later components. Composition is structural
and controller-frozen; it is not yet unrestricted autonomous program induction.
Parent selection is now an opaque deterministic ordered-pair search with staged
fresh verification. A clearly sub-threshold first rollout rejects immediately;
a promising candidate receives a fresh confirmation. The first stable child is
appended with explicit unique-bit, lifetime, stage, and replay accounting.

`external_compute_append_only_depth_growth.py` is the current bounded working-
memory result. A frozen source file masters n-back-16, then a fresh external
file learns n-back-32 while the source, controller, and event frontend remain
unchanged.

Across seeds 17 and 18:

| Measurement | Seed 17 | Seed 18 |
| --- | ---: | ---: |
| n-back-16 before extension | 1.0000 | 0.9992 |
| n-back-32 new file | 1.0000 | 1.0000 |
| n-back-16 retention | 1.0000 | 0.9992 |

Missing/corrupted history, shuffled actions, and fresh shuffled-outcome
training all failed the 0.80 mastery gate. Each seed used 512 learned-file
optimizer updates, 163,840 verifier bits, and zero replay.

Run it with:

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.external_compute_append_only_depth_growth
```

Evidence:
`session_records/brainworkshop_append_only_nback32_depth_promoted_2026-08-13/`.

## Current bottleneck

The external compute/artifact lifecycle is reliable, but inherited eviction
knowledge has not transferred to a held-out n-back family. The current active
audit tests a neutral probationary fallback. Until it passes replicated fresh
controls, the correct policy is to retain the architecture and reset inherited
maintenance weights.

The next n-back campaign must measure stable bits-to-threshold against a matched
fresh learner. Raising final accuracy alone is not the objective.
