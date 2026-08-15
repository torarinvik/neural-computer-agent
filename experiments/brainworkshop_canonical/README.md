# Canonical n-back frontier

Neural Workshop is the only training gym. This directory name is historical.
Do not check out or patch a separate Brain Workshop. Point `--neural-workshop`
at the Neural Workshop repo. Cell count, trial count, n-back, game mode, and
mute are constructor knobs on `NeuralWorkshopEnv`, not environment variables
or a local patch.

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

`live_session.py` connects a batch-one Neural Workshop device to the production
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

`live_executive_route_reversal.py` is the nonstationary follow-up. It reloads
the checksummed bank, changes the private verifier rule behind the same visible
cue, demotes the failing route after two aggregate lifetimes, and keeps the
successful replacement until its stable-prefix gate is met. The old skill is
then forced directly to verify retention, while a cue-shuffled control checks
that the learned route is not a generic slot preference.

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.live_executive_route_reversal \
  --report-out /tmp/live-executive-route-reversal.json
```

`live_executive_route_generalization.py` tests the held-out-context path after
reload. It calibrates two exact learned cue keys, then presents six unseen
nearby event representations. The router must generalize through the nearest
protected opaque context before recording each variant independently.

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.live_executive_route_generalization \
  --report-out /tmp/live-executive-route-generalization.json
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

## Optional physical I/O against Neural Workshop

These runners are not a second game. They capture Neural Workshop's own public
window (`brainworkshop.py`, title contains "Neural Workshop") and inject
ordinary keypresses. Training, curriculum, Dual acquisition, and promotion
use the gym constructors below. Desktop capture is human-parity I/O only.

`physical_live.py` connects the same runtime to that public window. The
learner receives only display-captured RGB pixels,
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
They are frozen during the physical I/O campaign. A fresh task file begins with a uniform
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

Cell count is an environment/frontend difficulty setting, not a controller
input. There is no local Brain Workshop patch in this repo.

Train Dual on the Neural Workshop gym (pixels + public PCM), not by
screen-capturing a desktop window:

```bash
NW_HEADLESS=0 PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.neural_workshop_dual_acquisition_pilot \
  --neural-workshop /absolute/path/to/neural-workshop \
  --trials 20 --sessions 3 --visible
```

Optional I/O only: run the bounded frozen-controller campaign while Neural
Workshop's Position N-Back window is frontmost and on its ready screen:

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
the visible sequence, and Neural Workshop scored 100 twice with unchanged
controller and program digests. This remains sub-minute probation pending a
roughly three-minute three-cell retention run; see
`session_records/brainworkshop_physical_3cell_transfer_probation_2026-08-14/`.
Desktop Dual now has a ScreenCaptureKit window tap and a Dual lifetime
runner. Missing or silent Dual audio fails closed. Mixed green+red or
green+blue labels are packed exact-match zero. macOS must allow Screen
Recording and Accessibility for this process. The Dual window must be
frontmost.

```bash
# Confirm the tap. Ready-screen silence is allowed; the stream must be active.
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.physical_dual_live \
  --mode probe --seconds 8 --prepare-nback 1

# Watch frozen Dual 1-back execute, then blank-file learn, then composed 2-back.
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.physical_dual_live \
  --mode execute --n-back 1 --seconds 25 --start-session --prepare-nback 1

PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.physical_dual_live \
  --mode learn --n-back 1 --seconds 45 --sessions 1 --then-compose-2back
```

A screen-only capture is still Position N-Back. These desktop Dual
sessions are optional I/O, not a second trainer and not a holdout
promotion. Measured Dual public-PCM training is the Neural Workshop gym.

## Neural Workshop live curriculum

`neural_workshop_live.py` replaces slow macOS capture and key injection with
Neural Workshop's headless public boundary. It still feeds rendered RGBA pixels
through a frozen visual adapter, maps opaque decoder actions to one or two
public ports, authenticates every visible scalar against the environment's
immutable frame archive and receipt ledger, and drains correct-rejection
silence as absent evidence. Dual also encodes the public stimulus waveform
as a second amodal event. Signed public scalars remain in the audit report;
only the learner input maps `[-1, +1]` to Bernoulli credit `[0, 1]`.

The resumable curriculum fixes Position 1-Back and the visible 3x3 board while
changing only active cells `2 -> 3 -> 4 -> 6 -> 8`. Promotion requires three
consecutive 60-trial sessions at or above the threshold, a minimum evidence
count, and a frozen deterministic retention session. Each admitted program is
written to `AgentBrain.bank`; later rungs continue from the preceding external
program while a matched fresh-program control measures transfer.

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.neural_workshop_curriculum_pilot \
  --neural-workshop /absolute/path/to/neural-workshop \
  --output-dir /tmp/neural-workshop-position1back \
  --trials 60 --cells 2 3 4 6 8
```

Use `--resume` to continue an incomplete evidence gate. Generated reports,
checkpoints, and banks belong in the output directory, not in Git unless a run
is deliberately curated with a manifest checksum.

`neural_workshop_nback_transfer_pilot.py` holds the rendered interface fixed,
loads the final 1-back program, and compares inherited versus uniform-fresh
2-back acquisition on identical seeds. Its discarded interventions preserve
both sides of every audit boundary: reports distinguish proposed from executed
actions and authenticated verifier rewards from learner-visible rewards.
Passive, random-action, fixed reversal, temporal-memory corruption,
reward-shuffled, missing-evidence, and action-shuffled arms cannot alter the
primary program or bank.

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.neural_workshop_nback_transfer_pilot \
  --neural-workshop /absolute/path/to/neural-workshop \
  --source-checkpoint /absolute/path/to/1back/checkpoint.pt \
  --source-bank /absolute/path/to/1back/AgentBrain.bank \
  --output-dir /tmp/neural-workshop-position2back
```

The first seed-51017 run learned and retained two-cell Position 2-Back, and all
causal controls failed. It also exposed negative transfer: the inherited
one-step address rewrote itself to the two-step slot, but stable mastery used
127 authenticated bits versus 94 for a uniform-fresh task file. Keep the
frozen controller and external-program mechanism, but do not claim that the
current address-file inheritance improves learning across memory depth.

`neural_workshop_recursive_transfer_pilot.py` addresses that failure with a
separately versioned recursive temporal interpreter. A migrated depth-one
artifact preserves the verified legacy behavior. Repeating its immutable row
means sequential function composition: the interpreter convolves relative
offsets, so a one-step `PREVIOUS` primitive composed twice resolves two-back.
Candidate depth is external program structure and never enters the controller
or event tensor. A bounded frontier tries progressively deeper compositions
and selects only from authenticated scalar outcomes.

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.neural_workshop_recursive_transfer_pilot \
  --neural-workshop /absolute/path/to/neural-workshop \
  --source-bank /absolute/path/to/1back/AgentBrain.bank --source-slot 4 \
  --output-dir /tmp/neural-workshop-recursive-position2back
```

In the seed-71017 live run, the migrated primitive retained 1-back at 100% for
three sessions. Depth one scored 0.326 on held-out 2-back; `PREVIOUS ∘
PREVIOUS` scored 0.969 without updates and retained at 0.971/0.971. Wrong-depth,
over-composed, memory-corrupted, and reversed controls scored 0.325, 0.475,
0.000, and 0.000. The recursive bank admitted both the primitive and verified
composition. Search used 78 verifier bits to find depth two; including the
failed depth-one candidate and stable retention costs 147 bits.

`neural_workshop_instruction_route_pilot.py` attacks that selection cost. A
second frozen encoder reads only the public header band; those events address
the program bank and never enter the one-source temporal comparator. After the
verified depth-one and composed files are bound to their visible headers,
held-out sessions retrieve the slot from the first public frame. `n_back`
remains verifier-private. Play-field context, a shuffled header, and the
wrong program are retained as controls.

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.neural_workshop_instruction_route_pilot \
  --neural-workshop /absolute/path/to/neural-workshop \
  --source-bank /absolute/path/to/recursive/AgentBrain.bank \
  --output-dir /tmp/neural-workshop-instruction-route
```

In the seed-81017 live run, the visible 1-back and 2-back headers were
distance `0.105` apart and identical across seeds. Verification retained the
source programs at `1.000/1.000/1.000` and `0.968/0.973/0.971`. Held-out
retrieval then selected the matching slot with propensity `1.0` on every
session: 1-back scored `1.000/1.000/1.000`, composed 2-back scored
`1.000/0.971/1.000`. Search used `0` verifier bits. Play-field context and a
shuffled header were unknown (`propensity 0.5`) and scored `0.195` and
`0.422`. The wrong program scored `0.326`. Controller, program, and replay
updates remained zero. This is a bounded routing result: the header crop is
frontend parameterization, not autonomous cue discovery.

The same encoder cannot safely generalize that route to a new cell count.
A 3-cell 2-back header and a 2-cell 3-back header both sit about `0.043`
from the trained 2-cell 2-back header, so nearest-neighbor would treat an
unseen depth as a cell-count nuisance. Exact match stays fail-closed.
`neural_workshop_instruction_header_transfer_pilot.py` rebinds the same
immutable files to the new public header and checks that 3-back remains
unknown.

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.neural_workshop_instruction_header_transfer_pilot \
  --neural-workshop /absolute/path/to/neural-workshop \
  --source-bank /absolute/path/to/instruction-route/AgentBrain.bank \
  --output-dir /tmp/neural-workshop-instruction-header-transfer
```

In the seed-91017 run, the 2-cell 1-back versus 2-back header distance was
`0.105`. The 3-cell 2-back shift and the 2-cell 3-back shift were both
`0.043`, and nearest-neighbor would have sent unseen 3-back to the 2-back
file. Exact match stayed unknown (`propensity 0.5`). Zero-shot 3-cell 2-back
therefore ran the 1-back fallback and scored `0.286`; 3-back scored `0.239`.
Zero-shot 3-cell 1-back scored `1.000` only because append order already
points at slot 0. Rebinding the same two files to the 3-cell headers, then
retrieving, scored `1.000/1.000/1.000` and `0.950/0.960/1.000`. The original
2-cell 2-back route retained `1.000/1.000/1.000`. After the rebind, 3-back
was still unknown (`0.262`). Search bits stayed `0`; no new program file was
created; controller, program, and replay updates stayed zero.

`neural_workshop_instruction_depth_growth_pilot.py` adds the missing depth
instead of stretching the 2-back route. It composes the same `PREVIOUS`
primitive a third time, verifies the child on live 3-back, and binds that
file to the public 3-back header. Four-back remains an unknown context.

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.neural_workshop_instruction_depth_growth_pilot \
  --neural-workshop /absolute/path/to/neural-workshop \
  --source-bank /absolute/path/to/instruction-route/AgentBrain.bank \
  --output-dir /tmp/neural-workshop-instruction-depth3
```

In the seed-93017 run, a 3-back header was unknown to the 2-program bank and
fell back to slot 0 at `0.388`. `PREVIOUS ∘ PREVIOUS ∘ PREVIOUS` then verified
at `0.968/0.971/1.000` with zero updates and entered slot 2. Held-out
retrieval selected that slot with propensity `1.0` and scored
`0.931/0.958/0.971`. Depth one and two retained `1.000` across three sessions
each. Wrong depth scored `0.375`, over-composition `0.304`, memory corruption
`0.000`, and an unseen 4-back header stayed unknown at `0.356`. Search bits
were `0`. The bank now holds three immutable files.

Instruction events now share the amodal bus with the play-field stream. The
frozen comparator binds only the play-field source, so the header cannot
pollute `PREVIOUS`. Same-slot header differences become a nuisance subspace:
an exact miss can still retrieve if the leftover residual after removing
those differences uniquely matches one file. Unknown headers then try
existing files by distance and length, and only then compose one deeper
step. `neural_workshop_autonomous_founding_pilot.py` measures that policy
against a matched fresh climb.

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.neural_workshop_autonomous_founding_pilot \
  --neural-workshop /absolute/path/to/neural-workshop \
  --source-bank /absolute/path/to/instruction-route/AgentBrain.bank \
  --output-dir /tmp/neural-workshop-autonomous-founding
```

`neural_workshop_sealed_frontier_pilot.py` discovers the one-step address
by outcome-only search, composes 2-back, misses 5-back at history 4, grows
history to 8 without changing relation weights, verifies 5-back, and runs
the same files on rendered audio. Dual N-Back keeps the frozen two-way
decoder and packs one match bit per source. Evidence is probation in
`session_records/brainworkshop_sealed_frontier_probation_2026-08-14/`.

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.neural_workshop_sealed_frontier_pilot \
  --neural-workshop /absolute/path/to/neural-workshop \
  --output-dir /tmp/neural-workshop-sealed-frontier
```

In the seed-95017 run, offset 0 mastered live 1-back at `1.000` (35 bits)
without a gifted primitive. Autonomous composition scored `1.000` on
2-back. Five-back missed at history 4, then scored `0.952` after growth.
The same primitive and two-step child scored `1.000` on rendered audio
1-back and 2-back. Fresh audio also found offset 0 first, so 1-back search
bits were tied; the transfer claim is execution on a new substrate, not a
cheaper discovery.

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.rendered_dual_nback_pilot \
  --report-out /tmp/rendered-dual-nback.json
```

`neural_workshop_dual_live_pilot.py` is the Neural Workshop Dual path.
Position is the public play-field crop. Audio is the queued stimulus
waveform (`audio_pcm`), not a letter ID. Each stream binds separately; the
frozen two-way decoder packs bits onto the two public ports. Letter IDs and
other privileged keys fail closed.

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.neural_workshop_dual_live_pilot \
  --neural-workshop /absolute/path/to/neural-workshop \
  --trials 60 --seed 98017
```

On seeds 98017 and 98117 the same frozen `PREVIOUS` composition scored
`1.000` on Dual 1-Back and Dual 2-Back. The wrong-depth control scored
`0.091` and `0.109`. Each stimulus produced one vision event and one audio
event. Controller, program, and replay updates were zero. Evidence:
`session_records/brainworkshop_neural_workshop_dual_live_2026-08-15/`.

`rendered_dual_transfer_pilot.py` and
`neural_workshop_dual_acquisition_pilot.py` start from a uniform address
file. Mixed Dual feedback is packed exact-match credit, not half-credit.
Rendered seeds 99017/99117 mastered Dual 1-Back at 47 and 94 bits, retained
`1.000`, and executed composed Dual 2-Back at `1.000`. Neural Workshop seeds
99117/99217 mastered at 95 and 49 bits, retained `1.000`, and executed
composed Dual 2-Back at `1.000` / `0.944`. Reward-shuffled, reversed, and
missing-history controls stayed below threshold. This is Dual acquisition
and composition, not a holdout promotion. Evidence:
`session_records/brainworkshop_dual_acquisition_2026-08-15/`.

In the seed-94017-v2 run, a warm 1-back/2-back bank rebound 3-cell 2-back in
one `try_existing` session at `1.000` (24 bits). New 2-cell 3-back still had
to fail the two existing files and compose: 117 warm bits versus 119 fresh,
a tie. After that compose, 3-cell 3-back retrieved by the same-slot
invariant at `0.900` with 20 bits versus 85 fresh, a `4.25×` fresh/warm
ratio. Source 1-back, 2-back, and 3-back then retrieved exactly and scored
`1.000/1.000/0.943`. Controller, program, and replay updates stayed zero.
After skip-shallower, seed 97017 first-time 3-back was 65 versus 127
(`1.95×`) and header transfer remained `3.46×`. That pair is development
probation.

`founding_promotion.py` freezes those gates and consumes a one-use holdout
on unused seeds 110017, 111017, and 112017. Header transfer retrieved by
the same-slot invariant at 23/20/26 warm bits versus 97/82/89 fresh
(`4.22×/4.10×/3.42×`). First-time 2-cell 3-back composed at 82/71/67
versus 116/119/125 (`1.41×/1.68×/1.87×`). Wrong-depth, missing-history,
and reversed-action controls stayed below threshold. Both records
re-evaluate as eligible against `holdout-ledger.jsonl`. Evidence:
`session_records/brainworkshop_founding_holdout_2026-08-15/`.

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.founding_promotion \
  --neural-workshop /absolute/path/to/neural-workshop \
  --claim-holdout
```

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

Header transfer and first-time depth invention are now holdout-promoted on
Neural Workshop. Dual I/O and Dual acquisition remain measured probation
on two substrates. Canonical Dual training is Neural Workshop, not desktop
screen capture. The gym constructor owns n-back, cells, trials, and mute.
There is no competing local Brain Workshop. Desktop ScreenCaptureKit remains
an optional human-parity I/O path against Neural Workshop's public window.
The architecture still does not claim autonomous general program induction,
unrestricted memory growth, or a complete executive ISA.
