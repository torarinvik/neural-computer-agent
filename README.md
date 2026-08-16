# Neural Computer Agent

A research system for learning reusable computation from amodal events and
deterministic outcome signals.

Audited model checkpoints are stored in the private Hugging Face repository:
<https://huggingface.co/torarin87/neural-computer-agent>.

The whole system's frontends receive rendered vision/audio/text streams. The
controller receives only their learned event representations, its own opaque
actions, its own latent state and memory, and scalar verifier outcomes. Neither
the controller nor its adapters receive game state, coordinates, semantic task
labels, rule IDs, correct-action labels, English chain-of-thought, or
counterfactual labels for actions it did not attempt.

## Canonical architecture

**The normative architecture and machine boundary are defined by
[docs/AMODAL_N_TO_M_ARCHITECTURE.md](docs/AMODAL_N_TO_M_ARCHITECTURE.md).**
[docs/DYNAMIC_BRAIN_ARCHITECTURE.md](docs/DYNAMIC_BRAIN_ARCHITECTURE.md) is the
compatible vision and knowledge-placement rationale; it does not override the
normative interface. Its non-negotiable storage rule is retained: durable
skills and information live in the external, unboundedly growing memory bank,
never in the fixed controller, encoder, or decoder weights. Any result that
stores task content in those weights is a recorded transitional violation.

The target is an **amodal N-to-M neural computer**:
The repository is intentionally centered on one architecture:

```text
N learned encoders -> amodal event bus -> one fixed controller/workspace
                   -> intention bus -> M learned decoders
                                      |
                                      +-> external memory and programs
```

The controller never consumes raw modality formats and never emits device
protocols. Encoders, controller, memory, programs, and decoders are separately
versioned and replaceable. Adding an input or output does not resize the
controller or add a modality-specific reasoning branch.

## Current design

The fixed machine is an interpreter and executive, not a store of task
policies. Its intended kernel is:

- `RECEIVE`: consume a typed event from the input bus;
- `EMIT`: publish a typed intention to the output bus;
- `WAIT` and `HALT`: control interaction time;
- `READ`, `WRITE`, and `COPY`: manipulate typed workspace handles;
- `CALL`: apply a verified external operator or program;
- `SEQUENCE`, `BRANCH`, `LOOP`, and `RETURN`: compose computation.

Reward is not a cognitive opcode. A trusted verifier adapter emits a causally
attributed outcome event through the same input boundary. Missing reward,
observed zero reward, and negative reward are distinct states.

`LiveInputInstruction` makes that boundary runtime-variable. Any number of
sensory or verifier adapters may be attached behind the same fixed-width input
ABI; learned event streams are concatenated without averaging, while attributed
outcomes remain exact-once transport records. `QueuedOutcomeInputDevice` is the
minimal reward adapter: an environment submits the agent's own action receipt,
a scalar outcome, presence/confidence, and observation time. It supplies no
task ID, program ID, correct action, or labels for actions that were not tried.
External program learners and route ledgers can subscribe to these resolved
outcome inputs during the same tick.

Interaction runs as a monotonic cognitive tick:

```text
RECEIVE learned events -> resolve causal outcomes -> bounded online update
                       -> controller/program step -> EMIT intention
```

The same causal loop supports an accelerated virtual clock during research and
a real clock for screen, audio, keyboard, pointer, and text devices. Raw FPS,
learned-event rate, cognitive ticks, optimizer updates, and actions are
accounted separately.

Boolean operations may be installed as verified library macros, but are not
the foundational ISA. The kernel controls information flow, time, memory, and
program execution; learned domain transformations live in the growing external
library.

See [the normative architecture](docs/AMODAL_N_TO_M_ARCHITECTURE.md) and the
[policy-free learning contract](docs/POLICY_FREE_CONTINUAL_LEARNING.md).

## Current empirical frontier

The canonical Neural Workshop path demonstrates bounded append-only external
working-memory growth:

- a frozen controller and event frontend first support an external `nback16`
  compute file;
- a new external file learns `nback32` without updating or replaying the old
  file;
- two seeds reached `1.0000` target accuracy while retaining the source at
  `1.0000` and `0.9992`;
- missing-history, corrupted-history, action-shuffled, and shuffled-training
  controls failed the mastery gate;
- replayed examples were zero.

This is a strong bounded working-memory result, not general continual learning.
The next scientific target is faster acquisition and reusable computation,
measured against a matched fresh learner—not merely final n-back accuracy.

The first live rendered rung now consumes clean-room RGB frames and audio
waveforms on the same batch-one tick runtime. Vision and audio remain separately
bound through opaque source keys and per-source temporal histories. A seed-17
diagnostic reached stable 0.80 accuracy after 230 vision outcomes, 115 audio
outcomes, and 920 outcomes for exact dual-stream actions, with zero replay.
These are mechanistic signals, not mastery claims.

The physical human-parity rung captures the public Neural Workshop window at
12 Hz, segments spatial position-stimulus onsets, emits ordinary keypresses,
and learns only from explicit feedback colors a human sees. A controller
pretrained across variable visual frontends is frozen during task acquisition;
only a fresh uniform external temporal-address program updates. In the promoted
2-cell Position 1-Back run, 86 unique public outcomes produced 86 program
updates, 0 controller updates, and 0 replay. The first full rolling-44 window
scored `0.8864`, every later window remained above `0.80`, and the final window
was `1.0000`. Rendered causal controls passed across 32 seeds, and the matched
fresh end-to-end learner required 2.32x as many median verifier bits among
successful runs. Neural Workshop Dual 1-Back acquisition and composed Dual
2-Back execution are now holdout-promoted on public pixels plus public PCM.
Desktop ScreenCaptureKit Dual remains optional human-parity I/O, not a
trainer.

The most recent eviction-transfer audit is also important negative evidence:
artifact storage, integrity, and lifecycle work, but the learned maintenance
policy did not transfer reliably to a held-out compute family. The blueprint is
retained; inherited maintenance weights are not promoted.

Verified temporal-address programs now have a real external-bank lifecycle.
A provisional program is learned outside the live bank, evaluated from ordered
public lifetime scores, and admitted only after a stable verifier prefix. The
durable entry is an immutable instruction tensor bound to the frozen controller
digest; optimizer state is not executable memory. Selection is learned in a
separate memory-side ledger from opaque event contexts, attempted program slots,
and scalar outcomes. The bank records exact selection propensities, rejects
tampered files, and leaves its digest unchanged after failed admission. This is
working infrastructure, not yet evidence that the physical agent can
discriminate several Neural Workshop rules: that requires a public visible rule
cue to be encoded as an ordinary learned event and a multi-program control.

The first physical lifecycle replication is now curated. A fresh uniform
two-cell Position 1-Back program earned admission after six lifetimes and 37
public outcomes. Four subsequent sessions started from a fresh controller
instance, withheld actions for three live sensory events, retrieved slot 0 from
the bank, and then executed the immutable program. Retrieval produced 13/15
positive public outcomes with zero controller updates, zero program updates,
and zero replay; only the external reward-route ledger changed. This validates
admission, persistence, retrieval, and frozen execution for one program. It
does not yet validate discrimination among multiple programs.

The external-executive path now closes the remaining transport gap: an admitted
`AgentBrain.bank` artifact can be saved, reloaded, and attached to the live tick
runtime through a sealed, frozen executive machine and a replaceable decoder.
The bounded diagnostic executes the same generic temporal-equality skill at
1-back and 2-back with `1.0000` accuracy over 60-step lifetimes, while keeping
controller and executive-program updates at zero. This proves durable
bank-backed live execution and WAIT-safe input pacing; autonomous skill
selection and skill learning remain the next bottlenecks.

The next bounded rung closes skill selection without unfreezing the controller:
`ExternalExecutiveRouterLiveMachine` selects between two reloaded temporal
equality artifacts from a visible mode cue encoded by the ordinary frozen event
frontend. Across 16 alternating training lifetimes (136 unique verifier bits),
the route ledger learns cue-specific preferred slots from one aggregate outcome
per lifetime, then reaches `1.0000` on held-out 1-back and 2-back lifetimes
(51 additional eligible bits across three lifetimes per route). Controller,
decoder, and executive-program updates remain zero; route updates are 22 and
replay is zero. This qualifies a
bounded opaque multi-skill live route, not open-ended route discovery: the cue
must be visible, the banked skills must already be verified, and the route
encoder is still a replaceable learned-event adapter. The lifetime aggregate is
important—per-action route updates can promote a wrong skill from a lucky
partial streak.

The route now also survives a nonstationary rule change. After a checksummed
bank reload, changing the private verifier from 1-back to 2-back behind the
same visible cue produced two failing slot-0 lifetimes, then three consecutive
slot-1 lifetimes at `1.0000`. The old slot remained perfect when forced on
1-back, and a cue-shuffled control fell to `0.3750`. This qualifies bounded
same-cue route reversal and immutable-skill retention; it does not yet provide
autonomous program induction or physical desktop deployment.

The router now also generalizes across held-out learned-event contexts. After
bank reload, six previously unseen cue perturbations selected the correct slot
in alternating 1-back/2-back episodes and scored `1.0000` on all 51 eligible
bits; exact variant rows were created only after their first outcome. This
closes the former behavior-probability/generalization mismatch, but remains a
bounded representation-radius result rather than semantic extrapolation.

Curated evidence is under `session_records/`. Historical experiments, obsolete
checkpoints, and superseded session dumps were removed from the working tree;
they remain recoverable from Git history.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/neural_computer/` | Production amodal runtime, memory, interpreter, and lifecycle contracts |
| `experiments/brainworkshop_canonical/` | N-back frontier; Neural Workshop is the only gym |
| `experiments/recipe_expressibility/` | Program composition and control-flow audits |
| `tests/` | Production and retained frontier tests |
| `session_records/` | Curated promoted evidence and decisive current rejections |
| `artifacts/` | Manifest-only home for future curated checkpoints |

## Development

```bash
./scripts/test_canonical.sh
./scripts/lint_canonical.sh
```

Run the promoted bounded n-back growth audit with:

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.external_compute_append_only_depth_growth
```

Every promoted experiment must report unique verifier bits, logical lifetimes,
optimizer updates, replay, time, stable bits-to-threshold, primitive retention,
and transfer against a fresh learner when transfer is claimed.
