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

**The authoritative statement of the end goal and of where every kind of
knowledge is allowed to live is
[docs/DYNAMIC_BRAIN_ARCHITECTURE.md](docs/DYNAMIC_BRAIN_ARCHITECTURE.md).**
Its non-negotiable rule: skills and information are stored in the external,
unboundedly growing memory bank — never burned into the weights of the
controller, the encoders, or the decoders, which all stay fixed-size. Any
result that stores skill in weights is a recorded transitional violation.

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

Boolean operations may be installed as verified library macros, but are not
the foundational ISA. The kernel controls information flow, time, memory, and
program execution; learned domain transformations live in the growing external
library.

See [the normative architecture](docs/AMODAL_N_TO_M_ARCHITECTURE.md) and the
[policy-free learning contract](docs/POLICY_FREE_CONTINUAL_LEARNING.md).

## Current empirical frontier

The canonical Brain Workshop path demonstrates bounded append-only external
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

The most recent eviction-transfer audit is also important negative evidence:
artifact storage, integrity, and lifecycle work, but the learned maintenance
policy did not transfer reliably to a held-out compute family. The blueprint is
retained; inherited maintenance weights are not promoted.

Curated evidence is under `session_records/`. Historical experiments, obsolete
checkpoints, and superseded session dumps were removed from the working tree;
they remain recoverable from Git history.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/neural_computer/` | Production amodal runtime, memory, interpreter, and lifecycle contracts |
| `experiments/brainworkshop_canonical/` | Current n-back and working-memory frontier |
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
