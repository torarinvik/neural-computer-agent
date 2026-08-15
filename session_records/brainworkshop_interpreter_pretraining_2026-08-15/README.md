# The controller learns to interpret (2026-08-15)

Status: **milestone**, the first step of
`docs/DECISION_CONTROLLER_IS_THE_INTERPRETER.md` completed. A frozen controller
now executes an external program and reproduces a capability the leases
already verified. Nothing was admitted to `AgentBrain.bank`.

## The skill, and why it is task-free

`interpreter_pretraining.py` teaches fetch-decode-branch and nothing else:
given an event, an instruction, and a workspace summary, emit the intention
naming the operator that instruction calls for. An instruction carries two
handle fields; when the event matches the workspace the first names the
operator, otherwise the second does.

**Handles are redrawn at random every batch.** There is no fixed opcode
vocabulary to memorise, so the only thing that can generalise is the skill of
reading a field and choosing between two of them. No verifier, no reward, no
rendered stimulus, no rule enters the loop — the training signal is the
machine's own instruction-set semantics, exactly as `controller_pretraining.py`
taught a generic temporal relation before it.

## Results

Trained on 8 operators, 4000 steps, 2.9 seconds, **10,384 parameters**.

| Held-out handle vocabulary | Accuracy | Condition met | Condition unmet | Chance |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1.000 | 1.000 | 1.000 | 0.500 |
| 4 | 1.000 | 1.000 | 1.000 | 0.250 |
| 8 | 1.000 | 1.000 | 1.000 | 0.125 |
| 16 | 1.000 | 1.000 | 1.000 | 0.063 |
| 32 | 1.000 | 1.000 | 1.000 | 0.031 |

Every handle is freshly drawn. 16 and 32 are vocabularies four times the size
training ever used. Both halves of the condition are perfect, so the controller
is branching rather than copying one field.

## The milestone

| | Accuracy |
| --- | ---: |
| 1-back interpreted, **learned intentions**, frozen controller | **1.000** |
| Same program, teacher intentions | 1.000 |
| Value recorded by the 1-back lease | 1.000 |
| Untrained controller, before this work | 0.501 |

Then the invariant that the whole decision rests on, with a trained network in
the loop: ten operators invented **after** the controller was frozen, growing
the table from 6 rows to 16.

| | Result |
| --- | --- |
| Controller digest | **unchanged** |
| Interpreted accuracy | **1.000** |

A capability is produced by interpreting a program, the operator vocabulary
lives in the program rather than the network, and growing it costs the
controller nothing.

## What is still not done

- **The reference program is an experimenter's**, like the rule compiler before
  it. Nothing here learns or admits a program; this establishes that a frozen
  controller can *run* one.
- **The curated controller is untouched.** This is a separate 10k-parameter
  interpreter, not a retrained `temporal_controller_previous_event_seed1001`.
  Whether the curated controller is replaced or retired is still open, and
  `AGENTS.md`'s weight-reset terms apply when it is.
- **Only three operators are exercised** by 1-back: `emit_match`, `store`,
  `halt`. `jump` and conditional dispatch are trained and generalise on
  synthetic problems but are not yet used by a program against the verifier.
- **No accumulation curve.** The decision states its own falsifier: if
  interpretation does not make capability N+1 cheaper, it is a worse policy
  with extra steps. That measurement does not exist yet.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.interpreter_pretraining
```

About three seconds. The artifact is curated at
`artifacts/checkpoints/interpreter_controller_seed1001.pt`.
