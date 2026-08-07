# External register read/execute snapshot: promoted composition frontier

Date: 2026-08-07
Status: promoted narrow result
Schema: `neural-computer.external-register-read-execute.v1`

The external register now exposes two explicit phases. `observe_register()`
ingests the standardized learned event, opaque feedback, and controller
intention into durable external working state. `read_execute_register()` runs
the selected opaque instruction chain against a transient snapshot and keeps
the execution result out of the durable state. The legacy in-place
`step_register()` path remains available for compatibility, while `step()` and
the rendered composition harness use the read/execute path.

This boundary was tested on valid rendered sequence-memory events with a
frozen parent controller, paired counterfactual credit, zero replay, exact
reload, checksum corruption, frozen-parent, reward-shuffled, and
missing-evidence controls.

## Two-instruction regression

The promoted reverse -> complement result was rerun at the original stable-
prefix resolution on seeds 69316 and 69317:

| Measure | Seed 69316 | Seed 69317 |
| --- | ---: | ---: |
| Stable inherited composition bits | 4,096 | 4,096 |
| Stable fresh composition bits | 8,192 | 8,192 |
| Fresh-over-inherited transfer | 2.0x | 2.0x |
| Reverse retention | 0.9648 | 1.0000 |
| Complement | 0.9219 | 0.9727 |
| Reward-shuffled composition | 0.4805 | 0.7188 |
| Missing evidence | 0.5000 | 0.5000 |

Both seeds passed the promotion gates. This is a behavior-preserving
regression of the earlier narrow result under the new execution contract.

## Three-instruction growth

The reverse -> complement -> rotate program was then acquired sequentially
without replay while the parent and interpreter remained frozen:

| Measure | Seed 69316 | Seed 69317 |
| --- | ---: | ---: |
| Stable inherited triple bits | 8,192 | 4,096 |
| Stable fresh triple bits | 16,384 | 12,288 |
| Fresh-over-inherited transfer | 2.0x | 3.0x |
| Reverse retention | 0.9805 | 1.0000 |
| Complement retention | 0.8828 | 0.9102 |
| Rotate retention | 0.8359 | 0.9063 |
| Triple composition | 0.9063 | 0.9766 |
| Reversed-order composition | 0.8945 | 0.9922 |
| Reward-shuffled composition | 0.6211 | 0.4570 |
| Missing evidence | 0.5000 | 0.5000 |

All registered promotion gates passed on both seeds. The result promotes a
three-instruction compositional growth frontier, not arbitrary program
induction, unrestricted memory growth, natural-language or speech capability,
or general continual learning without catastrophic forgetting.

The decisive mechanism is the state boundary: learned observations persist,
but repeated instruction execution no longer contaminates the evidence store.
The earlier combined in-place path failed depth-three composition; the
snapshot path recovered reliable order-sensitive composition while preserving
the two-instruction transfer gain.
