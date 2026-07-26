# Third-generation compounding: verified five-action hierarchy

## Question

Can the verified four-action hierarchy be reused as one opaque option so that
the controller learns a genuinely new fifth action with fewer verified
outcomes than a flat five-action reset, without forgetting old skills?

The answer is **yes under paired population exploration**. Two independently
trained routers passed the complete causal, statistical, retention, and
persistence audit.

## Setup and accounting

The new router makes one binary decision:

1. invoke the complete, frozen four-action hierarchy; or
2. execute the fifth physical read.

The flat control learns all five physical actions from zero. Inputs are eleven
generic rank/utility statistics; there is no task ID, semantic action label,
correct-action hook, or game-state hook.

The successful training signal is generated entirely from experience. Temporary
clones try the available alternatives on the same context and the verifier
returns scalar outcomes. The composed learner regresses the observed advantage
`fifth outcome - inherited-option outcome`; the flat learner regresses all five
observed utilities. Every observed outcome is charged:

- composition: 2 verifier bits per context;
- flat reset: 5 verifier bits per context.

Evaluation always uses untouched contexts and the real action costs.

## Localization before the successful fork

The progression was deliberately small:

- Seed 8071, seven features: both arms reached the gate at 960 bits; no strict
  speed win.
- Replay 8: failed.
- A discarded-weight probe showed that the relation was decodable and improved
  with generic feature rank:

| Features | Held-out preference accuracy | Held-out utility |
|---:|---:|---:|
| 7 | `89.93%` | `0.86517` |
| 9 | `92.23%` | `0.88004` |
| 11 | `93.06%` | `0.88202` |

- Eleven features with ordinary randomized bandit feedback still failed.
- A fifth-action cost curriculum also failed.
- Paired winner classification collapsed to the majority “reuse old option”
  shortcut.
- Preserving outcome magnitude as a verified advantage eliminated that
  shortcut. The first 16-step seed passed at 1,680 bits, but its immediate
  replication failed, so it was not promoted.
- A 64-step stability check passed at two learning rates. The `.003`
  configuration was then frozen before fresh-seed replication.

The result therefore localizes the frontier: the representation already
contained the relation; sparse single-action feedback used it unreliably.
Counterfactual experience shared across temporary clones converted the latent
relation into a stable action rule.

## Pre-registered replications

Both frozen-configuration replications passed:

| Seed | Old hierarchy | Final composition | Composition stable target | Flat stable target | Transfer ratio |
|---:|---:|---:|---:|---:|---:|
| 8083 | `0.79196` | `0.83149` | 3,360 bits | 15,600 bits | `4.64x` |
| 8084 | `0.76627` | `0.81010` | 3,600 bits | 5,400 bits | `1.50x` |

Target means inherited held-out utility plus two percentage points. “Stable”
is the first measured threshold after which every later checkpoint also
passes. Both composed learners retained the gain at the final checkpoint.

## Independent adversarial audits

Each trained router was audited on eight new 2,040-context streams, plus a
separate 2,400-record randomized-action confirmation set.

| Audit | Seed 8083 router | Seed 8084 router |
|---|---:|---:|
| inherited hierarchy | `0.78754` | `0.78258` |
| five-action composition | `0.82701` | `0.82962` |
| absolute gain | `+3.95` points | `+4.70` points |
| shuffled router features | `0.74274` | `0.73534` |
| reversed router decision | `0.49241` | `0.48570` |
| randomized improvement estimate | `+4.83` points | `+2.93` points |
| randomized lower 95% bound | `+3.52` points | `+1.49` points |
| all eight streams improve | yes | yes |
| binary mapping retained | yes | yes |
| four-rule task retained | yes | yes |

The shuffle and reversal results rule out constant fifth-action usage and
feature-independent reward hacking. The independent randomized estimator rules
out evaluation-only selection.

## Persistent lineage

Both routers were committed as children of verified four-action skill
`498db106964cd1eb5383`.

- Seed 8083 child: `31c77292b2935bf0ce30`
- Seed 8084 child: `3b9f24e1b5185256d0b1`

For both stores:

- a fresh process reproduced decisions exactly;
- appended-byte corruption was detected;
- corruption of the child did not damage the parent;
- the full earlier lineage remained loadable.

## Conclusion

This is the third consecutive verified generation of hierarchical option
growth. The controller can now preserve a four-action skill as a reusable
unit, explore only the old-vs-new frontier, and acquire a fifth action between
1.5 and 4.64 times sooner than flat relearning under exact outcome accounting.

The key new mechanism is **paired population advantage learning**: temporary
clones gather counterfactual experience, but the persistent agent remains one
controller with external verified skill memory. This is a concrete,
zero-hand-label route to compounding sample efficiency.
