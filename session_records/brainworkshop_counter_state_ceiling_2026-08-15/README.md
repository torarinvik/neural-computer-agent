# A program family with state clears every rule (2026-08-15)

Status: **diagnostic**. Programs are compiled by an experimenter's oracle from
rules the learner never sees, exactly as the enumeration ceiling was computed.
Nothing was searched, nothing was learned, nothing was admitted, and
`AgentBrain.bank` was not touched.

## The bridge

`control_flow.py` has been a two-counter machine with a fail-closed executor
since before this work, and nothing connected it to the rendered stream.
`counter_state_programs.py` is that connection, and it fixes an interface
rather than a rule class:

- counter 0 is the **press**, read after halt, cleared before each tick;
- counters 1..k are **input channels**, one per event cluster discovered from
  observation, set one-hot to the current event's nearest cluster;
- every later counter is **persistent working state**, carried across ticks and
  never touched by the runtime.

The clusters come from `prototype_templates`, which discovers them by distance
from the learner's own frontend — no alphabet size, symbol label, or verifier
state involved. A program may use the working counters however it likes.

## Results

Compiled programs executed against the real environment through the real
bounded executor, 448 steps, every one halting inside its step budget:

| Rule states | Temporal family ceiling | Counter family | Instructions | Counters | log10 search space |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000 | 1.000 | 21-22 | 7 | 50-53 |
| 2 | 0.790-0.895 | **1.000** | 43 | 9 | 123 |
| 3 | 0.703-0.804 | **1.000** | 63-66 | 11 | 196-207 |
| 4 | 0.580-0.804 | **1.000** | 81-85 | 13 | 268-283 |
| 5 | 0.710-0.766 | **1.000** | 106-108 | 15 | 370-378 |
| 6 | 0.629-0.748 | **1.000** | 124-130 | 17 | 448-472 |

**18 / 18** sampled rules at `1.000`, against 7/18 and a mean ceiling of
`0.786` for the temporal family. The four hand-written rules also compile, and
so does 2-back at 16 states and 321 instructions — a rule the temporal family
could only reach as a composed depth-2 child.

The expressiveness gap is closed. Every rule in the class is now representable
by a program this repository can already execute, admit, compose, and splice.

## What it costs: search is now the entire problem

The programs are 21 to 130 instructions long. Enumerating programs of that
length over this instruction basis means sifting **10^50 to 10^472**
candidates. The audit put enumeration's practical reach at length 6-8, around
10^9 to 10^11.

So the bottleneck has moved, and moved decisively. Before this bridge, 11 of 18
rules had no representation and a better proposer was worth nothing on them.
Now every rule has a representation and **no enumeration will ever find one**.
This is the point at which a proposer that infers structure from evidence, and
chooses what to test by expected information, stops being an optimisation and
becomes the only remaining route.

## What this does not do

- **Nothing is learned.** The compiler is an oracle over the rule class, used
  to establish a ceiling. It is never given to the agent, and no compiled
  program was admitted.
- **The neural controller is not in this path.** Presses come from the counter
  program driven by clustered frontend events; the controller's relation
  network and decoder are bypassed entirely. This measures whether the
  *substrate* suffices, and it does. An integrated design still has to decide
  what the controller's role is when the program carries the decision — that is
  a live architectural question, not a settled one.
- **Cluster quantisation is a lossy interface.** It works here because the
  stimuli are four well-separated positions. A frontend whose events do not
  cluster cleanly would need a different input contract.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.counter_state_programs
```

About one second.
