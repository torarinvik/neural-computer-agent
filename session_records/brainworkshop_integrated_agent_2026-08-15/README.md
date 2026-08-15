# The agent keeps what it learns (2026-08-15)

Status: **held out and admitted.** Three replicates on the previously unspent
`integrated_agent_holdout` seed block. Twenty induced programs persist under
`artifacts/checkpoints/*.library` and are registered in the curated manifest.
`AgentBrain.bank` is unchanged at `07319eb1`.

Every record in this session so far ends with the same two sentences: nothing
is admitted, and the controller executes none of this. They were honest, and
they were the reason none of it described an agent. Feedback inversion,
noise-tolerant fitting, episode segmentation, class escalation and regime
tracking each worked, each was measured, and each was thrown away.

This is the loop that does not throw it away.

## What the agent does

One task at a time, reading only rendered stimuli and its own scalar rewards:

1. **establish an alphabet, once**, by clustering the frontend's own events;
2. **buy one short episode** and invert its per-step reward into the target;
3. **ask the library** whether anything it already holds explains that;
4. if not, **buy another rung** of evidence and ask again;
5. once there is enough, **fit** the machine that disagrees least;
6. **compile** it to a counter program and **run it in the environment**;
7. **confirm** on full-length episodes it never learned from;
8. **admit** it, unless the library already presses that way.

## Why the ladder, and not a fixed budget

The obvious design does not work, and it is worth being exact about why.

Recognition after a *fixed* evidence budget saves nothing. The agent still buys
every episode and still reads every label; all it avoids is the search at the
end. That costs CPU, and the objective is written in experience.

So the budget is not fixed. A task the library covers stops at the first rung.
A task it does not cover walks to the budget induction needs and pays what a
fresh agent pays. That is what makes the two arms differ in the only currency
that matters.

## Result

Three replicates, 24 tasks each, drawn with repetition from a pool of six rules
spread across state counts one to six. **No task in any replicate is solvable
by a constant press-or-don't policy**, which is checked and reported rather
than assumed.

| | growing | control |
| --- | ---: | ---: |
| Tasks solved | **72/72** | 72/72 |
| Probe episodes to recognise | **2.67** | -- |
| Probe episodes to induce | 10.67 | 10.67 |
| Acquisition evidence, mean ratio | **0.487** | 1.0 |
| Acquisition evidence, worst replicate | **0.550** | 1.0 |
| False recognitions | **0** | -- |

**A task the library already covers costs 2.67 short episodes instead of
10.67** -- four times less evidence -- and the library never once adopted a
program for the wrong task.

On the already-consumed development seed the same measurement gives 0.358. The
holdout is worse, which is the direction that makes it worth reporting.

## What was kept

Twenty programs across three replicates, written as append-only checksummed
`.library` files, reloaded from disk after the run and verified to match what
the loop said it admitted. The store has no capacity constant, no eviction and
no slot reuse, so admitting capability N+1 cannot damage capability N -- and
its digest covers every record in order, so a load that succeeds is a proof
that nothing earlier changed.

Admission is by compression, in the only currency this family has: a candidate
that presses identically to an existing record on the canonical signature
stream is refused, on the index alone, without executing anything.

## Controls

**Reward-shuffled.** Permuting each trace's labels leaves the label marginal
exactly where it was and destroys only the relation between symbol and press.
Under it the agent solves **0/72** and admits **nothing**, after spending the
full ladder on every task.

**Matched control arm.** The same stream, the same rules, the same order, the
same seeds; the library is discarded before every task.

**Constant-policy baseline.** Reported per task, so a rule that a fixed policy
already clears cannot be counted as evidence for induction. None were.

## Four things that were wrong, and the measurements that said so

**Every task was inventing a private alphabet.** `cluster_events` is greedy and
first-come: a cluster's index is the order it first appeared. Rediscovering the
alphabet per task makes symbol 2 in one task a different stimulus from symbol 2
in the next, so a stored program means nothing when retrieved -- every
recognition would have been a coincidence. The agent now establishes one
alphabet from one observation pass and speaks it thereafter.

**Recognition asked the wrong question.** `control_below_threshold_report` asks
whether competence can be *ruled out*, and over sixteen labels almost nothing
can be ruled out either way. A wrong program was adopted and reproduced at
**0.73**. Adoption needs evidence *for* competence: a policy sitting exactly at
the gate must be unable to look this good more than one time in a hundred. Over
sixteen labels even a perfect program cannot clear that, which is correct, and
over thirty-two it can.

**So did confirmation.** The same test, the same direction, one stage later: a
machine fitted to *shuffled* feedback cleared the gate at **0.814** against a
threshold of 0.8. That is exactly the near-miss the lease machinery was built
to refuse. Both stages now use one test in one direction, and the shuffled arm
went from 1 solved to 0.

**Neighbouring tasks shared their stimulus streams.** Probe episodes are drawn
at `seed + 1000 + index`, and a per-task stride of ten made tasks four apart
draw literally the same streams. They were still scored by their own rules, so
nothing was wrong -- but they were not independent draws and the record would
have said they were. The stride is now ten thousand.

## What is honestly weak

**Verification dominates the total.** Acquisition falls to 0.49; total evidence
only falls to 0.78, because confirming a candidate costs two full episodes
whether it was recognised or induced. The library makes *learning* cheaper and
does nothing for *proving*. A recognised program arrives with prior evidence
behind it and is re-confirmed from scratch anyway.

**Hard rules are solved by machines of the wrong size.** At the budget the
ladder stops at, a five-state rule was fitted by an eight-state machine and a
six-state rule by a three-state one. Both clear the gate on held-out episodes,
so the agent is right to keep them, but they are approximations rather than
identifications -- and approximate programs are recognised less reliably, which
cost one replicate two duplicate admissions for tasks it already had.

**Recognition is behavioural, not semantic.** A record is adopted because it
predicts the evidence, not because anything establishes it is the same rule.
Two rules that agree over sixty-four labels and diverge later would be confused,
and only confirmation would catch it.

**The pool is six rules over one alphabet and one frontend.** Nothing here
shows the mechanism survives a different modality, a wider alphabet, or noisier
stimuli.

**The controller still executes none of this.** Presses come from the counter
executor. `DECISION_CONTROLLER_IS_THE_INTERPRETER.md` remains untested by this
record, and the accumulation curve now bends without it.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.integrated_agent
```

About thirty seconds on the development seed. The holdout, which writes the
libraries and must not be re-run against a spent block:

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.integrated_holdout
```

About one hundred seconds.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_integrated_agent.py tests/test_induced_library.py -q
```
