# Decision: the controller is the interpreter

Taken 2026-08-15. This is normative. `AMODAL_N_TO_M_ARCHITECTURE.md` always
said it; the repository had quietly grown a second answer, and this file
settles which one is load-bearing.

## The decision

The frozen controller executes external programs. Programs are **data the
controller reads**, never Python control flow that decides an action on the
controller's behalf. One environment tick is many controller microsteps.

The consequence is immediate and unwelcome: the counter-machine bridge that
reached 18/18 on sampled rules
(`session_records/brainworkshop_counter_state_ceiling_2026-08-15/`) is **not
the path**. In it, presses came from a Python executor driven by clustered
frontend events while the controller's relation network and decoder never ran.
It stands as a ceiling — proof that the substrate suffices — and must be
re-derived through the controller before any of it counts as capability.

## What it costs, measured

The current frozen controller cannot host an interpreter:

| Interface | Now | Needed |
| --- | ---: | --- |
| Decoder head | **2 actions** | at least one intention per micro-operation |
| Intention width | 16 | unchanged is plausible |
| Addressable state | 4 history slots | an addressable store |
| Instruction rows held | 1 address + 1 prototype | a stream |

A minimal interpreter needs to advance, jump, read state, write state, emit,
and halt. A two-way decoder cannot express six distinct micro-operations, so
this decision **requires a new controller blueprint**. That is allowed and it
is not a free action. `AGENTS.md` already prescribes the terms: retain the
blueprint, reset the weights, and prove the change on the next held-out
learning curve rather than asserting it. A wider controller that does not earn
its width on that curve must be rolled back.

## Invariants that keep this general

These exist to stop the next year of work painting into a corner. Each one has
a specific failure it prevents.

1. **Instructions are opaque data, never opcodes in the decoder head.** If the
   decoder gets one output per instruction type, adding an instruction resizes
   the controller and every admitted program is invalidated. The stable
   abstraction is `CALL` over a content-addressed operator handle, as the
   architecture doc already specifies. *Test: adding an operator must not
   change the controller digest.*
2. **Workspace is addressable external memory, not fixed registers.** Capacity
   grows in the bank, not in the network. The learned relative address already
   exists (`relative_address_logits`); generalise it from a four-tick history
   to a store. *Test: increasing working memory must not resize the
   controller.*
3. **The input contract is the learned event vector.** Today's one-hot cluster
   quantisation is expedient scaffolding for a four-position alphabet. It must
   not become the interface, or the agent inherits an alphabet it cannot
   outgrow. *Test: a frontend whose events do not cluster cleanly must still
   run.*
4. **Microstep budgets are explicit and fail closed.** No unbounded
   interpretation, and budget exhaustion is a recorded status, never a silent
   default action.
5. **Internal intentions and device actions stay separate.** Adding a decoder
   or a device must not resize the controller — the existing N-to-M invariant,
   now also applied to the interpreter's own micro-operations.
6. **Nothing in the deployed path may consult a rule, a task identity, or an
   oracle.** `counter_state_programs.compile_rule` is a diagnostic that reads
   the rule directly. It must never be imported by the agent path, and no
   compiled program may be admitted. *Test: the agent path does not import the
   compiler.*

## First milestone: behaviour-preserving re-derivation

Before anything new is claimed, take one capability that is already verified —
1-back, or `onset` — and reproduce it through the interpreter path: the
program is data, the controller executes microsteps, the accuracy matches the
recorded lease, the controller stays frozen, and the bank is untouched.
`AGENTS.md` requires exactly this ordering: refactor behaviour-preservingly
before claiming the new boundary.

Acceptance: held-out accuracy equal to the recorded lease, zero controller
updates, the microstep budget respected on every tick, and the old direct path
marked legacy rather than left as a second answer.

## What would falsify this decision

If a controller large enough to interpret cannot learn to interpret
sample-efficiently, the bet has failed: an interpreter that costs more than
the policy it replaces is a worse policy with extra steps. The test is the
accumulation curve — does capability N+1 get cheaper as the library grows? If
interpretation does not bend that curve, the honest conclusion is that the
external-program story is decoration and the capability lives in the network
after all.

## The curve, measured (2026-08-15)

It exists now, for the pre-interpreter path:
`session_records/brainworkshop_accumulation_curve_2026-08-15/`. Eighteen
sampled rules learned in sequence, once with the library growing and once with
it restored before every rule.

**The curve bends the wrong way.** The growing arm spent 1432 verifier
episodes against the control's 886 -- a cost ratio of **1.616** -- and gated
exactly the same 7 of 18 rules. Reuse does happen and is worth 2.3x to 4.8x
where it fires, but it fires only as `retrieve` of an exact behavioural
duplicate: across the whole curriculum there were zero composes, zero inverts
of a learned file, and zero ANDs over a learned file.

The loss is arithmetic rather than noise. Proposals run in a fixed order, so
growing the library from 3 files to 7 grew the pre-invention prefix from 18
proposals to 69. Eleven rules are inexpressible in this family and therefore
execute the whole list every time (120 episodes against the control's 67). The
tax is paid eleven times, the saving collected three.

This does not falsify the decision, because the interpreter was not in either
arm. It does three other things:

1. It sets the number an interpreter has to beat: **1.616, with reuse confined
   to exact duplicates.**
2. It relocates the bottleneck. Not representation -- the counter bridge
   closed that at 18/18. Not the controller -- it never ran differently
   between arms. The searcher has no way to decide what to try, so every
   admitted file becomes one more thing to execute blindly.
3. It reorders the work. An interpreter makes programs longer and the search
   space larger. Building more of one before the proposer exists adds load to
   the component the curve just identified as the failure.

### The searcher was fixed; the curve did not straighten

`PROPOSERS.md` in the same record runs two better proposers on the identical
curriculum. Collapsing observationally equivalent candidates takes the
curriculum from 1432 verifier episodes to 390 and the cost ratio from 1.616 to
1.032; recovering the target behaviour from per-step reward and ranking
candidates offline takes it to **125 episodes, 11.5x**. All three arms solve
the same seven rules, because both filters are lossless.

The number an interpreter has to beat is therefore **125 episodes**, not 1432,
and the honest reading of the rest is worse for the library rather than better.
With blind enumeration gone, **zero composes, zero inverts of a learned file
and zero gating ANDs over a learned file** remain, on all three arms. Every
winner is still a `retrieve` of an exact behavioural duplicate or a fresh
`invent`. The searcher was never what stood between this library and
composition.

What remains of the library's cost is now exactly one untestable proposal per
admitted file: an `and:slot` whose prototype is acquired rather than given
cannot be ruled out offline and must be trained against the verifier first.
That is a property of the proposal grammar, not of the bank.
